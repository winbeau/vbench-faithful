"""Batch independent greedy model requests without moving task parsing/scoring."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def prepared_router_generation(router, task, users, *, max_new_tokens=96, batch_size=32):
    """Keep the frozen router's predict/postprocessing, substituting raw texts.

    Uses AdapterRouter's exact chat template and generation settings. Request
    identity includes the complete user text and token budget; no approximate
    prompt matching, truncation or cross-adapter reuse is allowed. The original
    method and tokenizer state are restored even if downstream parsing fails.
    """
    import torch

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch and token limits must be positive")
    users = list(dict.fromkeys(users))
    original = router.generate
    padding = router.tokenizer.padding_side
    outputs = {}
    router.route(task)
    try:
        router.tokenizer.padding_side = "left"
        for start in range(0, len(users), batch_size):
            chosen = users[start:start + batch_size]
            texts = [router.build_prompt(user) for user in chosen]
            inputs = router.tokenizer(texts, return_tensors="pt", padding=True)
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(router.model.device)
            with torch.no_grad():
                generated = router.model.generate(**inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, num_beams=1, temperature=None, top_p=None,
                    pad_token_id=router.tokenizer.pad_token_id,
                    eos_token_id=router.tokenizer.eos_token_id)
            width = inputs["input_ids"].shape[1]
            raw = [router.tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                   for row in generated]
            if len(raw) != len(chosen):
                raise ValueError("Model changed text request coverage")
            outputs.update(zip(chosen, raw))
        router.tokenizer.padding_side = padding

        def replay(user, *, max_new_tokens=256):
            if router.active_task != task or max_new_tokens != token_budget or user not in outputs:
                raise ValueError("Prepared generation does not match task, text and token budget")
            return outputs[user]

        token_budget = max_new_tokens
        router.generate = replay
        yield
    finally:
        router.generate = original
        router.tokenizer.padding_side = padding
