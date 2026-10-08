"""Text batching must preserve requests, token slicing and frozen parsing calls."""
from types import SimpleNamespace

import pytest
import torch

from vbench_audit_models.batched_text import prepared_router_generation


class Encoding(dict):
    def to(self, device):
        return self


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        assert self.padding_side == "left"
        values = [[ord(c) for c in text] for text in texts]
        width = max(map(len, values))
        ids = torch.tensor([[0] * (width - len(v)) + v for v in values])
        return Encoding(input_ids=ids, attention_mask=ids.ne(0).int(),
                        token_type_ids=torch.zeros_like(ids))

    def decode(self, tokens, **kwargs):
        return "".join(chr(x) for x in tokens if x)


def make_router():
    calls = []
    tokenizer = Tokenizer()

    def generate(input_ids, attention_mask, **kwargs):
        assert "token_type_ids" not in kwargs
        assert kwargs == dict(max_new_tokens=96, do_sample=False, num_beams=1,
                              temperature=None, top_p=None, pad_token_id=0, eos_token_id=0)
        assert torch.equal(attention_mask.bool(), input_ids.ne(0))
        calls.append(input_ids.clone())
        return torch.cat((input_ids, input_ids[:, -1:], torch.zeros_like(input_ids[:, -1:])), dim=1)

    router = SimpleNamespace(tokenizer=tokenizer,
        model=SimpleNamespace(device="cpu", generate=generate),
        active_task=None, generate=lambda *a, **k: "original", build_prompt=lambda user: "chat:" + user)
    router.route = lambda task: setattr(router, "active_task", task)
    return router, calls


def test_varied_length_padding_replay_coverage_and_restoration():
    router, calls = make_router()
    original = router.generate
    with prepared_router_generation(router, "action", ["a", "longer", "a", "xyz"], batch_size=2):
        assert [router.generate(u, max_new_tokens=96) for u in ["a", "longer", "xyz"]] == ["a", "r", "z"]
        assert router.tokenizer.padding_side == "right"
        with pytest.raises(ValueError, match="does not match"):
            router.generate("a", max_new_tokens=95)
        with pytest.raises(ValueError, match="does not match"):
            router.generate("unknown", max_new_tokens=96)
        router.route("objects")
        with pytest.raises(ValueError, match="does not match"):
            router.generate("a", max_new_tokens=96)
    assert len(calls) == 2 and sum(len(c) for c in calls) == 3
    assert router.generate is original


def test_restore_original_on_generation_or_parser_failure():
    router, _ = make_router()
    original = router.generate
    with pytest.raises(RuntimeError):
        with prepared_router_generation(router, "action", ["a"]):
            raise RuntimeError("parser failure")
    assert router.generate is original and router.tokenizer.padding_side == "right"

    def failed(**kwargs):
        raise RuntimeError("model failure")
    router.model.generate = failed
    with pytest.raises(RuntimeError):
        with prepared_router_generation(router, "action", ["a"]):
            pytest.fail("failed generation must not produce a replay")
    assert router.generate is original and router.tokenizer.padding_side == "right"
