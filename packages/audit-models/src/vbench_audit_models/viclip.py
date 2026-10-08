"""Native ViCLIP embeddings with bounded batches and within-run visual reuse."""
import hashlib
import json
import os
from pathlib import Path


def load_model(tokenizer, device, submodules):
    """Skip overwritten random values only during strict native checkpoint load.

    Called in an isolated model worker, before prefetch threads are started.
    All patches are restored even when loading fails; deterministic buffers,
    including the causal attention mask, retain their original initialization.
    """
    from contextlib import ExitStack
    from unittest.mock import patch
    import torch
    from vbench.third_party.ViCLIP.viclip import ViCLIP

    if not submodules.get("pretrain") or not Path(submodules["pretrain"]).is_file():
        raise ValueError("ViCLIP requires the verified complete local checkpoint")
    with ExitStack() as stack:
        for name in ("uniform_", "normal_", "kaiming_uniform_", "xavier_uniform_"):
            stack.enter_context(patch.object(torch.nn.init, name, lambda tensor, *a, **k: tensor))
        model = ViCLIP(tokenizer=tokenizer, **submodules).to(device)
    # Native checkpoint wrappers are for backward recomputation. In no_grad
    # inference they execute the identical blocks with additional Python cost.
    model.vision_encoder.transformer.checkpoint_num = 0
    return model


def embeddings(rows, device, submodules, *, batch_size=4, diagnostics=None):
    import torch
    from vbench.utils import CACHE_DIR, clip_transform, read_frames_decord_by_fps
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
    from vbench_audit_core.video_batches import prefetch
    from vbench_audit_core.run_inference_cache import RunInferenceCache

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    tokenizer = SimpleTokenizer(str(Path(CACHE_DIR) / "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
    model = load_model(tokenizer, device, submodules)
    transform = clip_transform(224)
    root = os.environ.get("VBENCH_EVAL_RUN_INFERENCE_CACHE")
    context = os.environ.get("VBENCH_EVAL_RUN_INFERENCE_CONTEXT")
    cache = None
    if root and context:
        runtime = json.loads(context)
        checkpoint = str(Path(submodules["pretrain"]).resolve())
        # The controller hashes every selected asset on every invocation.
        # Reuse that current-run digest under the immutable-runtime contract;
        # standalone callers without a verified context still hash the file.
        checksum = runtime.get("assets", {}).get(checkpoint)
        if not isinstance(checksum, str) or len(checksum) != 64:
            from vbench_audit_core.eval_cache import file_digest
            checksum = file_digest(checkpoint)
        cache = RunInferenceCache(root, {"runtime": runtime, "checkpoint": checksum,
            "protocol": "viclip-native-eight-middle-fp32-v1", "batch_size": batch_size})
    video_features = []
    text_features = {}
    queries = list(dict.fromkeys(row["prompt"] for row in rows))
    with torch.no_grad():
        for start in range(0, len(queries), 32):
            batch = queries[start:start + 32]
            features = model.encode_text(batch).float()
            text_features.update(zip(batch, features))
        decoded = prefetch(range(len(rows)), lambda i: transform(read_frames_decord_by_fps(
            rows[i]["video"], num_frames=8, sample="middle")))
        pending = []

        def encode(batch):
            tensor = torch.stack(batch)
            specification = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "input_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
            def compute():
                return model.encode_vision(tensor.to(device), test=True).float().cpu().tolist()
            def valid(value):
                try:
                    t = torch.tensor(value, dtype=torch.float32)
                    return t.shape == (len(batch), model.embed_dim) and bool(torch.isfinite(t).all())
                except (TypeError, ValueError):
                    return False
            values = cache.get_or_compute(specification, compute, valid=valid) if cache else compute()
            return torch.tensor(values, device=device, dtype=torch.float32)

        for _, frames in decoded:
            pending.append(frames)
            if len(pending) == batch_size:
                video_features.extend(encode(pending))
                pending = []
        if pending:
            video_features.extend(encode(pending))
    if diagnostics is not None:
        diagnostics.update(video_count=len(rows), frames_per_video=8, video_batch_size=batch_size,
                           fresh_visual_inference=cache is None or cache.misses > 0)
        if cache is not None:
            diagnostics["run_shared_inference"] = {"hits": cache.hits, "misses": cache.misses,
                "unit": "video feature batch", "scope": "current evaluation only", "cache": str(cache.root)}
    return torch.stack(video_features), torch.stack([text_features[r["prompt"]] for r in rows])
