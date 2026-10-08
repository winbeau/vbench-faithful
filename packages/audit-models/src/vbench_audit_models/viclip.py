"""Native ViCLIP embeddings with bounded batches and within-run visual reuse."""
import hashlib
import json
import os
from pathlib import Path


def embeddings(rows, device, submodules, *, batch_size=4):
    import torch
    from vbench.utils import CACHE_DIR, clip_transform, read_frames_decord_by_fps
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
    from vbench.third_party.ViCLIP.viclip import ViCLIP
    from vbench_audit_core.video_batches import prefetch
    from vbench_audit_core.run_inference_cache import RunInferenceCache

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    tokenizer = SimpleTokenizer(str(Path(CACHE_DIR) / "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
    model = ViCLIP(tokenizer=tokenizer, **submodules).to(device)
    transform = clip_transform(224)
    root = os.environ.get("VBENCH_EVAL_RUN_INFERENCE_CACHE")
    context = os.environ.get("VBENCH_EVAL_RUN_INFERENCE_CONTEXT")
    cache = None
    if root and context:
        h = hashlib.sha256()
        with Path(submodules["pretrain"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1048576), b""):
                h.update(block)
        cache = RunInferenceCache(root, {"runtime": json.loads(context), "checkpoint": h.hexdigest(),
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
    return torch.stack(video_features), torch.stack([text_features[r["prompt"]] for r in rows])
