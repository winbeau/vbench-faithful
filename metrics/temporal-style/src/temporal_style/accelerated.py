"""Batch original eight-frame ViCLIP inputs; retain the native cosine reducer."""


def compute(rows, device, submodules):
    import numpy as np
    import torch
    from vbench_audit_models.viclip import embeddings

    with torch.no_grad():
        video, text = embeddings(rows, device, submodules)
        video = video / video.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        # Retain each original dot product; no cross-video score weighting.
        values = [float((v[None] @ t[:, None])[0, 0].cpu()) for v, t in zip(video, text)]
    return float(np.mean(values)), [{"video_path": r["video"], "video_results": v} for r, v in zip(rows, values)]
