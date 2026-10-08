"""Batch native frames across videos, retaining the LAION score and reducer."""


def compute(rows, device, submodules, *, batch_size=32):
    import torch
    from torch.nn import functional as F
    from vbench.aesthetic_quality import get_aesthetic_model, clip_transform, load_video
    from vbench_audit_core.video_batches import prefetch, frame_batches
    from vbench_audit_models.clip_vision import load_clip_vision
    predictor = get_aesthetic_model(submodules[1]).to(device).eval()
    encoder = load_clip_vision(submodules[0], device)
    transform = clip_transform(224)
    scores = [[] for _ in rows]
    decoded = prefetch(range(len(rows)), lambda i: transform(load_video(rows[i]["video"])))
    with torch.no_grad():
        for owners, frames in frame_batches(decoded, batch_size):
            features = F.normalize(encoder.encode_image(frames.to(device)).float(), dim=-1, p=2)
            values = predictor(features).squeeze(-1) / 10
            for owner, value in zip(owners, values):
                scores[owner].append(value)
    values = [torch.stack(s).mean().item() for s in scores]
    return sum(values) / len(values), [{"video_path": r["video"], "video_results": v} for r, v in zip(rows, values)]
