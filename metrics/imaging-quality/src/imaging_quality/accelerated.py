"""Batch MUSIQ frames with the original longer-side transform and scales."""


def compute(rows, device, submodules, *, batch_size=32):
    import torch
    from vbench.imaging_quality import MUSIQ, load_video, transform
    from vbench_audit_core.video_batches import prefetch, frame_batches
    model = MUSIQ(pretrained_model_path=submodules["model_path"]).to(device)
    # Match upstream's setting; do not silently change child-module training.
    model.training = False
    scores = [[] for _ in rows]
    decoded = prefetch(range(len(rows)), lambda i: transform(load_video(rows[i]["video"]), "longer"))
    with torch.no_grad():
        for owners, frames in frame_batches(decoded, batch_size):
            values = model(frames.to(device)).reshape(-1).tolist()
            if len(values) != len(owners):
                raise ValueError("MUSIQ changed frame coverage")
            for owner, value in zip(owners, values):
                scores[owner].append(float(value))
    values = [sum(s) / len(s) for s in scores]
    return sum(values) / len(values) / 100, [{"video_path": r["video"], "video_results": v} for r, v in zip(rows, values)]
