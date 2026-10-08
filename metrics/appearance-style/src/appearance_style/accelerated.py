"""Encode frame batches and each distinct style once; preserve CLIP logits."""


def compute(rows, device, submodules, *, batch_size=32):
    import numpy as np
    import torch
    from PIL import Image
    from vbench.appearance_style import clip, clip_transform_Image, load_video
    from vbench_audit_core.video_batches import prefetch, frame_batches
    model, _ = clip.load(device=device, **submodules)
    transform = clip_transform_Image(224)
    queries = [r["auxiliary_info"]["appearance_style"]["appearance_style"] for r in rows]
    text_features = {}
    scores = [[] for _ in rows]
    with torch.no_grad():
        for text in dict.fromkeys(queries):
            feature = model.encode_text(clip.tokenize([text]).to(device))
            text_features[text] = feature / feature.norm(dim=-1, keepdim=True)
        decoded = prefetch(range(len(rows)), lambda i: torch.stack([
            transform(Image.fromarray(f)) for f in load_video(rows[i]["video"], return_tensor=False)]))
        for owners, frames in frame_batches(decoded, batch_size):
            features = model.encode_image(frames.to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            # Retain upstream CLIP's exp(logit_scale), then its /100 reducer.
            for text in dict.fromkeys(queries[i] for i in owners):
                positions = [j for j, owner in enumerate(owners) if queries[owner] == text]
                logits = model.logit_scale.exp() * features[positions] @ text_features[text].t()
                for position, value in zip(positions, logits[:, 0].tolist()):
                    scores[owners[position]].append(float(value) / 100)
    returned = [{"video_path": row["video"], "video_results": float(np.mean(values)),
                 "frame_results": values, "cur_sim": values[-1]} for row, values in zip(rows, scores)]
    return sum(sum(s) for s in scores) / sum(map(len, scores)), returned
