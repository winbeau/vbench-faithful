"""Independent MobileSAM scoring localization with frozen human prompts.

Construction masks (SegFormer/GrabCut) never enter this module. DINO tokens
never enter any localization decision. A single prompt is frozen per source
clip and reused, unchanged, on clean and every corrupted variant. MobileSAM
is rerun on each actual variant, so corruption-induced misses remain visible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

from .models import SubjectMasks


def prompt_sha256(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "sha256"}
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def validate_prompt(row: Mapping[str, Any]) -> None:
    import numpy as np

    if row.get("role") != "scoring" or row.get("prompt_source") != "human_clean_frame":
        raise ValueError("scoring prompts must come from a human-reviewed clean frame, not construction masks")
    if row.get("human_confirmed") is not True or not row.get("reviewer") or not row.get("confirmed_at"):
        raise ValueError("one human confirmation per source clip is required")
    if row.get("sha256") != prompt_sha256(row):
        raise ValueError("frozen localizer prompt hash mismatch")
    if not row.get("video_uid") or not row.get("phrase"):
        raise ValueError("prompt needs video_uid and phrase")
    source_hash = row.get("source_frame_sha256", "")
    if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("prompt needs the clean source frame SHA256")
    height, width = row["image_size"]
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("invalid prompt image_size")
    box = row.get("box")
    points, labels = row.get("points", []), row.get("point_labels", [])
    if box is None and not points:
        raise ValueError("prompt requires a box or points")
    if box is not None:
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError("invalid box")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("box outside image")
    if len(points) != len(labels) or any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("points and labels disagree")
    for point in points:
        if len(point) != 2 or not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError("point outside image")


def load_prompts(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prompts = {}
    for row in rows:
        validate_prompt(row)
        uid = row["video_uid"]
        if uid in prompts:
            raise ValueError(f"duplicate localizer prompt: {uid}")
        prompts[uid] = row
    return prompts


from vbench_audit_models.foreground import (
    deterministic_dense_position_encoding, load_mobile_sam_predictor,
)


class MobileSamSubjectMaskProvider:
    def __init__(self, prompts: Mapping[str, Mapping[str, Any]], *, clip_to_video_uid: Mapping[str, str],
                 checkpoint: Path | None = None, device: Any = "cpu", predictor: Any | None = None):
        from vbench_audit_core.inputs import sha256_file

        self.prompts = {key: dict(value) for key, value in prompts.items()}
        for row in self.prompts.values():
            validate_prompt(row)
        self.clip_to_video_uid = dict(clip_to_video_uid)
        if predictor is None:
            if checkpoint is None or not checkpoint.is_file():
                raise ValueError("local MobileSAM vit_t checkpoint is required")
            predictor = load_mobile_sam_predictor(checkpoint, device)
        self.predictor = predictor
        self.provenance = {"role": "scoring", "family": "mobilesam", "model": "vit_t",
                           "weights_sha256": sha256_file(checkpoint) if checkpoint else None,
                           "multimask_output": False, "prompt_policy": "frozen_per_source_clip",
                           "dense_position_grid": "arange_equivalent_to_cumsum_of_ones",
                           "construction_masks_reused": False}

    def masks_for(self, video: Path, frames: Any, phrase: str) -> SubjectMasks:
        import numpy as np
        import torch

        uid = self.clip_to_video_uid[video.name]
        row = self.prompts[uid]
        if phrase != row["phrase"]:
            raise ValueError("scoring phrase differs from the frozen human prompt")
        if list(frames.shape[-2:]) != row["image_size"]:
            raise ValueError("scoring frame size differs from the frozen prompt")
        images = frames.detach().cpu().permute(0, 2, 3, 1).numpy()
        masks = []
        with torch.inference_mode():
            for image in images:
                self.predictor.set_image(np.ascontiguousarray(image, dtype=np.uint8), image_format="RGB")
                output, _, _ = self.predictor.predict(
                    point_coords=np.asarray(row["points"], np.float32) if row.get("points") else None,
                    point_labels=np.asarray(row["point_labels"], np.int32) if row.get("points") else None,
                    box=np.asarray(row["box"], np.float32) if row.get("box") is not None else None,
                    multimask_output=False,
                )
                output = np.asarray(output)
                if output.shape != (1, *image.shape[:2]) or not np.isin(output, [0, 1]).all():
                    raise ValueError("MobileSAM must return one binary mask at native frame size")
                masks.append(torch.from_numpy(output.astype(np.float32)))
        stacked = torch.stack(masks)
        present = stacked.flatten(2).sum(-1) > 0
        return SubjectMasks(stacked, present, phrase, "mobilesam-vit_t-scoring", row["sha256"])
