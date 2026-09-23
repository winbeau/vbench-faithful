"""Local SAM automatic region proposals, without scores or semantic labels.

All returned proposals are retained. A mask is evidence to check, not an object
identity, a temporal track, or a certificate that a motion is physical.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np

from vbench_audit_core.inputs import sha256_file


def load_local_sam(source_root):
    root = Path(source_root).resolve()
    if not (root / "segment_anything" / "build_sam.py").is_file():
        raise FileNotFoundError("local source must contain segment_anything/build_sam.py")
    for name, module in list(sys.modules.items()):
        if name == "segment_anything" or name.startswith("segment_anything."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError("another segment_anything source is already imported")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("segment_anything")
    finally:
        sys.path.remove(str(root))
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("SAM import escaped configured source")
    return module


def pack_proposals(records, shape):
    """Lossless binary packing; no area/category/rank filtering after SAM."""
    h, w = map(int, shape)
    if h < 1 or w < 1:
        raise ValueError("positive mask geometry required")
    masks, quality, boxes, points, areas = [], [], [], [], []
    for record in records:
        mask = np.asarray(record["segmentation"])
        if mask.shape != (h, w) or not np.isin(mask, (0, 1)).all():
            raise ValueError("native binary mask required")
        area = int(mask.sum())
        if area <= 0 or area != record["area"]:
            raise ValueError("nonempty mask with matching area required")
        q = np.asarray([record["predicted_iou"], record["stability_score"]], dtype=np.float32)
        box, point = np.asarray(record["bbox"], np.float32), np.asarray(record["point_coords"], np.float32)
        if (box.shape != (4,) or point.shape != (1, 2) or not np.isfinite(q).all()
                or not np.isfinite(box).all() or not np.isfinite(point).all()):
            raise ValueError("finite proposal metadata with expected geometry required")
        masks.append(mask.astype(bool)); quality.append(q); boxes.append(box)
        points.append(point[0]); areas.append(area)
    masks = np.stack(masks) if masks else np.zeros((0, h, w), bool)
    return {"masks_packed": np.packbits(masks, axis=-1),
            "quality": np.asarray(quality, np.float32).reshape(-1, 2),
            "boxes_xywh": np.asarray(boxes, np.float32).reshape(-1, 4),
            "prompt_points_xy": np.asarray(points, np.float32).reshape(-1, 2),
            "areas": np.asarray(areas, np.int64), "image_shape": np.array([h, w]),
            "union_fraction": np.array(masks.any(axis=0).mean())}


def unpack_masks(packed, shape):
    h, w = map(int, shape)
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 3 or packed.shape[1:] != (h, (w + 7) // 8):
        raise ValueError("packed mask geometry/dtype mismatch")
    return np.unpackbits(packed, axis=-1, count=w).astype(bool)


class SamRegionModel:
    def __init__(self, source_root, checkpoint, device, generator_config):
        import torch

        weight, root = Path(checkpoint), Path(source_root).resolve()
        if not weight.is_file():
            raise FileNotFoundError(weight)
        module = load_local_sam(root)
        model = module.sam_model_registry["vit_h"](checkpoint=None)
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        self.model = model.eval().to(torch.device(device))
        self.generator = module.SamAutomaticMaskGenerator(self.model, **generator_config)
        self.identity = {
            "backend": "local_sam_vit_h_automatic_regions", "source_root": str(root),
            "source_files": {str(p.relative_to(root)): sha256_file(p)
                             for p in sorted((root / "segment_anything").rglob("*.py"))},
            "checkpoint_sha256": sha256_file(weight), "generator": dict(generator_config),
            "post_selection": "none; keep every proposal returned by upstream AMG",
            "scope": "per-frame automatic regions; no category/prompt/tracking or motion score"}

    def propose(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
            raise ValueError("nonempty H,W,3 uint8 RGB required")
        return pack_proposals(self.generator.generate(np.ascontiguousarray(frame)), frame.shape[:2])


class SamPromptRegionModel(SamRegionModel):
    """Local SAM box/point re-localization; returns all hypotheses, not scores."""

    def __init__(self, source_root, checkpoint, device):
        super().__init__(source_root, checkpoint, device, {})
        self.predictor = load_local_sam(source_root).SamPredictor(self.model)
        self.identity.pop('generator')
        self.identity.update(backend='local_sam_vit_h_prompted_regions',
            post_selection='none; retain single-mask and three-mask outputs for each prompt',
            scope='one RGB image with explicit pixel prompts; no semantic or motion acceptance')

    def propose(self, frame, prompts):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError('native uint8 RGB required')
        h, w = frame.shape[:2]
        checked = []
        for prompt in prompts:
            box, point = np.asarray(prompt['box_xyxy'], float), np.asarray(prompt['point_xy'], float)
            if (box.shape != (4,) or point.shape != (2,) or not np.isfinite(box).all()
                    or not np.isfinite(point).all() or not 0 <= box[0] < box[2] <= w
                    or not 0 <= box[1] < box[3] <= h or not 0 <= point[0] < w or not 0 <= point[1] < h):
                raise ValueError('in-frame finite box and positive point required')
            checked.append((box, point))
        if not checked:
            return []
        self.predictor.set_image(np.ascontiguousarray(frame), image_format='RGB')
        records = []
        for i, (box, point) in enumerate(checked):
            # Both are proposals: a box can cover several objects and a point
            # can miss its intended target. Never pick one by predicted IoU.
            for use_point in (False, True):
                for multi in (False, True):
                    masks, quality, _ = self.predictor.predict(box=box,
                        point_coords=point[None] if use_point else None,
                        point_labels=np.ones(1, int) if use_point else None, multimask_output=multi)
                    masks, quality = np.asarray(masks), np.asarray(quality)
                    if (masks.ndim != 3 or masks.shape[1:] != (h, w)
                            or not np.isin(masks, (0, 1)).all() or quality.shape != (len(masks),)
                            or not np.isfinite(quality).all()):
                        raise ValueError('invalid prompted SAM output')
                    for j, mask in enumerate(masks):
                        records.append({'prompt_index': i, 'uses_positive_point': use_point,
                                        'multimask_output': multi, 'hypothesis': j,
                                        'predicted_iou': float(quality[j]), 'segmentation': mask.astype(bool)})
        return records
