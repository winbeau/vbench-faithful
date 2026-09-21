"""Read-only pinned GRiT adapters retaining both heads' instance evidence.

No task predicates or scores are computed here. Hooking only records returned
ROI instances; it neither changes proposals nor replaces the official return.
"""
from __future__ import annotations

from dataclasses import asdict
import os
import sys
from pathlib import Path
from types import MethodType

import numpy as np

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.upstream import import_official_module, verify_upstream


def preflight(checkpoint: Path, upstream: Path | None = None) -> dict:
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"existing local GRiT checkpoint required: {checkpoint}")
    state = verify_upstream(upstream)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    # local_files_only prevents any BERT cache repair/download. Loading the
    # tokenizer is CPU-only and occurs before importing the GRiT model.
    from transformers import BertTokenizer
    BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    return {"upstream": asdict(state), "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint), "threshold": .5,
            "rgb": True, "num_frames": 16}


def serialize_instances(instances) -> list[dict]:
    if not all(instances.has(name) for name in ("pred_boxes", "scores", "pred_object_descriptions")):
        raise ValueError("GRiT instances lack boxes/scores/descriptions")
    boxes = instances.pred_boxes.tensor.detach().cpu().tolist()
    scores = instances.scores.detach().cpu().tolist()
    texts = list(instances.pred_object_descriptions.data)
    if not len(boxes) == len(scores) == len(texts):
        raise ValueError("GRiT instance field lengths disagree")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rank = {index: position for position, index in enumerate(order)}
    return [{"index": i, "box": box, "score": float(score), "text": str(text),
             "confidence_rank": rank[i]} for i, (box, score, text) in enumerate(zip(boxes, scores, texts))]


def box_iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def bind_heads(captions: list[dict], objects: list[dict], *, threshold: float = .9) -> dict:
    """Verify index alignment or bind by uniquely qualifying one-to-one IoU.

    All records, including unmatched and ambiguous ones, remain in evidence.
    Equal lengths alone are not proof of common instance identity.
    """
    if not 0 < threshold <= 1:
        raise ValueError("invalid IoU threshold")
    overlaps = [[box_iou(c["box"], o["box"]) for o in objects] for c in captions]
    candidates = [[j for j, value in enumerate(row) if value >= threshold] for row in overlaps]
    aligned = len(captions) == len(objects) and all(
        np.allclose(c["box"], o["box"], rtol=0, atol=1e-4) and candidates[i] == [i]
        for i, (c, o) in enumerate(zip(captions, objects)))
    pairs = []
    for i, caption in enumerate(captions):
        if aligned:
            j, method = i, "verified_index"
        elif len(candidates[i]) == 1 and sum(candidates[i][0] in row for row in candidates) == 1:
            j, method = candidates[i][0], "unique_iou"
        else:
            j, method = None, "ambiguous" if candidates[i] else "unmatched"
        pairs.append({"caption_index": i, "object_index": j, "method": method,
                      "iou": overlaps[i][j] if j is not None else None,
                      "caption": caption, "object": objects[j] if j is not None else None})
    matched = {item["object_index"] for item in pairs if item["object_index"] is not None}
    return {"pairs": pairs, "unmatched_object_indices": sorted(set(range(len(objects))) - matched),
            "index_aligned": aligned, "iou_threshold": threshold}


class GritEvidenceModel:
    def __init__(self, dimension: str, checkpoint: Path, *, device: str, upstream: Path | None = None):
        if dimension not in {"object_class", "color"}:
            raise ValueError("GRiT evidence dimension must be explicit")
        self.provenance = preflight(checkpoint, upstream)
        self.dimension = dimension
        self.module, state = import_official_module(dimension, upstream)
        import torch
        self.torch = torch
        self.model = self.module.DenseCaptioning(torch.device(device))
        initializer = self.model.initialize_model_det if dimension == "object_class" else self.model.initialize_model
        initializer(model_weight=str(checkpoint))
        self.head_outputs = []
        roi = self.model.demo.predictor.model.roi_heads
        original = roi._forward_box
        recorder = self

        def observed(_roi, *args, **kwargs):
            value = original(*args, **kwargs)
            recorder.head_outputs.append({"head": "object" if kwargs.get("det_box", False) else "primary",
                                          "instances": serialize_instances(value[0])})
            return value

        roi._forward_box = MethodType(observed, roi)

    def frames(self, path: Path):
        """Exactly the dimension's pinned official sampling and resize path."""
        if self.dimension == "object_class":
            video = self.module.load_video(str(path), num_frames=16)
            _, _, h, w = video.size()
            if min(h, w) > 768:
                scale = 720. / min(h, w)
                video = self.module.transforms.Resize((int(scale*h), int(scale*w)))(video)
            return video.permute(0, 2, 3, 1).numpy()
        video = self.module.load_video(str(path), num_frames=16, return_tensor=False)
        _, h, w, _ = video.shape
        if min(h, w) > 768:
            import cv2
            scale = 720. / min(h, w)
            video = np.stack([cv2.resize(f, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR) for f in video])
        return video

    def detect(self, image) -> dict:
        self.head_outputs.clear()
        try:
            with self.torch.no_grad():
                predictions, _ = self.model.run_det_tensor(np.ascontiguousarray(image))
            if [r["head"] for r in self.head_outputs] != ["primary", "object"]:
                raise ValueError("unexpected GRiT two-head call sequence")
            primary, objects = (r["instances"] for r in self.head_outputs)
            instances = predictions["instances"]
            labels = list(instances.det_obj.data)
            if labels != [r["text"] for r in objects]:
                raise ValueError("upstream det_obj differs from captured ObjectDet output")
            # Both raw heads use identical pre-postprocess coordinates. Keep
            # postprocessed boxes separately for visualization and trace audit.
            return {"status": "succeeded", "primary": primary, "objects": objects,
                    "binding": bind_heads(primary, objects),
                    "postprocessed": serialize_instances(instances), "legacy_labels": labels,
                    "image_shape": list(image.shape), "coordinate_space": "raw_roi_image"}
        except Exception as exc:
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "primary": [], "objects": [], "binding": None,
                    "partial_heads": self.head_outputs.copy()}

    def detect_video(self, path: Path) -> list[dict]:
        return [{"frame_index": i, **self.detect(frame)} for i, frame in enumerate(self.frames(path))]
