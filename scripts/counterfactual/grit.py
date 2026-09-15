"""GRiT object grounding, used only to build dataset-construction masks.

GRiT is a *caption-free* detector: it emits its own predicted category names and
never sees a query.  Matching a target therefore means comparing the bare noun
from the official annotation (for example `cat` from `object_en` = "cat and dog")
against GRiT's emitted labels -- `prompt_en` ("a cat and a dog") would match
nothing.

Two conventions matter for parity with the metric being audited:

* **RGB input.**  `vbench.utils.load_video` decodes with decord and hands GRiT
  RGB frames; detectron2's own `read_image` would give BGR.  The two produce
  different detections, so this module feeds RGB, matching VBench.
* **ObjectDet mode.**  `initialize_model_det` yields short category names that
  exact-match the annotations; DenseCap mode emits free-form phrases that do not.

Boxes are absolute pixels in `xyxy` order, in the coordinate frame of the frame
passed in (verified by the accompanying probe: concatenating two frames shifts x
by the frame width).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_WEIGHTS = "/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth"

_NORMALISE = re.compile(r"[^a-z0-9]+")


def normalise_label(text: str) -> str:
    """Fold a category name so `cell phone` and `cellphone` compare equal."""
    return _NORMALISE.sub("", (text or "").strip().lower())


@dataclass(frozen=True)
class Detection:
    label: str
    box: tuple[int, int, int, int]
    score: float

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


class GritNotAvailable(RuntimeError):
    """Raised when the GRiT runtime or its weights are missing."""


class GritDetector:
    """Thin wrapper around VBench's locked GRiT usage."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cuda") -> None:
        try:
            import torch
            from vbench.third_party.grit_model import DenseCaptioning
        except ImportError as error:  # pragma: no cover - depends on the H100 image
            raise GritNotAvailable(
                "GRiT requires the VBench venv with torch and detectron2 installed"
            ) from error
        weights = Path(weights).expanduser()
        if not weights.is_file():
            raise GritNotAvailable(f"GRiT weights not found: {weights}")
        self._torch = torch
        self.device = torch.device(device)
        self.model = DenseCaptioning(self.device)
        self.model.initialize_model_det(str(weights))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Every detection in one RGB frame."""
        with self._torch.no_grad():
            predictions, _ = self.model.run_det_tensor(frame)
        instances = predictions["instances"].to("cpu")
        boxes = (
            instances.pred_boxes.tensor.numpy()
            if instances.has("pred_boxes")
            else np.zeros((0, 4))
        )
        names = (
            list(instances.pred_object_descriptions.data)
            if instances.has("pred_object_descriptions")
            else []
        )
        scores = (
            instances.scores.numpy() if instances.has("scores") else np.zeros(len(boxes))
        )
        out: list[Detection] = []
        for name, box, score in zip(names, boxes, scores):
            rounded = tuple(int(round(float(value))) for value in box)
            out.append(Detection(label=str(name), box=rounded, score=float(score)))  # type: ignore[arg-type]
        return out

    def detect_many(self, frames: np.ndarray, progress: Any = None) -> list[list[Detection]]:
        results = []
        for index, frame in enumerate(frames):
            results.append(self.detect(np.ascontiguousarray(frame)))
            if progress is not None:
                progress(index)
        return results


def select_target(
    detections: Sequence[Detection],
    target: str,
    min_score: float = 0.0,
) -> Detection | None:
    """Largest detection whose normalised label matches `target` exactly.

    Exact normalised matching mirrors VBench's own `key in predicted_names`
    check.  Deterministic: ties break on the higher score, then on the box.
    """
    wanted = normalise_label(target)
    if not wanted:
        return None
    matches = [
        detection
        for detection in detections
        if normalise_label(detection.label) == wanted and detection.score >= min_score
    ]
    if not matches:
        return None
    return max(matches, key=lambda detection: (detection.area, detection.score))


def track_target(
    per_frame: Sequence[Sequence[Detection]],
    target: str,
    min_score: float = 0.0,
) -> list[tuple[int, int, int, int] | None]:
    """One box per frame for `target`, carrying the nearest detection forward.

    A detector drop-out in the middle of a clip must not punch a hole in the
    suppression region, so missing frames inherit the nearest detected box
    (searching forward first, then backward).  If the target is never detected
    the whole base is ineligible and the caller must reject it.
    """
    boxes: list[tuple[int, int, int, int] | None] = []
    for frame_detections in per_frame:
        found = select_target(frame_detections, target, min_score)
        boxes.append(found.box if found else None)
    if all(box is None for box in boxes):
        return [None] * len(boxes)
    known = [index for index, box in enumerate(boxes) if box is not None]
    for index, box in enumerate(boxes):
        if box is not None:
            continue
        nearest = min(known, key=lambda candidate: (abs(candidate - index), candidate))
        boxes[index] = boxes[nearest]
    return boxes


def coverage(boxes: Iterable[tuple[int, int, int, int] | None]) -> float:
    boxes = list(boxes)
    if not boxes:
        return 0.0
    return sum(1 for box in boxes if box is not None) / len(boxes)
