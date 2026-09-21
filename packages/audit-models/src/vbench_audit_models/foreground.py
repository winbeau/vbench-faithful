"""Shared local-only COCO box and MobileSAM adapters; no scoring formulas."""
from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

COCO_SUBJECTS = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
                 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
                 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20,
                 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25}


class CocoSubjectBoxDetector:
    def __init__(self, checkpoint: Path, *, expected_sha256: str, device: str,
                 threshold: float = .8, size: int = 512):
        import torch
        import torchvision
        from torchvision.models import resnet50
        from torchvision.models.detection import MaskRCNN
        from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
        from torchvision.ops.misc import FrozenBatchNorm2d
        from vbench_audit_core.inputs import sha256_file

        if sha256_file(checkpoint) != expected_sha256:
            raise ValueError('unexpected COCO detector checkpoint')
        if torchvision.__version__.split('+')[0] != '0.20.1':
            raise ValueError('automatic localization pins torchvision 0.20.1')
        if not 0 < threshold <= 1:
            raise ValueError('invalid detection threshold')
        backbone = _resnet_fpn_extractor(resnet50(weights=None, norm_layer=FrozenBatchNorm2d), 3)
        model = MaskRCNN(backbone, num_classes=91, min_size=size, max_size=size)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0
        # Only box outputs prompt SAM. Disable the unused mask branch after
        # loading the complete checkpoint; box computation is unchanged.
        model.roi_heads.mask_roi_pool = None
        model.roi_heads.mask_head = None
        model.roi_heads.mask_predictor = None
        self.model = model.to(device).eval()
        self.device, self.threshold = device, threshold
        self.provenance = {'family': 'maskrcnn_resnet50_fpn_coco_v1_box_branch',
                           'weights_sha256': expected_sha256, 'threshold': threshold,
                           'min_size': size, 'max_size': size, 'batch_size': 4,
                           'native_roi_align_forward': True,
                           'strict_determinism_disabled_only_inside_box_forward': True}

    def boxes_for(self, frames, phrase: str | None):
        import torch
        if phrase is not None and phrase not in COCO_SUBJECTS:
            raise ValueError(f'unsupported official subject: {phrase}')
        result = []
        strict = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            # Torchvision otherwise switches ROIAlign to its compiled Python
            # implementation for CUDA, even without gradients. Record this
            # scoped exception; cached detections/masks remain replayable.
            torch.use_deterministic_algorithms(False)
            with torch.inference_mode():
                for start in range(0, len(frames), 4):
                    images = [frame.to(self.device, dtype=torch.float32)/255 for frame in frames[start:start+4]]
                    for pred in self.model(images):
                        wanted = torch.as_tensor([COCO_SUBJECTS[phrase]] if phrase is not None else list(COCO_SUBJECTS.values()), device=pred['labels'].device)
                        keep = torch.isin(pred['labels'], wanted) & (pred['scores'] >= self.threshold)
                        result.append({'boxes': pred['boxes'][keep].cpu().tolist(),
                                       'scores': pred['scores'][keep].cpu().tolist()})
        finally:
            torch.use_deterministic_algorithms(strict, warn_only=warn)
        return result


def deterministic_dense_position_encoding(layer: Any, size: tuple[int, int]):
    """Exact dense SAM grid, avoiding CUDA's unsupported float cumsum kernel.

    Pinned MobileSAM uses cumsum(ones) - 0.5 for this coordinate grid. Integer
    arange yields the same exactly representable half-integers at SAM sizes.
    All positional projection weights and sparse prompt encoding stay intact.
    This local runtime adapter never edits the external MobileSAM checkout.
    """
    import torch

    height, width = size
    device = layer.positional_encoding_gaussian_matrix.device
    y = ((torch.arange(1, height + 1, device=device, dtype=torch.float32) - .5) / height)
    x = ((torch.arange(1, width + 1, device=device, dtype=torch.float32) - .5) / width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return layer._pe_encoding(torch.stack([xx, yy], dim=-1)).permute(2, 0, 1)


def load_mobile_sam_predictor(checkpoint: Path, device: Any):
    """Load the local model for human or explicitly automatic prompts."""
    if not checkpoint.is_file():
        raise ValueError("local MobileSAM vit_t checkpoint is required")
    try:
        from mobile_sam import sam_model_registry, SamPredictor
    except ImportError as exc:
        raise ValueError("install the pinned MobileSAM source in the model environment") from exc
    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint)).to(device).eval()
    layer = model.prompt_encoder.pe_layer
    layer.forward = MethodType(deterministic_dense_position_encoding, layer)
    return SamPredictor(model)


class MobileSamForegroundProvider:
    """Class-agnostic union of the frozen movable COCO class vocabulary.

    A missing foreground detection produces an empty mask and an explicit
    diagnostic. It is not a claim that no physical foreground exists.
    """
    def __init__(self, detector, predictor, *, weights_sha256):
        self.detector, self.predictor = detector, predictor
        self.provenance = {"role": "scoring", "family": "mobilesam", "model": "vit_t",
                           "weights_sha256": weights_sha256, "box_detector": detector.provenance,
                           "classes": COCO_SUBJECTS, "human_confirmed": False,
                           "construction_masks_reused": False, "prompt_source": "automatic_per_frame_detection",
                           "missing_foreground": "empty_mask_no_interpolation"}
        self.last_diagnostics = None

    def masks_for(self, frames, *, phrase=None):
        import numpy as np
        import torch
        if frames.ndim != 4 or frames.shape[1] != 3 or not torch.isfinite(frames).all():
            raise ValueError("expected native RGB [T,3,H,W]")
        if frames.min() < 0 or frames.max() > 255 or not torch.equal(frames, frames.to(torch.uint8).to(frames.dtype)):
            raise ValueError("native RGB must contain exact byte values")
        frames = frames.to(torch.uint8)
        detections = self.detector.boxes_for(frames, phrase)
        if len(detections) != len(frames):
            raise ValueError("detector frame count mismatch")
        masks, traces = [], []
        with torch.inference_mode():
            for image, detection in zip(frames.cpu().permute(0, 2, 3, 1).numpy(), detections):
                height, width = image.shape[:2]
                boxes = np.asarray(detection["boxes"], dtype=np.float32).reshape(-1, 4)
                if (not np.isfinite(boxes).all() or (boxes < 0).any()
                    or (boxes[:, 2:] <= boxes[:, :2]).any()
                    or (boxes[:, [0, 2]] > width).any() or (boxes[:, [1, 3]] > height).any()):
                    raise ValueError("invalid detector box")
                union = np.zeros((height, width), np.uint8)
                qualities = []
                if len(boxes):
                    self.predictor.set_image(np.ascontiguousarray(image), image_format="RGB")
                    for box in boxes:
                        prediction, quality, _ = self.predictor.predict(box=box, multimask_output=False)
                        if np.shape(prediction) != (1, height, width) or not np.isin(prediction, (0, 1)).all():
                            raise ValueError("SAM must return one native binary mask")
                        union |= np.asarray(prediction[0], dtype=np.uint8)
                        qualities.append(float(np.asarray(quality).reshape(-1)[0]))
                masks.append(union)
                traces.append({**detection, "sam_predicted_iou": qualities})
        result = np.stack(masks)
        self.last_diagnostics = {"num_frames": len(frames), "foreground_fraction": result.mean((1, 2)).tolist(),
                                 "num_empty_foreground_frames": int((result.sum((1, 2)) == 0).sum()),
                                 "detections": traces}
        return result
