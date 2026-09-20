from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class SubjectMasks:
    """Per-frame instance masks for one subject phrase.

    ``instance_masks`` is [T, I, h, w] with float coverage in [0, 1] (a binary
    mask is the special case of 0/1 coverage), and ``instance_present`` is
    [T, I]. Masks are at the *decoded* frame resolution; the caller resamples
    them onto whichever patch grid its feature extractor uses.
    """

    instance_masks: Any
    instance_present: Any
    phrase: str
    source: str
    source_sha256: str | None = None


class SubjectMaskProvider(Protocol):
    """Text-conditioned localizer. ``frames`` is the decoded [T, C, H, W] tensor.

    Providers receive the same decoded frames the feature extractor sees, so a
    provider can never silently score a different decode than the backbone.
    """

    def masks_for(self, video: Path, frames: Any, phrase: str) -> SubjectMasks:
        ...


class NpzSubjectMaskProvider:
    """Frozen offline masks: one ``<video stem>.npz`` per clip.

    Keys: ``masks`` [T, h, w] or [T, I, h, w] coverage, optional ``present``
    [T, I], optional ``scores`` [T, I]. Running the localizer offline keeps the
    metric's runtime free of the localizer's dependencies and makes the masks a
    hashable, replayable input instead of hidden model state.
    """

    def __init__(self, root: Path, *, key: str = "masks", present_key: str = "present", score_key: str = "scores"):
        self.root = Path(root)
        self.key = key
        self.present_key = present_key
        self.score_key = score_key

    def _path(self, video: Path) -> Path:
        for name in (f"{video.stem}.npz", f"{video.name}.npz"):
            candidate = self.root / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"subject mask file not found for {video.name} under {self.root}")

    def masks_for(self, video: Path, frames: Any, phrase: str) -> SubjectMasks:
        import numpy as np
        import torch

        from vbench_audit_core.inputs import sha256_file

        path = self._path(Path(video))
        with np.load(path) as payload:
            if self.key not in payload:
                raise ValueError(f"{path} has no '{self.key}' array")
            masks = torch.from_numpy(np.asarray(payload[self.key])).to(dtype=torch.float32)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            if masks.ndim != 4:
                raise ValueError(f"{path}: masks must be [T, h, w] or [T, I, h, w]")
            frames_count, instances = int(masks.shape[0]), int(masks.shape[1])
            if self.present_key in payload:
                present = torch.from_numpy(np.asarray(payload[self.present_key])).to(dtype=torch.bool)
            else:
                present = masks.flatten(2).sum(dim=-1) > 0
            if tuple(present.shape) != (frames_count, instances):
                raise ValueError(f"{path}: present must have shape [T, I]")
            scores = None
            if self.score_key in payload:
                scores = torch.from_numpy(np.asarray(payload[self.score_key])).to(dtype=torch.float32)
                if tuple(scores.shape) != (frames_count, instances):
                    raise ValueError(f"{path}: scores must have shape [T, I]")
        total_frames = int(frames.shape[0]) if hasattr(frames, "shape") else len(frames)
        if frames_count != total_frames:
            raise ValueError(f"{path}: {frames_count} mask frames for a {total_frames}-frame decode")
        return SubjectMasks(masks, present, phrase, str(path), sha256_file(path))


class Sam3SubjectMaskProvider:
    """In-process SAM 3 text-prompted localizer, one independent frame per call.

    Written against the released `facebookresearch/sam3` usage
    (`build_sam3_image_model` + `Sam3Processor.set_image` / `set_text_prompt`).
    Per-frame image mode is chosen over the video predictor on purpose: no
    tracker means no propagated identity, so every frame's evidence is
    independent, and the union of instances needs no cross-frame matching.

    REAL-MODEL PARITY IS NOT RUN here: the checkpoint is licence-gated on
    `facebook/sam3` and neither the package nor the weights live in this repo.
    """

    def __init__(self, device: Any, *, score_threshold: float = 0.5, model: Any | None = None):
        if not 0.0 <= float(score_threshold) <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        self.device = device
        self.score_threshold = float(score_threshold)
        if model is None:
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError("SAM 3 is not installed; install the pinned `sam3` package or use an offline mask directory") from exc
            model = build_sam3_image_model().to(device)
            self.processor = Sam3Processor(model)
        else:
            self.processor = model
        self.model = model

    def masks_for(self, video: Path, frames: Any, phrase: str) -> SubjectMasks:
        import torch
        from torchvision.transforms.functional import to_pil_image

        if not phrase.strip():
            raise ValueError("subject phrase must be a non-empty string")
        images = frames.detach().cpu() if hasattr(frames, "detach") else frames
        per_frame_masks: list[Any] = []
        per_frame_present: list[bool] = []
        with torch.no_grad():
            for image in images:
                pil = to_pil_image(image.to(torch.uint8) if image.dtype != torch.uint8 else image)
                state = self.processor.set_image(pil)
                output = self.processor.set_text_prompt(state=state, prompt=phrase)
                masks = output["masks"]
                scores = output.get("scores")
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                if scores is not None:
                    import numpy as np

                    keep = torch.as_tensor(np.asarray(scores)).reshape(-1).to(dtype=torch.float32) >= self.score_threshold
                    masks = masks[keep.to(masks.device)]
                masks = masks.to(dtype=torch.float32).cpu()
                per_frame_masks.append(masks)
                per_frame_present.append(int(masks.shape[0]) > 0)
        instances = max((int(mask.shape[0]) for mask in per_frame_masks), default=1) or 1
        height, width = int(per_frame_masks[0].shape[-2]), int(per_frame_masks[0].shape[-1])
        stacked = torch.zeros((len(per_frame_masks), instances, height, width), dtype=torch.float32)
        present = torch.zeros((len(per_frame_masks), instances), dtype=torch.bool)
        for index, mask in enumerate(per_frame_masks):
            count = int(mask.shape[0])
            if count:
                stacked[index, :count] = mask
                present[index, :count] = True
        return SubjectMasks(stacked, present, phrase, "sam3-image")


class OfficialDinoFeatureExtractor:
    """Locked VBench whole-frame DINO ViT-B/16 representation."""

    def __init__(self, device: Any, config: Mapping[str, Any], upstream: Path):
        from .backends.vbench import import_official_module

        self.module, self.upstream_state = import_official_module(upstream)
        self.device = device
        self.config = dict(config)
        if self.config.get("read_frame") is not False:
            raise ValueError("ordinary mp4 evaluation requires read_frame=false")
        self.model = self.module.torch.hub.load(
            repo_or_dir=self.config["repo_or_dir"], model=self.config["model"],
            source=self.config["source"], pretrained=False,
        )
        state_dict = self.module.torch.load(self.config["path"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device)
        self.model.eval()

    def extract(self, video: Path):
        images = self.module.load_video(str(video))
        return self.features_from_frames(images)

    def features_from_frames(self, images: Any):
        images = self.module.dino_transform(224)(images)
        features = []
        with self.module.torch.no_grad():
            for image in images:
                feature = self.model(image.unsqueeze(0).to(self.device))
                features.append(self.module.F.normalize(feature, dim=-1, p=2))
        if not features:
            raise ValueError("video contains no decoded frames")
        return self.module.torch.cat(features, dim=0)


class OfficialDinoPatchExtractor(OfficialDinoFeatureExtractor):
    """Same locked backbone and preprocessing, read out as per-frame patch tokens.

    The patch grid is rectangular because the official ``dino_transform`` resizes
    the short side to 224 *without* cropping, and DINO interpolates its
    positional encoding to the actual grid. This is the mask-only ablation
    backbone: it isolates "localised evidence" from "changed representation".
    """

    def extract_patches(self, video: Path):
        frames = self.module.load_video(str(video))
        return self.patches_from_frames(frames)

    def patches_from_frames(self, frames: Any):
        images = self.module.dino_transform(224)(frames)
        self.transformed_size = tuple(int(size) for size in images.shape[-2:])
        patch_size = int(getattr(self.model.patch_embed, "patch_size", 16))
        tokens = []
        grid = None
        with self.module.torch.no_grad():
            for image in images:
                layers = self.model.get_intermediate_layers(image.unsqueeze(0).to(self.device), n=1)
                # Pool the backbone's patch tokens, then L2-normalize the
                # frame subject vector in subject_evidence. No per-token norm.
                patches = layers[0][:, 1:, :]
                tokens.append(patches.squeeze(0))
                if grid is None:
                    height, width = int(image.shape[-2]), int(image.shape[-1])
                    grid = (height // patch_size, width // patch_size)
        if not tokens:
            raise ValueError("video contains no decoded frames")
        stacked = self.module.torch.stack(tokens, dim=0)
        if grid[0] * grid[1] != int(stacked.shape[1]):
            raise ValueError("patch grid does not match the token count")
        return stacked, grid
