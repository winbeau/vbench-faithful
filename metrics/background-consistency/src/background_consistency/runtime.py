"""Shared execution for official, aggregate-only and background-view variants."""
from __future__ import annotations

from pathlib import Path

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.schemas import VideoResult

from .algorithms import METHODS, score_views, suppress_foreground, temporal_score
from .models import build_model


def build_foreground_provider(config, *, device):
    from vbench_audit_models.foreground import CocoSubjectBoxDetector, MobileSamForegroundProvider, load_mobile_sam_predictor
    sam_path = Path(config["sam_checkpoint"])
    if sha256_file(sam_path) != config["sam_sha256"]:
        raise ValueError("MobileSAM checkpoint SHA256 mismatch")
    detector = CocoSubjectBoxDetector(Path(config["detector_checkpoint"]),
        expected_sha256=config["detector_sha256"], device=device,
        threshold=float(config.get("threshold", .8)), size=int(config.get("size", 512)))
    predictor = load_mobile_sam_predictor(sam_path, device)
    if config.get("cuda_graph", False):
        from vbench_audit_models.cuda_image_graph import capture_image_encoder
        capture_image_encoder(predictor.model.image_encoder)
    return MobileSamForegroundProvider(detector, predictor,
                                       weights_sha256=config["sam_sha256"])


def encode_views(encoder, frames, foreground, *, fill_rgb=(128, 128, 128)):
    import numpy as np
    import torch
    if not torch.equal(frames, frames.to(torch.uint8).to(frames.dtype)):
        raise ValueError("scoring views require exact native RGB bytes")
    images = frames.to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    views, fractions = {"global": encoder.features(frames)}, {}
    previous = images
    previous_features = views["global"]
    for key in ("frame", "union"):
        edited, fractions[key] = suppress_foreground(images, foreground, temporal_union=key == "union", fill_rgb=fill_rgb)
        if np.array_equal(edited, previous):
            views[key] = previous_features
        else:
            views[key] = encoder.features(torch.from_numpy(edited).permute(0, 3, 1, 2))
        previous, previous_features = edited, views[key]
    return views, fractions


def evaluate_videos(backend, videos, metadata, device, config, *, method):
    if method not in (*METHODS, 'patch_frame_calibrated', 'patch_frame_balanced'):
        raise ValueError(f"unknown background method: {method}")
    model_config = config.get("model", {})
    if "clip" not in model_config or not model_config["clip"].get("checkpoint"):
        return [VideoResult(str(v), "failed", error="model config requires [clip].checkpoint (local file)",
                            metric={"backend": backend, "variant": method}) for v in videos]
    if device is None:
        raise ValueError("an explicit device is required; there is no automatic CPU fallback")
    encoder = build_model(model_config["clip"], device=device)
    provider = None if method in ("official", "aggregation") else build_foreground_provider(model_config["localizer"], device=device)
    results = []
    for video in videos:
        try:
            frames = encoder.decode(video)
            details = {"backend": backend, "variant": method, "num_frames": len(frames),
                       "encoder": encoder.provenance, "video_sha256": sha256_file(video)}
            if provider is None:
                score = temporal_score(encoder.features(frames), aggregation="official" if method == "official" else "all_pairs")
            else:
                masks = provider.masks_for(frames)
                if method in ('patch_frame_calibrated', 'patch_frame_balanced'):
                    from .calibration import PATCH_GAIN, PATCH_OFFICIAL_WEIGHT, REPAIR_GAIN, patch_score_grid
                    from .patches import patch_views
                    cls, _, views, fractions, valid = patch_views(encoder, frames, masks)
                    scores = patch_score_grid(cls, views, valid)
                    score = scores['patch_frame_balanced_gain2' if method == 'patch_frame_balanced' else 'patch_frame_all_pairs_calibrated']
                    fractions = {k: v.cpu().numpy() for k, v in fractions.items()}
                    raw_score = (PATCH_OFFICIAL_WEIGHT*scores['patch_frame_official']+(1-PATCH_OFFICIAL_WEIGHT)*scores['patch_frame_all_pairs']
                                 if method == 'patch_frame_balanced' else scores['patch_frame_all_pairs'])
                    details.update(raw_patch_score=raw_score,
                                   patch_gain=PATCH_GAIN if method == 'patch_frame_balanced' else REPAIR_GAIN,
                                   aggregation=('half_official_half_all_pairs' if method == 'patch_frame_balanced' else 'all_pairs_same_dtype_cosine'),
                                   representation='area_weighted_background_patch_tokens')
                else:
                    views, fractions = encode_views(encoder, frames, masks)
                    score = score_views(views, fractions)[method]
                details.update(localizer=provider.provenance, localization=provider.last_diagnostics,
                               background_fraction={k: v.tolist() for k, v in fractions.items()})
            results.append(VideoResult(str(video), "succeeded", score=score, metric=details))
        except Exception as exc:
            results.append(VideoResult(str(video), "failed", error=f"{type(exc).__name__}: {exc}",
                                       metric={"backend": backend, "variant": method}))
    return results
