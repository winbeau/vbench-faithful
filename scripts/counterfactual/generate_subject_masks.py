"""Freeze SegFormer-B0 + GrabCut construction masks; never scoring masks.

Run as ``python -m scripts.counterfactual.generate_subject_masks --help``.
All pretrained assets are local, outside the repository. No network fallback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, sha256_file
from .region_discrimination import RejectedBase, refine_grabcut, window_indices
from .subject_artifacts import (new_output, object_sha256, read_jsonl, safe_id, upstream_frames,
                                write_json, write_jsonl, write_npz, write_png_sequence)


CONFIG = ROOT / "configs/subject-repair"


def deterministic_setup() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)


class SegFormerConstructionLocalizer:
    def __init__(self, model_dir: Path, *, device: str):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        model_dir = model_dir.expanduser().resolve()
        if ROOT == model_dir or ROOT in model_dir.parents:
            raise ValueError("weights must be outside the repository")
        labels = json.loads((CONFIG / "ade20k_labels.json").read_text())
        config_path = model_dir / "config.json"
        if sha256_file(config_path) != labels["source"]["sha256"]:
            raise ValueError("SegFormer config does not match the frozen model revision")
        weights = sorted([*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")])
        if not weights:
            raise ValueError("no local SegFormer weights")
        self.processor = SegformerImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
        self.device = device
        self.torch = torch
        self.id2label = {int(k): v for k, v in labels["id2label"].items()}
        files = {p.name: sha256_file(p) for p in [config_path, model_dir / "preprocessor_config.json", *weights]}
        self.provenance = {"role": "construction", "family": "segformer", **labels["source"],
                           "files_sha256": files, "weights_sha256": object_sha256({p.name: files[p.name] for p in weights}),
                           "parameter_count": sum(p.numel() for p in self.model.parameters()),
                           "input_size": [512, 512], "device": device,
                           "torch_version": torch.__version__, "opencv_version": cv2.__version__,
                           "grabcut": {"kernel": [11, 11], "shape": "ellipse", "rng_seed": 0, "iterations": 4},
                           "components": {"minimum_area": 20, "person_largest": True, "person_hole_area_below": 200}}

    def labels_for(self, frame: np.ndarray) -> np.ndarray:
        values = self.processor(images=frame, return_tensors="pt", size={"height": 512, "width": 512})
        values = {key: value.to(self.device) for key, value in values.items()}
        if tuple(values["pixel_values"].shape[-2:]) != (512, 512):
            raise ValueError("SegFormer input must be 512 x 512")
        with self.torch.inference_mode():
            logits = self.model(**values).logits
            resized = self.torch.nn.functional.interpolate(logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
        return resized.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def localize_frames(frames: np.ndarray, subject: str, localizer, mapping: dict) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    labels_by_name = {name: key for key, name in localizer.id2label.items()}
    target_ids = [labels_by_name[name] for name in mapping["classes"][subject]]
    stuff_ids = [labels_by_name[name] for name in mapping["unoccupied_stuff_labels"]]
    masks, occupied, reasons = [], [], []
    for frame in frames:
        labels = localizer.labels_for(frame)
        target = np.isin(labels, target_ids).astype(np.uint8)
        why = []
        try:
            mask = refine_grabcut(frame, target, person=subject == "person")
        except RejectedBase as exc:
            mask = np.zeros(frame.shape[:2], np.uint8)
            why.append(exc.reason)
        area = float(mask.mean())
        if area < .01:
            why.append("mask_area_below_1_percent")
        if area > .50:
            why.append("mask_area_above_50_percent")
        masks.append(mask)
        occupied.append((~np.isin(labels, stuff_ids) & (mask == 0)).astype(np.uint8))
        reasons.append(why)
    return np.stack(masks), np.stack(occupied), reasons


def generate(bases: list[dict], video_root: Path, output: Path, mapping: dict, localizer,
             *, decoder=upstream_frames) -> dict:
    counts: Counter = Counter()
    frame_counts: Counter = Counter()
    rows = []
    seen = set()
    for base in bases:
        uid = safe_id(base["video_uid"])
        base_id = safe_id(base["base_id"])
        if uid in seen:
            raise ValueError(f"duplicate video_uid: {uid}")
        seen.add(uid)
        entry = {"schema_version": 1, "role": "construction", "base": base,
                 "class_map_sha256": object_sha256(mapping), "localizer": localizer.provenance,
                 "rejection_reasons": [], "status": "rejected"}
        subject = base.get("subject_en")
        try:
            if not subject:
                raise RejectedBase("subject_en_missing")
            if subject not in mapping["classes"]:
                raise RejectedBase("class_not_mapped")
            relative = Path(base["relative_video_path"])
            video = (video_root / relative).resolve()
            if relative.is_absolute() or video_root.resolve() not in video.parents:
                raise ValueError("source video path escapes input root")
            if not video.is_file():
                raise RejectedBase("source_video_missing")
            frames = decoder(video)
            entry["source_video_sha256"] = sha256_file(video)
            entry["decode"] = "VBench@fd18b3d.load_video; all frames; native resolution; RGB uint8"
            window_indices(len(frames), "middle")
            masks, other_instances, why = localize_frames(frames, subject, localizer, mapping)
            entry["shape"] = list(masks.shape)
            entry["area_pixels"] = masks.sum(axis=(1, 2)).tolist()
            entry["area_ratio"] = masks.mean(axis=(1, 2)).tolist()
            entry["frame_rejection_reasons"] = why
            entry["rejection_reasons"] = sorted({r for frame in why for r in frame})
            frame_counts.update(r for frame in why for r in frame)
            mask_path = output / "masks" / f"{uid}.npz"
            write_npz(mask_path, masks=masks, other_instances=other_instances,
                      area=np.asarray(entry["area_pixels"]),
                      metadata_json=np.asarray(json.dumps(entry, sort_keys=True)))
            entry["mask_file"] = {"path": str(mask_path.relative_to(output)), "sha256": sha256_file(mask_path)}
            if not entry["rejection_reasons"]:
                entry["frames"] = write_png_sequence(output, f"frames/{uid}", frames)
                entry["status"] = "accepted"
        except RejectedBase as exc:
            entry["rejection_reasons"].append(exc.reason)
        counts.update(entry["rejection_reasons"])
        write_json(output / "manifests" / f"{base_id}.json", entry)
        rows.append({"base_id": base_id, "video_uid": uid, "status": entry["status"],
                     "rejection_reasons": entry["rejection_reasons"], "manifest": f"manifests/{base_id}.json"})
    summary = {"total_bases": len(rows), "accepted": sum(row["status"] == "accepted" for row in rows),
               "rejected": sum(row["status"] != "accepted" for row in rows),
               "base_rejection_counts": dict(counts), "frame_rejection_counts": dict(frame_counts),
               "absence_scope": mapping["absence_scope"], "role": "construction"}
    write_jsonl(output / "index.jsonl", rows)
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, default=CONFIG / "bases.jsonl")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, default=CONFIG / "class_map.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.class_map.read_text())  # JSON is a strict YAML subset.
    bases = read_jsonl(args.bases)
    deterministic_setup()
    localizer = SegFormerConstructionLocalizer(args.model_dir, device=args.device)
    output = new_output(args.output)
    print(json.dumps(generate(bases, args.video_root, output, mapping, localizer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
