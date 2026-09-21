"""Build and verify a lossless region_discrimination dataset from frozen masks."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, sha256_file
from .region_discrimination import LEVELS, POSITIONS, RejectedBase, region_discrimination
from .subject_artifacts import (artifact_path, new_output, read_jsonl, read_png_sequence,
                                write_json, write_jsonl, write_npz, write_png_sequence)


def load_construction(root: Path, entry: dict):
    if entry["role"] != "construction" or entry["localizer"]["family"] != "segformer":
        raise ValueError("construction must use SegFormer, never DINO or scoring masks")
    path = artifact_path(root, entry["mask_file"]["path"])
    if sha256_file(path) != entry["mask_file"]["sha256"]:
        raise ValueError("construction mask hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        masks, others = data["masks"], data["other_instances"]
    return read_png_sequence(root, entry["frames"]), masks, others


def build_entry(row: dict, construction: Path, output: Path, protocol_path: Path, operator: str,
                gaussian_reference_short_side: int | None = None) -> dict:
    """One independent base; output paths are disjoint after ID validation."""
    cv2.setNumThreads(1)
    path = artifact_path(construction, row["manifest"])
    source = json.loads(path.read_text())
    base = source["base"]
    if row["base_id"] != base["base_id"] or row["status"] != source["status"]:
        raise ValueError("construction index and manifest disagree")
    base_id = base["base_id"]
    manifest = {"schema_version": 2, "family": "region_discrimination", "base": base,
                    "construction_manifest_sha256": sha256_file(path),
                    "protocol_sha256": sha256_file(protocol_path),
                    "status": "rejected", "rejection_reasons": list(source["rejection_reasons"]),
                    "variants": {}, "positions": {}, "opencv_version": cv2.__version__,
                    "isolation": {"construction": "segformer+grabcut", "scoring_localizer": "mobilesam",
                                  "encoder": "dino_vitb16", "construction_masks_allowed_for_scoring": False}}
    if source["status"] == "accepted":
        frames, masks, others = load_construction(construction, source)
        families = {}
        try:
            # All declared positions must be constructible before writing any
            # derived clip; never retain a score-favorable position only.
            for position in POSITIONS:
                families[position] = region_discrimination(frames, masks, others,
                        subject=base["subject_en"], position=position, operator=operator,
                        gaussian_reference_short_side=gaussian_reference_short_side)
        except RejectedBase as exc:
            manifest["rejection_reasons"].append(exc.reason)
        else:
            mask_path = output / "construction_masks" / f"{base_id}.npz"
            write_npz(mask_path, masks=masks, other_instances=others)
            manifest["construction_masks"] = {"path": str(mask_path.relative_to(output)), "sha256": sha256_file(mask_path)}
            manifest["construction_localizer"] = source["localizer"]
            manifest["source_video_sha256"] = source["source_video_sha256"]
            manifest["shape"] = list(frames.shape)
            manifest["area_ratio"] = masks.mean(axis=(1, 2)).tolist()
            manifest["variants"]["clean"] = write_png_sequence(output, f"clips/{base_id}/clean", frames)
            for position, family in families.items():
                manifest["positions"][position] = {"parameters": family.parameters, "proofs": family.proofs}
                for level in LEVELS[1:]:
                    key = f"{position}/{level}"
                    manifest["variants"][key] = write_png_sequence(output, f"clips/{base_id}/{key}", family.frames[level])
            manifest["status"] = "accepted"
    relative = f"manifests/{base_id}.json"
    write_json(output / relative, manifest)
    return {"base_id": base_id, "video_uid": base["video_uid"],
            "source_prompt_id": base.get("prompt_id", base["prompt_en"]),
            "status": manifest["status"], "rejection_reasons": manifest["rejection_reasons"],
            "manifest": relative, "manifest_sha256": sha256_file(output / relative)}


def build(construction: Path, output: Path, *, operator: str = "gaussian", workers: int = 1,
          protocol_path: Path | None = None, gaussian_reference_short_side: int | None = None) -> dict:
    protocol_path = protocol_path or ROOT / "configs/subject-repair/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('gaussian_reference_short_side') != gaussian_reference_short_side:
        raise ValueError('blur normalization must match its separately frozen protocol')
    rows = read_jsonl(construction / "index.jsonl")
    if len({r["base_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate bases in construction input")
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        results = [build_entry(row, construction, output, protocol_path, operator, gaussian_reference_short_side) for row in rows]
    else:
        from itertools import repeat
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(build_entry, rows, repeat(construction), repeat(output), repeat(protocol_path), repeat(operator),
                                    repeat(gaussian_reference_short_side)))
    counts = Counter(reason for row in results for reason in row["rejection_reasons"])
    summary = {"family": "region_discrimination", "total_bases": len(results),
               "accepted": sum(row["status"] == "accepted" for row in results),
               "rejected": sum(row["status"] != "accepted" for row in results),
               "rejection_counts": dict(counts), "operator": operator,
               "background_support": "subject_mask_complement", "equal_area_claim": False,
               "natural_preference_measurement": "NOT RUN", "real_model_parity": "NOT RUN"}
    write_jsonl(output / "index.jsonl", results)
    if gaussian_reference_short_side is not None:
        summary['gaussian_reference_short_side'] = gaussian_reference_short_side
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    write_json(output / "summary.json", summary)
    return summary


def verify(dataset: Path) -> dict:
    """Recompute every accepted artifact/proof from its stored clean pixels."""
    checked_frames = 0
    rows = read_jsonl(dataset / "index.jsonl")
    for row in rows:
        path = artifact_path(dataset, row["manifest"])
        if sha256_file(path) != row["manifest_sha256"]:
            raise ValueError("manifest hash mismatch")
        manifest = json.loads(path.read_text())
        if manifest["protocol_sha256"] != sha256_file(dataset / "protocol.json"):
            raise ValueError("preregistered protocol changed")
        if manifest["status"] != "accepted":
            continue
        clean = read_png_sequence(dataset, manifest["variants"]["clean"])
        mask_file = manifest["construction_masks"]
        mask_path = artifact_path(dataset, mask_file["path"])
        if sha256_file(mask_path) != mask_file["sha256"]:
            raise ValueError("mask hash mismatch")
        with np.load(mask_path, allow_pickle=False) as data:
            masks, others = data["masks"], data["other_instances"]
        for position in manifest["positions"]:
            saved = manifest["positions"][position]
            rebuilt = region_discrimination(clean, masks, others, subject=manifest["base"]["subject_en"],
                position=position, operator=saved["parameters"]["operator"],
                background_mode="complement" if saved["parameters"]["background_support"] == "subject_mask_complement" else "mirror",
                gaussian_reference_short_side=saved['parameters'].get('gaussian_reference_short_side'))
            if saved != {"parameters": rebuilt.parameters, "proofs": rebuilt.proofs}:
                raise AssertionError("stored proof or parameters do not match replay")
            for level in LEVELS[1:]:
                actual = read_png_sequence(dataset, manifest["variants"][f"{position}/{level}"])
                if not np.array_equal(actual, rebuilt.frames[level]):
                    raise AssertionError("derived pixels do not match deterministic replay")
                checked_frames += len(actual)
    return {"bases": len(rows), "verified_corrupted_frames": checked_frames, "outside_mask_changed_pixels": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--operator", choices=("gaussian", "mosaic"), default="gaussian")
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU workers for lossless per-base construction")
    parser.add_argument('--protocol',type=Path,help='Separate preregistration for a changed construction rule')
    parser.add_argument('--gaussian-reference-short-side',type=int)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        if not args.construction or not args.output:
            parser.error("--construction and --output are required when building")
        print(json.dumps(build(args.construction, new_output(args.output), operator=args.operator, workers=args.workers,
                               protocol_path=args.protocol,gaussian_reference_short_side=args.gaussian_reference_short_side), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
