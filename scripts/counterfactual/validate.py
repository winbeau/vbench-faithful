"""Independent verification of a generated VBench-CF dataset.

The strongest check here is `--replay`: every derived clip is re-derived from
its stored source and the transformation parameters recorded in the manifest,
then byte-compared against what is on disk.  Because the manifest stores the
detector box as a parameter, the replay does not need GRiT or any other model,
so the check is cheap enough to run over the whole dataset.

Duration is checked against a one-output-frame tolerance: an integer frame count
cannot represent every duration exactly (a 3.3 s/10 fps GIF resampled to 8 fps
lands within 0.075 s of its original), and the plan's requirement is that
duration is preserved, not that quantisation is impossible.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import build
from .common import (
    ROOT,
    CounterfactualError,
    decode_video,
    read_jsonl,
    sha256_file,
    write_json,
)

REQUIRED_FIELDS = (
    "schema_version",
    "dimension",
    "family",
    "base_id",
    "derived_id",
    "level",
    "expected_relation",
    "input_sha256",
    "output_sha256",
    "fps",
    "frame_count",
    "duration_s",
    "code_sha",
    "split",
    "prompt_id",
    "generator",
    "video_uid",
    "output_path",
)

# Families whose frame count must match the source exactly.
FRAME_PRESERVING = {
    "temporal_relocation",
    "directional_flip",
    "environment_coverage",
    "weakest_object_visibility",
    "temporal_jerk",
}
EXPECTED_LEVELS = {
    "fps_resampling": {"fps8", "fps6", "fps4", "fps2"},
    "temporal_relocation": {"clean", "corrupt_start", "corrupt_middle", "corrupt_end"},
    "filename_invariance": {"filename_correct", "filename_wrong", "filename_neutral"},
    "directional_flip": {"original", "horizontal_flip", "vertical_flip"},
    "environment_coverage": {f"coverage_{step:03d}" for step in (0, 25, 50, 75, 100)},
    "weakest_object_visibility": {
        f"occlusion_{step:03d}" for step in (0, 25, 50, 75, 100)
    } | {"conjunction_control"},
    "temporal_jerk": {
        "jerk_0_original",
        "jerk_1_duplicate",
        "jerk_2_duplicate_skip",
        "jerk_3_local_reverse",
        "jerk_4_multiple",
    },
}


class VerificationError(AssertionError):
    """Raised when the generated dataset violates its documented contract."""


def check_structure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    families: dict[str, set[str]] = defaultdict(set)
    per_split: Counter[tuple[str, str]] = Counter()
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise VerificationError(f"{row.get('derived_id')}: missing fields {missing}")
        if row["derived_id"] in seen:
            raise VerificationError(f"duplicate derived_id {row['derived_id']}")
        seen.add(row["derived_id"])
        if row["expected_relation"] not in (-1, 0, 1):
            raise VerificationError(f"{row['derived_id']}: bad expected_relation")
        families[row["family"]].add(row["level"])
        per_split[(row["dimension"], row["split"])] += 1
    for family, levels in families.items():
        expected = EXPECTED_LEVELS.get(family)
        if expected is None:
            raise VerificationError(f"unknown family {family}")
        if not levels <= expected:
            raise VerificationError(f"{family}: unexpected levels {sorted(levels - expected)}")
        if levels != expected:
            raise VerificationError(f"{family}: missing levels {sorted(expected - levels)}")
    return {
        "rows": len(rows),
        "distinct_derived_ids": len(seen),
        "families": {family: sorted(levels) for family, levels in sorted(families.items())},
        "rows_per_dimension_split": {
            f"{dimension}/{split}": count for (dimension, split), count in sorted(per_split.items())
        },
    }


def check_split_isolation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        prompts[(row["dimension"], row["prompt_id"])].add(row["split"])
    leaked = {key for key, splits in prompts.items() if len(splits) > 1}
    if leaked:
        raise VerificationError(f"prompt appears in both splits: {sorted(leaked)[:5]}")
    return {"prompts": len(prompts), "leaked": 0}


def check_files(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    missing = []
    mismatched = []
    for row in rows:
        path = root / row["output_path"]
        if not path.is_file():
            missing.append(row["output_path"])
            continue
        if sha256_file(path) != row["output_sha256"]:
            mismatched.append(row["output_path"])
        elif path.stat().st_size != row["bytes"]:
            mismatched.append(f"{row['output_path']} (size)")
    if missing:
        raise VerificationError(f"{len(missing)} derived clips are missing, e.g. {missing[:3]}")
    if mismatched:
        raise VerificationError(f"{len(mismatched)} clips changed bytes, e.g. {mismatched[:3]}")
    return {"files": len(rows), "missing": 0, "sha256_mismatch": 0}


def check_properties(rows: list[dict[str, Any]], root: Path, dataset_root: Path) -> dict[str, Any]:
    """Frame count, frame rate and duration must match the transformation's promise."""
    source_cache: dict[str, tuple[Any, Any]] = {}
    donor_cache: dict[str, Any] = {}
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)

    problems: list[str] = []
    for base_id, group in by_base.items():
        source = dataset_root / "videos" / group[0]["relative_video_path"]
        key = str(source)
        if key not in source_cache:
            source_cache[key] = decode_video(source)
        _, source_meta = source_cache[key]
        for row in group:
            path = root / row["output_path"]
            if row["family"] == "filename_invariance":
                if sha256_file(path) != row["input_sha256"]:
                    problems.append(f"{row['derived_id']}: filename variant changed bytes")
                continue
            from .common import probe_video

            meta = probe_video(path)
            if meta.fps != row["fps"]:
                problems.append(f"{row['derived_id']}: fps {meta.fps} != {row['fps']}")
            if meta.frame_count != row["frame_count"]:
                problems.append(
                    f"{row['derived_id']}: frames {meta.frame_count} != {row['frame_count']}"
                )
            tolerance = 1.0 / row["fps"] + 1e-6
            if abs(meta.duration_s - row["duration_s"]) > tolerance:
                problems.append(
                    f"{row['derived_id']}: duration {meta.duration_s} != {row['duration_s']}"
                )
            if row["family"] in FRAME_PRESERVING:
                if meta.frame_count != source_meta.frame_count:
                    problems.append(
                        f"{row['derived_id']}: frame count changed "
                        f"{meta.frame_count} != source {source_meta.frame_count}"
                    )
                if abs(meta.fps - source_meta.fps) > 1e-6:
                    problems.append(
                        f"{row['derived_id']}: fps changed {meta.fps} != source {source_meta.fps}"
                    )
            if row["family"] == "fps_resampling":
                if meta.frame_count > source_meta.frame_count:
                    problems.append(f"{row['derived_id']}: fps rung invented frames")
    if problems:
        raise VerificationError(f"{len(problems)} property violations, e.g. {problems[:5]}")
    return {"bases": len(by_base), "violations": 0, "duration_tolerance": "one output frame"}


def check_replay(
    rows: list[dict[str, Any]],
    root: Path,
    dataset_root: Path,
    sample: int = 0,
) -> dict[str, Any]:
    """Re-derive each variant from its recorded parameters and compare bytes."""
    import numpy as np

    from .common import encode_video

    selected = rows if not sample else rows[:: max(1, len(rows) // sample)]
    source_cache: dict[str, tuple[Any, Any]] = {}
    donor_cache: dict[str, Any] = {}
    scratch = root / "metadata" / "_replay"
    scratch.mkdir(parents=True, exist_ok=True)
    verified = 0
    problems: list[str] = []
    for row in selected:
        if row["family"] == "filename_invariance":
            verified += 1
            continue
        source = dataset_root / "videos" / row["relative_video_path"]
        key = str(source)
        if key not in source_cache:
            source_cache[key] = decode_video(source)
        frames, meta = source_cache[key]
        donor_frames = None
        if row["family"] == "environment_coverage":
            donor_rel = row["transformation_parameters"]["donor"]["relative_video_path"]
            if donor_rel not in donor_cache:
                donor_cache[donor_rel] = decode_video(dataset_root / "videos" / donor_rel)[0]
            donor_frames = donor_cache[donor_rel]
        try:
            replayed, fps = build.rederive(row, frames, meta, donor_frames)
        except CounterfactualError as error:
            problems.append(f"{row['derived_id']}: replay failed: {error}")
            continue
        out = scratch / f"{row['derived_id']}.mp4"
        encode_video(np.asarray(replayed), out, fps)
        if sha256_file(out) != row["output_sha256"]:
            problems.append(f"{row['derived_id']}: replay differs from stored clip")
        out.unlink(missing_ok=True)
        verified += 1
    try:
        scratch.rmdir()
    except OSError:
        pass
    if problems:
        raise VerificationError(f"{len(problems)} replay mismatches, e.g. {problems[:5]}")
    return {"replayed": verified, "mismatches": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--replay-sample", type=int, default=0, help="0 verifies every row")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.root / "manifest.jsonl")
    if not rows:
        raise VerificationError(f"no manifest rows under {args.root}")
    report: dict[str, Any] = {
        "schema_version": rows[0].get("schema_version"),
        "code_sha": rows[0].get("code_sha"),
        "structure": check_structure(rows),
        "split_isolation": check_split_isolation(rows),
        "files": check_files(rows, args.root),
        "properties": check_properties(rows, args.root, args.dataset_root),
    }
    if not args.skip_replay:
        report["replay"] = check_replay(rows, args.root, args.dataset_root, args.replay_sample)
    report["status"] = "VALID"
    destination = args.output or (args.root / "metadata" / "verification.json")
    write_json(destination, report)
    print(json.dumps({"status": "VALID", "rows": len(rows), "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
