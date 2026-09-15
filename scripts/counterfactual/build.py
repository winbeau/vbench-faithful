"""Materialise the VBench-CF counterfactual dataset.

For each selected base this module applies the dimension's deterministic family,
encodes every variant with identical settings (so the only difference between
variants is the transformation itself), and writes a manifest row carrying the
full provenance record required by plan section 3.2.

Level 0 of every family is a *re-encoded control*, not the source bytes: if the
control were a byte copy it would differ from its siblings by an extra
generation of compression, and any score gap could be blamed on the encoder
rather than the transformation.  The unmodified source clip is stored alongside
under `source/` so the control can still be checked against the true original.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import transforms
from .common import (
    ROOT,
    CounterfactualError,
    decode_video,
    encode_video,
    probe_video,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .transforms import Variant

SCHEMA_VERSION = "counterfactual-vbench/1"
SOURCE_DATASET = "VBench-1.0-human-preference"
LICENSE_NOTE = (
    "Derived from the public VBench 1.0 human-preference video package "
    "(Vchitect/VBench sampled_videos, upstream fd18b3d0). Redistribution follows "
    "the upstream release terms; see the dataset README before reuse."
)

# Which variant anchors `expected_relation` for each family.
FAMILY_REFERENCE: dict[str, str] = {
    "fps_resampling": "fps8",
    "temporal_relocation": "clean",
    "filename_invariance": "filename_correct",
    "directional_flip": "original",
    "environment_coverage": "coverage_100",
    "weakest_object_visibility": "occlusion_000",
    "temporal_jerk": "jerk_0_original",
}
FAMILY_NAMES: dict[str, str] = {
    "dynamics_degree": "fps_resampling",
    "subject_consistency": "temporal_relocation",
    "human_action": "filename_invariance",
    "spatial_relationship": "directional_flip",
    "scene": "environment_coverage",
    "multiplt_object": "weakest_object_visibility",
    "motion_smoothness": "temporal_jerk",
}

# GRiT is caption-free: the detector returns every labelled box in a frame and
# the caller matches labels against the official bare nouns.
Detector = Callable[[np.ndarray], list[Any]]

DETECTION_MIN_SCORE = 0.0


@dataclass
class BuildContext:
    dataset_root: Path
    output_root: Path
    code_sha: str
    detector: Detector | None = None
    annotation_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifest_index: dict[str, list[dict[str, str]]] = field(default_factory=dict)


def git_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def source_path(base: dict[str, Any], dataset_root: Path) -> Path:
    return dataset_root / "videos" / base["relative_video_path"]


def build_base(base: dict[str, Any], ctx: BuildContext) -> list[dict[str, Any]]:
    """Produce every variant for one base and return its manifest rows."""
    dimension = base["dimension"]
    family = FAMILY_NAMES[dimension]
    path = source_path(base, ctx.dataset_root)
    if not path.is_file():
        raise CounterfactualError(f"source video missing: {path}")

    input_sha = sha256_file(path)
    if family == "filename_invariance":
        return _build_filename_family(base, ctx, path, input_sha, family)

    frames, meta = decode_video(path)
    variants = _variants_for(base, ctx, frames, meta)
    reference_rank = _reference_rank(variants, family)

    # Store the unmodified source clip so base-vs-counterfactual comparisons do
    # not require the original VBench package.
    source_out = ctx.output_root / dimension / "source" / f"{base['base_id']}{path.suffix}"
    source_out.parent.mkdir(parents=True, exist_ok=True)
    if not source_out.exists():
        shutil.copy2(path, source_out)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        derived_id = f"{base['base_id']}__{variant.name}"
        out = ctx.output_root / dimension / "interventions" / family / f"{derived_id}.mp4"
        written = encode_video(variant.frames, out, variant.fps)
        rows.append(
            _row(
                base=base,
                ctx=ctx,
                family=family,
                variant=variant,
                derived_id=derived_id,
                output_path=out,
                input_sha=input_sha,
                output_meta=written,
                output_sha=sha256_file(out),
                expected_relation=_relation(variant.expected_rank, reference_rank),
                source_relative=base["relative_video_path"],
                source_meta=meta,
            )
        )
    return rows


def _build_filename_family(
    base: dict[str, Any], ctx: BuildContext, path: Path, input_sha: str, family: str
) -> list[dict[str, Any]]:
    """Human Action: byte-identical copies differing only in the filename."""
    from .select_bases import choose_wrong_action

    vocabulary = _action_vocabulary(ctx)
    correct = base["parsed"]["action"]
    wrong = choose_wrong_action(correct, vocabulary)
    variants = transforms.filename_invariance(correct, wrong)
    meta = probe_video(path)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        derived_id = f"{base['base_id']}__{variant.name}"
        out = ctx.output_root / base["dimension"] / "interventions" / family / f"{derived_id}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        if sha256_file(out) != input_sha:
            raise CounterfactualError(f"filename family changed bytes for {derived_id}")
        rows.append(
            _row(
                base=base,
                ctx=ctx,
                family=family,
                variant=variant,
                derived_id=derived_id,
                output_path=out,
                input_sha=input_sha,
                output_meta=meta,
                output_sha=input_sha,
                expected_relation=0,
                source_relative=base["relative_video_path"],
                source_meta=meta,
            )
        )
    return rows


def _action_vocabulary(ctx: BuildContext) -> set[str]:
    """Every `A person is <action>` action present in the frozen prompt split."""
    from .select_bases import load_prompts, parse_prompt

    vocabulary = set()
    for (dimension, _), text in load_prompts().items():
        if dimension != "human_action":
            continue
        try:
            vocabulary.add(parse_prompt("human_action", text)["action"])
        except Exception:
            continue
    return vocabulary


def _variants_for(
    base: dict[str, Any], ctx: BuildContext, frames: np.ndarray, meta
) -> list[Variant]:
    dimension = base["dimension"]
    if dimension == "dynamics_degree":
        return transforms.fps_resample(frames, meta)
    if dimension == "subject_consistency":
        caption = _subject_caption(base, ctx)
        boxes = _track_boxes(ctx, caption, frames, base)
        return transforms.temporal_relocation(frames, meta, boxes)
    if dimension == "spatial_relationship":
        return transforms.directional_flip(frames, meta, _relation_of(base, ctx))
    if dimension == "scene":
        donor_frames, donor = _donor_frames(base, ctx, meta)
        variants = transforms.environment_coverage(frames, donor_frames, meta)
        # Record the donor so the variant can be re-derived without re-running
        # the deterministic donor search (plan section 3.2 provenance).
        for variant in variants:
            variant.parameters["donor"] = {
                "video_uid": donor["video_uid"],
                "prompt_id": donor["prompt_id"],
                "relative_video_path": donor["relative_video_path"],
                "generator": donor["generator"],
            }
        return variants
    if dimension == "multiplt_object":
        targets = _targets_of(base, ctx)
        tracked = [_track_boxes(ctx, name, frames, base) for name in targets]
        weaker = _weaker_target(tracked)
        variants = transforms.weakest_object_visibility(frames, meta, tracked[weaker])
        variants.extend(
            transforms.temporal_conjunction_control(frames, meta, tracked[0], tracked[1])
        )
        return variants
    if dimension == "motion_smoothness":
        return transforms.temporal_jerk(frames, meta)
    raise CounterfactualError(f"no family wired for dimension {dimension}")


def _subject_caption(base: dict[str, Any], ctx: BuildContext) -> str:
    """Prefer the official `subject_en` label over parsing the prompt."""
    annotation = _annotation_for(base, ctx)
    subject = str(annotation.get("subject_en") or "").strip()
    if subject:
        return subject
    prompt = base["prompt_en"].strip()
    for article in ("a ", "an ", "the "):
        if prompt.lower().startswith(article):
            return prompt[len(article) :].split()[0]
    return prompt.split()[0]


def _annotation_for(base: dict[str, Any], ctx: BuildContext) -> dict[str, Any]:
    dimension = base["dimension"]
    if dimension not in ctx.annotation_cache:
        name = _annotation_filename(dimension)
        payload = json.loads((ctx.dataset_root / "annotations" / name).read_text(encoding="utf-8"))
        ctx.annotation_cache[dimension] = {
            str(entry.get("prompt_en", "")): entry for entry in payload
        }
    return ctx.annotation_cache[dimension].get(base["prompt_en"], {})


ANNOTATION_FILES = {
    "dynamics_degree": "Dynamics_Degree.json",
    "subject_consistency": "Subject_consistency.json",
    "human_action": "Human_Action.json",
    "spatial_relationship": "Spatial_Relationship.json",
    "scene": "Scene.json",
    "multiplt_object": "Multiplt_Object.json",
    "motion_smoothness": "Motion_Smoothness.json",
}


def _annotation_filename(dimension: str) -> str:
    try:
        return ANNOTATION_FILES[dimension]
    except KeyError as error:
        raise CounterfactualError(f"no annotation file known for {dimension}") from error


def _targets_of(base: dict[str, Any], ctx: BuildContext) -> list[str]:
    """The two bare target nouns, preferring the official `object_en` field.

    GRiT emits short category names, so the query must be the annotation's bare
    noun ("cat and dog"), never `prompt_en` ("a cat and a dog").
    """
    annotation = _annotation_for(base, ctx)
    raw = str(annotation.get("object_en") or "").strip()
    if raw:
        parts = [part.strip() for part in raw.split(" and ")]
        if len(parts) == 2 and all(parts):
            return parts
    return list(base["parsed"]["targets"])


def _relation_of(base: dict[str, Any], ctx: BuildContext) -> str:
    """Directional relation, preferring the official `relationship_en` field."""
    annotation = _annotation_for(base, ctx)
    raw = str(annotation.get("relationship_en") or "").strip().lower()
    for relation in ("left", "right", "top", "bottom"):
        if raw.endswith(relation) or raw == relation:
            return relation
    return str(base["parsed"]["relation"])


def _track_boxes(
    ctx: BuildContext, target: str, frames: np.ndarray, base: dict[str, Any]
) -> list[tuple[int, int, int, int]]:
    """Track one named target across every frame; reject the base if never seen."""
    if ctx.detector is None:
        raise CounterfactualError(
            f"{base['dimension']} needs GRiT boxes for {target!r}; pass --detector"
        )
    from .grit import coverage, track_target

    per_frame = [ctx.detector(frame) for frame in frames]
    boxes = track_target(per_frame, target, DETECTION_MIN_SCORE)
    if all(box is None for box in boxes):
        raise CounterfactualError(
            f"GRiT never detected {target!r} in {base['base_id']}"
        )
    held = coverage([None if box is None else box for box in boxes])
    if held < 1.0:
        raise CounterfactualError(f"incomplete tracking for {target!r}")
    return [box for box in boxes if box is not None]


def _weaker_target(tracked: Sequence[Sequence[tuple[int, int, int, int]]]) -> int:
    """Index of the less prominent target, by median tracked box area.

    Deterministic and independent of any metric score: only the detector's own
    boxes decide which object counts as the weak one (plan section 11.3).
    """
    medians = [
        float(np.median([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]))
        for boxes in tracked
    ]
    return int(np.argmin(medians))


def _donor_frames(base: dict[str, Any], ctx: BuildContext, meta) -> tuple[np.ndarray, dict[str, str]]:
    """Decode a wrong-scene donor clip from the same generator and split."""
    donor = _pick_donor(base, ctx)
    path = source_path(donor, ctx.dataset_root)
    frames, donor_meta = decode_video(path)
    if (donor_meta.width, donor_meta.height) != (meta.width, meta.height):
        raise CounterfactualError(
            f"donor resolution {donor_meta.width}x{donor_meta.height} != "
            f"target {meta.width}x{meta.height}"
        )
    return frames, donor


def _pick_donor(base: dict[str, Any], ctx: BuildContext) -> dict[str, str]:
    from .common import stable_sample
    from .select_bases import SEED, load_manifest

    if "scene" not in ctx.manifest_index:
        ctx.manifest_index["scene"] = load_manifest()
    candidates = [
        row
        for row in ctx.manifest_index["scene"]
        if row["dimension"] == "scene"
        and row["generator"] == base["generator"]
        and row["split"] == base["split"]
        and row["prompt_id"] != base["prompt_id"]
    ]
    if not candidates:
        raise CounterfactualError(f"no donor scene clip for {base['base_id']}")
    prompt_ids = sorted({row["prompt_id"] for row in candidates})
    donor_prompt = stable_sample(prompt_ids, 1, f"{SEED}:donor:{base['base_id']}")[0]
    pool = sorted(row["video_uid"] for row in candidates if row["prompt_id"] == donor_prompt)
    donor_uid = stable_sample(pool, 1, f"{SEED}:donorvideo:{base['base_id']}")[0]
    for row in candidates:
        if row["video_uid"] == donor_uid:
            return row
    raise CounterfactualError(f"donor lookup failed for {base['base_id']}")


def _reference_rank(variants: Sequence[Variant], family: str) -> int:
    name = FAMILY_REFERENCE.get(family)
    for variant in variants:
        if variant.name == name:
            return variant.expected_rank
    raise CounterfactualError(f"family {family} has no reference variant {name!r}")


def _relation(rank: int, reference_rank: int) -> int:
    if rank > reference_rank:
        return 1
    if rank < reference_rank:
        return -1
    return 0


def _row(
    *,
    base: dict[str, Any],
    ctx: BuildContext,
    family: str,
    variant: Variant,
    derived_id: str,
    output_path: Path,
    input_sha: str,
    output_meta,
    output_sha: str,
    expected_relation: int,
    source_relative: str,
    source_meta: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dimension": base["dimension"],
        "family": family,
        "base_id": base["base_id"],
        "derived_id": derived_id,
        "level": variant.name,
        "expected_rank": variant.expected_rank,
        "expected_relation": expected_relation,
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "fps": output_meta.fps,
        "frame_count": output_meta.frame_count,
        "duration_s": output_meta.duration_s,
        "width": output_meta.width,
        "height": output_meta.height,
        "bytes": output_path.stat().st_size,
        "output_path": str(output_path.relative_to(ctx.output_root)),
        "code_sha": ctx.code_sha,
        "split": base["split"],
        "prompt_id": base["prompt_id"],
        "prompt_en": base["prompt_en"],
        "generator": base["generator"],
        "group_id": base["group_id"],
        "video_uid": base["video_uid"],
        "relative_video_path": source_relative,
        "source_dataset": SOURCE_DATASET,
        "source_video_sha256": input_sha,
        "source_fps": getattr(source_meta, "fps", None),
        "source_frame_count": getattr(source_meta, "frame_count", None),
        "source_fps_source": getattr(source_meta, "fps_source", None),
        "transformation_parameters": variant.parameters,
        "transformation_note": variant.note,
        "manual_validity_status": "pending",
        "license_or_usage_note": LICENSE_NOTE,
    }


def rederive(
    row: dict[str, Any],
    source_frames: np.ndarray,
    source_meta,
    donor_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Replay one manifest row from its recorded parameters alone.

    Everything needed is stored in `transformation_parameters` -- including the
    detector box -- so validation never has to re-run GRiT.  Byte-comparing the
    replay against the stored clip is therefore a genuine end-to-end check of
    the transformation, not merely of the encoder.
    """
    family = row["family"]
    level = row["level"]
    parameters = row["transformation_parameters"]
    if family == "fps_resampling":
        variants = transforms.fps_resample(source_frames, source_meta)
    elif family == "temporal_relocation":
        variants = transforms.temporal_relocation(source_frames, source_meta, parameters["boxes"])
    elif family == "directional_flip":
        variants = transforms.directional_flip(source_frames, source_meta, parameters["relation"])
    elif family == "environment_coverage":
        if donor_frames is None:
            raise CounterfactualError("environment_coverage replay needs the donor frames")
        variants = transforms.environment_coverage(
            source_frames, donor_frames, source_meta, parameters["quadrant_order"]
        )
    elif family == "weakest_object_visibility":
        if level == "conjunction_control":
            variants = transforms.temporal_conjunction_control(
                source_frames, source_meta, parameters["boxes_a"], parameters["boxes_b"]
            )
        else:
            variants = transforms.weakest_object_visibility(
                source_frames, source_meta, parameters["boxes"]
            )
    elif family == "temporal_jerk":
        variants = transforms.temporal_jerk(source_frames, source_meta)
    else:
        raise CounterfactualError(f"no replay path for family {family}")
    for variant in variants:
        if variant.name == level:
            if variant.frames is None:
                raise CounterfactualError(f"{family}/{level} unexpectedly has no frames")
            return variant.frames, variant.fps
    raise CounterfactualError(f"family {family} has no level {level!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    bases = [json.loads(line) for line in Path(args.bases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.dimension:
        bases = [base for base in bases if base["dimension"] in set(args.dimension)]
    if args.limit:
        bases = bases[: args.limit]
    ctx = BuildContext(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        code_sha=git_sha(),
        detector=load_detector(args),
    )
    ctx.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, base in enumerate(bases, start=1):
        try:
            produced = build_base(base, ctx)
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            failures.append(
                {
                    "base_id": base["base_id"],
                    "dimension": base["dimension"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[{index}/{len(bases)}] FAIL {base['base_id']}: {error}")
            continue
        rows.extend(produced)
        print(f"[{index}/{len(bases)}] {base['base_id']}: {len(produced)} variants")

    manifest = ctx.output_root / "manifest.jsonl"
    previous = read_jsonl(manifest) if args.merge else []
    if previous:
        # Merge by derived_id so a second invocation (for example the
        # GRiT-dependent dimensions) extends the dataset instead of replacing
        # the rows an earlier run already produced.
        merged: dict[str, dict[str, Any]] = {row["derived_id"]: row for row in previous}
        for row in rows:
            merged[row["derived_id"]] = row
        rows = [merged[key] for key in sorted(merged)]
    write_jsonl(manifest, rows)

    failure_path = ctx.output_root / "metadata" / "build_failures.json"
    known_failures = {
        entry["base_id"]: entry
        for entry in (json.loads(failure_path.read_text(encoding="utf-8")).get("failures", [])
                      if args.merge and failure_path.is_file() else [])
    }
    for failure in failures:
        known_failures[failure["base_id"]] = failure
    # A base that succeeds on a later run is no longer a failure.
    for row in rows:
        known_failures.pop(row["base_id"], None)
    failures = [known_failures[key] for key in sorted(known_failures)]
    write_json(failure_path, {"failures": failures})

    summary = {
        "schema_version": SCHEMA_VERSION,
        "bases_attempted": len(bases),
        "bases_ok": len(bases) - len([f for f in failures if f["base_id"] in {b["base_id"] for b in bases}]),
        "bases_failed": len([f for f in failures if f["base_id"] in {b["base_id"] for b in bases}]),
        "derived_rows": len(rows),
        "code_sha": ctx.code_sha,
        "manifest": str(manifest),
        "failures": failures,
    }
    write_json(ctx.output_root / "metadata" / "build_summary.json", summary)
    return summary


def load_detector(args: argparse.Namespace) -> Detector | None:
    if not getattr(args, "detector", None):
        return None
    from .grit import DEFAULT_WEIGHTS, GritDetector

    weights = DEFAULT_WEIGHTS if args.detector == "default" else args.detector
    return GritDetector(weights, device=args.detector_device).detect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dimension", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--detector", help="path to GRiT weights directory, or 'default'")
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge into an existing manifest instead of replacing it",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])
    return 0 if summary["bases_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
