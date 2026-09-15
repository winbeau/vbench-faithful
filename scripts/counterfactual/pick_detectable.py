"""Re-select the detector-dependent dimensions with oversampling.

GRiT is caption-free and runs over low-resolution VBench generations, many of
which do not actually contain the object the prompt names (for example the
`tie/suitcase` clip depicts a mouse and an umbrella).  Plan section 11.2 makes
"usable boxes" an eligibility requirement, so those bases are legitimately
rejected -- but the fixed budget still has to be met.

This script ranks every eligible candidate by the same stable hash the selector
uses, runs GRiT once per candidate, and keeps the first `budget` candidates per
(dimension, split) that satisfy that dimension's eligibility rule.  The
non-detector dimensions are copied through unchanged, so their already-built
clips stay byte-identical.

Spatial Relationship needs a *directional* gate on top of box presence.  Plan
section 9.2 requires the original relation to be confirmed before the
`directional_flip` family may claim `score(original) > score(flip)`; without that
confirmation the expectation is only true when the generator happened to place A
on the required side of B, so a direction-sensitive repair is scored against a
coin flip.  The gate counts, over frames where the detector natively found both
targets, how often the ordered relation holds, and rejects a base unless it
holds in at least `--relation-min-frames` of them.  The gate is labelled on
every kept base: it uses the audited geometry on the shared detector's boxes, so
a repair that consumes those same boxes is being asked whether it can *rank* a
verified arrangement above its mirror, not whether it can discover the
arrangement from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .build import DETECTION_MIN_SCORE
from .common import ROOT, decode_video, read_jsonl, write_jsonl
from .grit import Detection, GritDetector, coverage, select_target, track_target
from .select_bases import BUDGET, SEED, load_manifest, load_prompts, ranked_pool

DETECTOR_DIMENSIONS = ("subject_consistency", "multiplt_object", "spatial_relationship")
DIRECTIONAL_DIMENSIONS = ("spatial_relationship",)

_ANNOTATION_CACHE: dict[Path, list[dict[str, Any]]] = {}


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if path not in _ANNOTATION_CACHE:
        _ANNOTATION_CACHE[path] = json.loads(path.read_text(encoding="utf-8"))
    return _ANNOTATION_CACHE[path]


def _spatial_relation_module():
    """The audited geometry itself, so the gate cannot drift from the metric."""

    source = ROOT / "metrics" / "spatial-relationship" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from spatial_relationship import relation as spatial_relation

    return spatial_relation


def detectable_targets(dimension: str, base: dict[str, Any], dataset_root: Path) -> list[str]:
    """The bare nouns a candidate must ground, read from the official annotations."""
    annotations = dataset_root / "annotations"
    if dimension == "subject_consistency":
        payload = _load_annotations(annotations / "Subject_consistency.json")
        subject = next(e["subject_en"] for e in payload if e["prompt_en"] == base["prompt_en"])
        return [str(subject).strip()]
    if dimension == "spatial_relationship":
        entry = _spatial_annotation(base, dataset_root)
        return [entry["object_a_en"], entry["object_b_en"]]
    payload = _load_annotations(annotations / "Multiplt_Object.json")
    objects = next(e["object_en"] for e in payload if e["prompt_en"] == base["prompt_en"])
    return [part.strip() for part in str(objects).split(" and ")]


def _spatial_annotation(base: dict[str, Any], dataset_root: Path) -> dict[str, str]:
    payload = _load_annotations(dataset_root / "annotations" / "Spatial_Relationship.json")
    entry = next((e for e in payload if e.get("prompt_en") == base["prompt_en"]), None)
    if entry is None:
        raise ValueError(f"no Spatial_Relationship annotation for {base['prompt_en']!r}")
    subject = str(entry.get("object_a_en") or "").strip()
    object_ = str(entry.get("object_b_en") or "").strip()
    if not subject or not object_:
        raise ValueError(
            f"spatial annotation for {base['prompt_en']!r} lacks the bare object nouns; "
            "the metric matches GRiT labels exactly, so articles would never match"
        )
    raw = str(entry.get("relationship_en") or base["parsed"]["relation"]).strip().lower()
    relation = _spatial_relation_module().normalize_relation(raw)
    return {"object_a_en": subject, "object_b_en": object_, "relationship": relation}


def spatial_query(base: dict[str, Any], dataset_root: Path) -> tuple[str, str, str]:
    """The ordered `(subject, object, relation)` the flip family must falsify."""

    entry = _spatial_annotation(base, dataset_root)
    return entry["object_a_en"], entry["object_b_en"], entry["relationship"]


def relation_frame_rate(
    detections: Sequence[Sequence[Detection]],
    subject: str,
    object_: str,
    relation: str,
    min_score: float = DETECTION_MIN_SCORE,
) -> tuple[float, int]:
    """How often the detector's own boxes satisfy the ordered relation.

    Only frames where *both* targets were detected natively are counted.  A box
    carried forward by `track_target` would let a detector drop-out masquerade as
    spatial evidence the detector never produced.
    """

    ordered_position_score = _spatial_relation_module().ordered_position_score
    hits = scored = 0
    for frame_detections in detections:
        first = select_target(frame_detections, subject, min_score)
        second = select_target(frame_detections, object_, min_score)
        if first is None or second is None:
            continue
        scored += 1
        if ordered_position_score(relation, first.box, second.box).score > 0:
            hits += 1
    return (hits / scored if scored else 0.0), scored


def detector_verdict(
    detector: GritDetector,
    base: dict[str, Any],
    dataset_root: Path,
    targets: list[str],
    directional: tuple[str, str, str] | None,
    min_relation_frames: float,
    min_co_detected_frames: float,
) -> tuple[bool, str, dict[str, float | int] | None]:
    """Box-presence eligibility, plus the directional gate for flip families."""

    frames, _ = decode_video(dataset_root / "videos" / base["relative_video_path"])
    detections = [detector.detect(frame) for frame in frames]
    for target in targets:
        tracked = track_target(detections, target, DETECTION_MIN_SCORE)
        if all(box is None for box in tracked) or coverage(tracked) < 1.0:
            return False, f"{target} not tracked in {base['video_uid']}", None
    if directional is None:
        return True, "", None
    subject, object_, relation = directional
    rate, scored = relation_frame_rate(detections, subject, object_, relation, DETECTION_MIN_SCORE)
    total = len(detections)
    stats: dict[str, float | int] = {
        "frame_rate": rate,
        "frames_scored": scored,
        "frames_total": total,
    }
    if scored == 0:
        return False, f"no frame natively detects both targets in {base['video_uid']}", stats
    co_detected = scored / total if total else 0.0
    if co_detected < min_co_detected_frames:
        return (
            False,
            f"both targets detected natively in only {co_detected:.2f} of {total} frames "
            f"in {base['video_uid']}",
            stats,
        )
    if rate < min_relation_frames:
        return (
            False,
            f"ordered relation {relation!r} holds in only {rate:.2f} of {scored} frames "
            f"in {base['video_uid']}",
            stats,
        )
    return True, "", stats


def load_relation_validity(path: Path) -> dict[str, float | None]:
    """Human validity check: `{base_id: rate}` or a bare list of verified ids."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): (None if value is None else float(value)) for key, value in payload.items()}
    if isinstance(payload, list):
        return {str(item): None for item in payload}
    raise ValueError(f"{path} must hold a JSON object {{base_id: rate}} or a list of base_ids")


def human_verdict(
    base: dict[str, Any], validity: dict[str, float | None]
) -> tuple[bool, str, dict[str, float | int] | None]:
    if base["base_id"] not in validity:
        return False, f"{base['base_id']} is not in the human validity list", None
    rate = validity[base["base_id"]]
    return True, "", {"frame_rate": rate} if rate is not None else None


def select_dimension(
    dimension: str,
    pool: list[dict[str, Any]],
    args: argparse.Namespace,
    detector: GritDetector,
    validity: dict[str, float | None],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the highest-ranked `budget` candidates that pass eligibility."""

    budget = dict(zip(("dev", "test"), BUDGET[dimension]))
    directional_mode = dimension in DIRECTIONAL_DIMENSIONS
    kept: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    scanned = 0
    observed_rates: list[float] = []
    for base in pool:
        split = base["split"]
        if len(kept[split]) >= budget[split]:
            continue
        scanned += 1
        try:
            if directional_mode and args.relation_oracle == "human":
                ok, why, stats = human_verdict(base, validity)
            else:
                targets = detectable_targets(dimension, base, args.dataset_root)
                directional = (
                    spatial_query(base, args.dataset_root)
                    if directional_mode and args.relation_oracle == "detector"
                    else None
                )
                ok, why, stats = detector_verdict(
                    detector, base, args.dataset_root, targets, directional,
                    args.relation_min_frames, args.min_co_detected_frames,
                )
        except (KeyError, StopIteration, ValueError) as error:
            ok, why, stats = False, f"{type(error).__name__}: {error}", None
        if directional_mode and stats and stats.get("frame_rate") is not None:
            observed_rates.append(float(stats["frame_rate"]))
        if not ok:
            if directional_mode:
                print(f"reject {dimension}/{base['video_uid']}: {why}", file=sys.stderr)
            continue
        del base["rank"]
        if directional_mode:
            base["relation_oracle"] = args.relation_oracle
            base["relation_frame_rate"] = None if stats is None else stats.get("frame_rate")
            base["relation_frames_scored"] = None if stats is None else stats.get("frames_scored")
            base["relation_frames_total"] = None if stats is None else stats.get("frames_total")
        kept[split].append(base)
    for split in ("dev", "test"):
        got = len(kept[split])
        if got < budget[split]:
            detail = ""
            if directional_mode and observed_rates:
                ordered = sorted(observed_rates)
                detail = (
                    f"; observed relation rates over {len(ordered)} scanned candidates: "
                    f"min {ordered[0]:.2f} median {ordered[len(ordered) // 2]:.2f} max {ordered[-1]:.2f}"
                    f" (relax --relation-min-frames / --min-co-detected-frames if the premise is"
                    f" satisfied less often than the threshold)"
                )
            raise SystemExit(
                f"{dimension}/{split}: only {got}/{budget[split]} eligible candidates "
                f"(oracle={args.relation_oracle}, scanned {scanned}){detail}"
            )
    return sorted(kept["dev"] + kept["test"], key=lambda b: (b["split"], b["video_uid"])), scanned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, default=ROOT / "output/counterfactual/bases.jsonl")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/counterfactual/bases.jsonl")
    parser.add_argument("--detector", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dimension",
        action="append",
        choices=DETECTOR_DIMENSIONS,
        help="re-select only this dimension (repeatable); unlisted dimensions are copied through",
    )
    parser.add_argument(
        "--relation-oracle",
        choices=("detector", "human", "none"),
        default="detector",
        help="how the `directional_flip` premise (original satisfies the relation) is confirmed",
    )
    parser.add_argument(
        "--relation-validity",
        type=Path,
        default=None,
        help="human verdicts: JSON {base_id: frame_rate} or a list of verified base_ids",
    )
    parser.add_argument(
        "--relation-min-frames",
        type=float,
        default=0.75,
        help="detector oracle: minimum fraction of both-detected frames that must satisfy the relation",
    )
    parser.add_argument(
        "--min-co-detected-frames",
        type=float,
        default=0.25,
        help="detector oracle: minimum fraction of frames that must natively detect both targets",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="write the per-dimension scan/rejection counts here as JSON",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dimensions = tuple(args.dimension) if args.dimension else DETECTOR_DIMENSIONS

    existing = read_jsonl(args.bases)
    prompts = load_prompts()
    manifest = load_manifest()

    validity: dict[str, float | None] = {}
    if args.relation_oracle == "human":
        if args.relation_validity is None:
            raise SystemExit("--relation-oracle human requires --relation-validity")
        validity = load_relation_validity(args.relation_validity)
    elif args.relation_oracle == "none":
        print(
            "warning: --relation-oracle none keeps the unverified directional family; "
            "the `original > flip` expectation is then not guaranteed to hold",
            file=sys.stderr,
        )

    # The two box-presence dimensions always need GRiT; the directional oracle
    # only changes whether Spatial Relationship candidates are also scanned with
    # it.  A human validity list therefore still requires the GRiT runtime.
    detector = GritDetector(
        "/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth"
        if args.detector == "default"
        else args.detector,
        device=args.device,
    )

    final: list[dict[str, Any]] = [
        base for base in existing if base["dimension"] not in dimensions
    ]
    report: dict[str, Any] = {}
    for dimension in dimensions:
        pool = ranked_pool(dimension, prompts, manifest, args.seed)
        kept, scanned = select_dimension(dimension, pool, args, detector, validity)
        for base in kept:
            base["ordinal"] = len(
                [x for x in final if x["dimension"] == dimension and x["split"] == base["split"]]
            )
            final.append(base)
        entry: dict[str, Any] = {
            "pool": len(pool),
            "scanned": scanned,
            "rejected": scanned - sum(1 for b in kept),
            "kept": {split: sum(1 for b in kept if b["split"] == split) for split in ("dev", "test")},
        }
        if dimension in DIRECTIONAL_DIMENSIONS:
            entry["relation_oracle"] = args.relation_oracle
            if args.relation_oracle == "detector":
                entry["relation_min_frames"] = args.relation_min_frames
                entry["min_co_detected_frames"] = args.min_co_detected_frames
        report[dimension] = entry

    write_jsonl(args.output, final)
    payload = {
        "status": "COMPLETE",
        "bases": len(final),
        "dimensions": list(dimensions),
        "seed": args.seed,
        "detector": args.detector,
        "per_dimension": report,
    }
    if args.summary is not None:
        # The scan/rejection counts are the fixture-eligibility record plan
        # section 11.2 asks for; printing them left them unreproducible once the
        # terminal scrolled away, so they are written next to the selection.
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
