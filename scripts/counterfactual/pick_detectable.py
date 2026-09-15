"""Re-select the detector-dependent dimensions with oversampling.

GRiT is caption-free and runs over low-resolution VBench generations, many of
which do not actually contain the object the prompt names (for example the
`tie/suitcase` clip depicts a mouse and an umbrella).  Plan section 11.2 makes
"usable boxes" an eligibility requirement, so those bases are legitimately
rejected -- but the fixed budget still has to be met.

This script ranks every eligible candidate by the same stable hash the selector
uses, runs GRiT once per candidate, and keeps the first `budget` candidates per
(dimension, split) for which the required targets are detected in every frame.
The non-detector dimensions are copied through unchanged, so their already-built
clips stay byte-identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .build import DETECTION_MIN_SCORE
from .common import ROOT, decode_video, read_jsonl, write_jsonl
from .grit import GritDetector, coverage, track_target
from .select_bases import BUDGET, SEED, load_manifest, load_prompts, ranked_pool

DETECTOR_DIMENSIONS = ("subject_consistency", "multiplt_object")


def detectable_targets(dimension: str, base: dict[str, Any], dataset_root: Path) -> list[str]:
    """The bare nouns a candidate must ground, read from the official annotations."""
    annotations = dataset_root / "annotations"
    if dimension == "subject_consistency":
        payload = json.loads((annotations / "Subject_consistency.json").read_text(encoding="utf-8"))
        subject = next(e["subject_en"] for e in payload if e["prompt_en"] == base["prompt_en"])
        return [str(subject).strip()]
    payload = json.loads((annotations / "Multiplt_Object.json").read_text(encoding="utf-8"))
    objects = next(e["object_en"] for e in payload if e["prompt_en"] == base["prompt_en"])
    return [part.strip() for part in str(objects).split(" and ")]


def candidate_ok(
    detector: GritDetector, base: dict[str, Any], dataset_root: Path, targets: list[str]
) -> tuple[bool, str]:
    frames, _ = decode_video(dataset_root / "videos" / base["relative_video_path"])
    detections = [detector.detect(frame) for frame in frames]
    for target in targets:
        tracked = track_target(detections, target, DETECTION_MIN_SCORE)
        if all(box is None for box in tracked) or coverage(tracked) < 1.0:
            return False, f"{target} not tracked in {base['video_uid']}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, default=ROOT / "output/counterfactual/bases.jsonl")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/counterfactual/bases.jsonl")
    parser.add_argument("--detector", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    existing = read_jsonl(args.bases)
    prompts = load_prompts()
    manifest = load_manifest()
    detector = GritDetector(
        "/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth"
        if args.detector == "default"
        else args.detector,
        device=args.device,
    )

    final: list[dict[str, Any]] = [
        base for base in existing if base["dimension"] not in DETECTOR_DIMENSIONS
    ]
    report: dict[str, Any] = {}
    for dimension in DETECTOR_DIMENSIONS:
        budget = dict(zip(("dev", "test"), BUDGET[dimension]))
        pool = ranked_pool(dimension, prompts, manifest, args.seed)
        kept: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
        scanned = 0
        for base in pool:
            split = base["split"]
            if len(kept[split]) >= budget[split]:
                continue
            scanned += 1
            targets = detectable_targets(dimension, base, args.dataset_root)
            ok, why = candidate_ok(detector, base, args.dataset_root, targets)
            if ok:
                del base["rank"]
                kept[split].append(base)
        for split in ("dev", "test"):
            got = len(kept[split])
            if got < budget[split]:
                raise SystemExit(f"{dimension}/{split}: only {got}/{budget[split]} detectable")
        for base in sorted(kept["dev"] + kept["test"], key=lambda b: (b["split"], b["video_uid"])):
            base["ordinal"] = len([x for x in final if x["dimension"] == dimension and x["split"] == base["split"]])
            final.append(base)
        report[dimension] = {"scanned": scanned, "kept": {k: len(v) for k, v in kept.items()}}

    write_jsonl(args.output, final)
    print(json.dumps({"status": "COMPLETE", "bases": len(final), "per_dimension": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
