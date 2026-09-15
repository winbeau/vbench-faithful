"""Generate the dataset card and publish VBench-CF to the Hugging Face Hub.

The card is derived from `manifest.jsonl`, so the published counts, code SHA and
per-family tables cannot drift from the artefacts that were actually built.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_json
from .validate import EXPECTED_LEVELS

FAMILY_DESCRIPTION = {
    "fps_resampling": (
        "Dynamic Degree",
        "Resample one trajectory to 8/6/4/2 fps with duration held fixed.",
        "Invariance: every rung should score the same.",
    ),
    "temporal_relocation": (
        "Subject Consistency",
        "One fixed subject-region corruption placed at the start, middle and end.",
        "clean > corrupted, and the three positions should tie.",
    ),
    "filename_invariance": (
        "Human Action",
        "Byte-identical copies named with a correct, wrong and neutral action.",
        "Invariance: only the filename changes.",
    ),
    "directional_flip": (
        "Spatial Relationship",
        "Mirror the axis named by the ordered relation, prompt unchanged.",
        "original > flip.",
    ),
    "environment_coverage": (
        "Scene",
        "2x2 grid mixing a wrong-scene donor with target-scene quadrants at 0-100%.",
        "Monotone increasing in target coverage.",
    ),
    "weakest_object_visibility": (
        "Multiple Objects",
        "Alpha-blend the tracked weaker target towards its local mean at 0-100%.",
        "Monotone decreasing as the weak target disappears.",
    ),
    "temporal_jerk": (
        "Motion Smoothness",
        "Duplicate, skip and locally reverse short segments; frame count and rate fixed.",
        "Monotone decreasing in perturbation severity.",
    ),
}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    per_dimension = Counter()
    per_split = Counter()
    clips = 0
    bytes_total = 0
    for row in rows:
        per_dimension[row["dimension"]] += 1
        per_split[(row["dimension"], row["split"])] += 1
        clips += 1
        bytes_total += int(row.get("bytes") or 0)
    return {
        "rows": len(rows),
        "bases": len({row["base_id"] for row in rows}),
        "clips": clips,
        "bytes": bytes_total,
        "per_dimension": dict(sorted(per_dimension.items())),
        "per_split": {f"{dimension}/{split}": count for (dimension, split), count in sorted(per_split.items())},
        "families": {
            family: {
                "rows": len(items),
                "levels": sorted({item["level"] for item in items}),
                "dimension": FAMILY_DESCRIPTION.get(family, ("", "", ""))[0],
            }
            for family, items in sorted(families.items())
        },
        "code_sha": rows[0].get("code_sha") if rows else None,
    }


def render_card(summary: dict[str, Any], repo_id: str) -> str:
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- video-classification",
        "tags:",
        "- video",
        "- benchmark",
        "- vbench",
        "- counterfactual",
        "- metamorphic-testing",
        "---",
        "",
        "# VBench-CF: counterfactual probes for VBench 1.0",
        "",
        "VBench-CF is a **metamorphic test set** for VBench 1.0 metrics. Rather than",
        "arguing about which of two videos is better, each clip comes from a",
        "deterministic transformation whose effect on the score is known in advance.",
        "If a metric violates that relation, the violation is a property of the metric,",
        "not of anybody's taste.",
        "",
        "Every family contains an unmodified control plus one or more transformed",
        "levels derived from it. The transformation is applied to the *video*; the",
        "prompt and the dimension contract stay fixed.",
        "",
        "## Families",
        "",
        "| Dimension | Transformation | Expected relation | Clips |",
        "|---|---|---|---:|",
    ]
    for family, info in summary["families"].items():
        dimension, description, expectation = FAMILY_DESCRIPTION.get(family, (family, "", ""))
        lines.append(
            f"| {dimension} | {description} | {expectation} | {info['rows']} |"
        )
    lines += [
        "",
        "`Overall Consistency` (condition substitution) is deliberately absent: it",
        "requires human-authored prompt conditions and cannot be generated from",
        "metadata alone.",
        "",
        "## Layout",
        "",
        "```text",
        "manifest.jsonl                     one row per derived clip, full provenance",
        "metadata/bases.jsonl               the selected bases and why they qualify",
        "metadata/verification.json         integrity, property and replay results",
        "metadata/build_summary.json        counts, code SHA, rejected bases",
        "<dimension>/source/<base_id>.*     the unmodified source clip",
        "<dimension>/interventions/<family>/<base_id>__<level>.mp4",
        "```",
        "",
        "`manifest.jsonl` fields include `dimension`, `family`, `base_id`,",
        "`derived_id`, `level`, `expected_relation` (-1/0/+1 against the family",
        "reference), `expected_rank`, `split`, `prompt_en`, `generator`,",
        "`input_sha256`, `output_sha256`, `fps`, `frame_count`, `duration_s`,",
        "`transformation_parameters` and `code_sha`.",
        "",
        "## Reading the levels",
        "",
        "`expected_relation` is stated against the family's reference level:",
        "",
    ]
    for family, info in summary["families"].items():
        reference = {
            "fps_resampling": "fps8",
            "temporal_relocation": "clean",
            "filename_invariance": "filename_correct",
            "directional_flip": "original",
            "environment_coverage": "coverage_100",
            "weakest_object_visibility": "occlusion_000",
            "temporal_jerk": "jerk_0_original",
        }.get(family, "?")
        lines.append(f"- `{family}`: reference `{reference}`, levels {', '.join(f'`{l}`' for l in info['levels'])}")
    lines += [
        "",
        "## Construction guarantees",
        "",
        "- **Deterministic.** No random sampling and no generated frames. Level 0 of",
        "  every family is re-encoded with the same encoder settings as its siblings,",
        "  so the only difference between levels is the transformation itself.",
        "- **Replayable.** Every clip is re-derived from its recorded parameters and",
        "  byte-compared during verification; the detector box is stored as a",
        "  parameter so verification needs no GPU.",
        "- **Score-independent.** Bases are chosen from the frozen prompt split by",
        "  stable hash. No Official or Audit score is consulted at any point, so the",
        "  set cannot be biased towards a desired outcome.",
        "- **Split-safe.** The frozen prompt split is reused and is prompt-disjoint,",
        "  so a prompt never appears in both `dev` and `test`.",
        "",
        "## Counts",
        "",
        f"- bases: **{summary['bases']}**",
        f"- derived clips: **{summary['clips']}**",
        f"- built from code SHA `{summary['code_sha']}`",
        "",
        "| Dimension | dev clips | test clips |",
        "|---|---:|---:|",
    ]
    dimensions = sorted({key.split("/")[0] for key in summary["per_split"]})
    for dimension in dimensions:
        dev = summary["per_split"].get(f"{dimension}/dev", 0)
        test = summary["per_split"].get(f"{dimension}/test", 0)
        lines.append(f"| {dimension} | {dev} | {test} |")
    lines += [
        "",
        "## Source and licence",
        "",
        "Derived from the public **VBench 1.0 human-preference** video package",
        "(`Vchitect/VBench`, `sampled_videos`, upstream commit",
        "`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`). Original clips are redistributed",
        "unaltered under `<dimension>/source/` so that each counterfactual can be",
        "compared against its own base.",
        "",
        "The upstream release terms govern reuse of these clips and of anything",
        "derived from them. This dataset is a research artefact for auditing metric",
        "behaviour; it is not a human-preference benchmark and carries no new human",
        "labels.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="xjuIcthub/counterfactual-vbench")
    parser.add_argument("--write-card", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Publish VBench-CF counterfactual dataset")
    args = parser.parse_args()

    rows = read_jsonl(args.root / "manifest.jsonl")
    if not rows:
        raise SystemExit(f"no manifest rows under {args.root}")
    summary = summarise(rows)
    write_json(args.root / "metadata" / "dataset_summary.json", summary)
    card = render_card(summary, args.repo_id)
    if args.write_card or args.upload:
        (args.root / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "families"}, indent=2))
    print(f"families: {sorted(summary['families'])}")
    print(f"payload: {summary['bytes'] / 1e9:.2f} GB")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(args.root),
            commit_message=args.commit_message,
            ignore_patterns=["metadata/_replay/*"],
        )
        print(json.dumps({"status": "UPLOADED", "repo_id": args.repo_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
