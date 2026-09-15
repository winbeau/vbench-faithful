"""Build the independent FPS validation set for Dynamic Degree (P1.3).

The fixed exponent 0.5 was chosen from a bootstrap CI over the same 40 bases the
counterfactual family uses, so it has not been validated on data that played no
part in choosing it.  This builds a prompt-disjoint holdout: every eligible
candidate prompt for `dynamics_degree` is ranked with the same stable hash the
selector uses, the 40 in-use bases are removed, and the next 30 become the
validation set.

The prompt split is the frozen E0 split, so the holdout is also disjoint from the
counterfactual test prompts by construction, not merely by exclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, read_jsonl, write_jsonl
from .select_bases import SEED, load_manifest, load_prompts, ranked_pool

VALIDATION_BASES = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, default=ROOT / "output/counterfactual/bases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output/counterfactual/bases_dynamic_validation.jsonl")
    parser.add_argument("--count", type=int, default=VALIDATION_BASES)
    parser.add_argument("--seed", type=int, default=SEED + 1)
    args = parser.parse_args()

    used = {
        row["video_uid"]
        for row in read_jsonl(args.bases)
        if row["dimension"] == "dynamics_degree"
    }
    prompts = load_prompts()
    manifest = load_manifest()
    pool = ranked_pool("dynamics_degree", prompts, manifest, SEED)

    holdout = [base for base in pool if base["video_uid"] not in used]
    if len(holdout) < args.count:
        raise SystemExit(
            f"only {len(holdout)} unused candidates for {args.count} validation bases; "
            "the dynamic_degree pool is prompt-limited"
        )
    chosen = holdout[: args.count]

    used_prompts = {row["prompt_id"] for row in read_jsonl(args.bases) if row["dimension"] == "dynamics_degree"}
    overlap = {base["prompt_id"] for base in chosen} & used_prompts
    if overlap:
        raise SystemExit(f"validation prompts overlap the counterfactual set: {sorted(overlap)[:3]}")

    rows = []
    for index, base in enumerate(chosen):
        rows.append(
            {
                "base_id": base["base_id"],
                "dimension": "dynamics_degree",
                "split": "validation",
                "ordinal": index,
                "prompt_id": base["prompt_id"],
                "prompt_en": base["prompt_en"],
                "group_id": base["group_id"],
                "generator": base["generator"],
                "video_uid": base["video_uid"],
                "relative_video_path": base["relative_video_path"],
                "parsed": {},
            }
        )
    # Sanity: this set must not reuse a base or an original prompt.
    if len({row["video_uid"] for row in rows}) != len(rows):
        raise SystemExit("validation set contains duplicate bases")
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "validation_bases": len(rows),
                "unused_candidates": len(holdout),
                "prompt_disjoint_from_counterfactual": True,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
