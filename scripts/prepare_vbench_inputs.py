#!/usr/bin/env python3
"""Build explicit standard-suite video metadata without inferring targets from scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_common import ROOT, dimension, readl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "configs/reproduction/VBench_full_info.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true", help="explicit partial suite; missing files remain listed")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    subjects = {}
    for row in readl(ROOT / "configs/subject-repair/stability_official1440_manifest_v8.jsonl"):
        prompt, subject = row["prompt_en"], row["subject_en"]
        if prompt in subjects and subjects[prompt] != subject:
            raise ValueError("Conflicting frozen subject annotations")
        subjects[prompt] = subject
    metadata = json.loads(args.metadata.read_text())
    rows, missing = [], []
    for item in metadata:
        prompt = item["prompt_en"]
        for index in range(5):
            matches = [args.video_dir / f"{prompt}-{index}{suffix}" for suffix in (".mp4", ".gif")]
            found = [p for p in matches if p.is_file()]
            if len(found) > 1:
                raise ValueError(f"Ambiguous video formats: {prompt}-{index}")
            if not found:
                missing.append(f"{prompt}-{index}")
                continue
            rows.append({"video": str(found[0].resolve()), "prompt": prompt,
                         "dimensions": [dimension(d) for d in item["dimension"]],
                         "auxiliary_info": item.get("auxiliary_info", {}), "subject_en": subjects.get(prompt)})
    if missing and not args.allow_missing:
        raise ValueError(f"Incomplete standard suite ({len(missing)} missing); use --allow-missing only for an explicitly partial run")
    if not rows:
        raise ValueError("No standard-suite video found")
    write_json(args.output, {"videos": rows, "standard_suite_complete": not missing,
                             "missing_standard_inputs": missing, "subject_metadata": "frozen official1440 annotations; no prompt guessing"})
    print(json.dumps({"videos": len(rows), "missing_standard_inputs": len(missing)}))


if __name__ == "__main__":
    main()
