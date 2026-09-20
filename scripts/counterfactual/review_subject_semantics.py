"""Prepare a blind human review form or score a completed 150–200-item review."""
import argparse
import json
from pathlib import Path

from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json
from .subject_semantics import reviewed_agreement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--silver", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vocabulary = json.loads((ROOT / "configs/subject-repair/vocabulary.json").read_text())["subjects"]
    output = new_output(args.output)
    if args.reviewed:
        if not args.silver or not args.pool:
            parser.error("--silver and --pool are required with --reviewed")
        reviewed = read_jsonl(args.reviewed)
        reserved = read_jsonl(args.pool / "human_review.pending.jsonl")
        if {row["sample_id"] for row in reviewed} != {row["sample_id"] for row in reserved}:
            raise ValueError("evaluate the entire reserved human subset; do not remove difficult or invalid teacher cases")
        report = reviewed_agreement(reviewed, read_jsonl(args.silver), set(vocabulary),
                                    predictions=read_jsonl(args.predictions) if args.predictions else None)
        report["review_sha256"] = sha256_file(args.reviewed)
        report["silver_sha256"] = sha256_file(args.silver)
        write_json(output / "agreement.json", report)
        print(json.dumps(report, indent=2))
    else:
        if not args.pool:
            parser.error("--pool is required to prepare the review form")
        rows = read_jsonl(args.pool / "human_review.pending.jsonl")
        template = Path(__file__).with_name("subject_semantics_review.html").read_text()
        document = template.replace("/*ROWS*/[]", json.dumps(rows).replace("</", "<\\/"))
        document = document.replace("/*VOCABULARY*/[]", json.dumps(vocabulary))
        (output / "human_review.html").write_text(document)
        write_json(output / "manifest.json", {"reserved": len(rows), "completed": 0,
                                              "source_sha256": sha256_file(args.pool / "human_review.pending.jsonl"),
                                              "llm_human_agreement": "NOT RUN", "head_accuracy": "NOT RUN"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
