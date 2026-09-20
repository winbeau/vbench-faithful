"""Replay saved teacher responses and verify provenance without network calls."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, read_jsonl
from .subject_semantics import normalized_prompt, validate_label


def validate(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, digest in manifest["files_sha256"].items():
        if sha256_file(artifact_path(root, relative)) != digest:
            raise ValueError(f"silver artifact hash mismatch: {relative}")
    vocabulary_path = ROOT / "configs/subject-repair/vocabulary.json"
    if sha256_file(vocabulary_path) != manifest["vocabulary_sha256"]:
        raise ValueError("vocabulary differs from the frozen silver run")
    vocabulary = set(json.loads(vocabulary_path.read_text())["subjects"])
    rows = read_jsonl(root / "silver.jsonl")
    groups, text_splits = {}, {}
    counts = Counter()
    for row in rows:
        for mapping, key in ((groups, row["group_id"]), (text_splits, normalized_prompt(row["prompt"]))):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("source group or rewritten text leaked across train/test")
            mapping[key] = row["split"]
        counts[row["quality"]] += 1
        if row["quality"] != "silver":
            continue
        validate_label(row["label"], row["prompt"], vocabulary, original_prompt=row.get("original_prompt"))
        request = json.loads((root / "requests" / f"{row['request_id']}.json").read_text())
        user = json.loads(request["user"])
        if user != {"prompt": row["prompt"]}:
            raise ValueError("teacher input must contain only the raw prompt text")
        response = json.loads((root / "responses" / f"{row['request_id']}.json").read_text())
        if json.loads(response["raw"]) != row["label"]:
            raise ValueError("saved silver label differs from teacher response")
        identity = response["response"]
        if identity["model"] != manifest["config"]["expected_response_model"] or identity["system_fingerprint"] != manifest["config"]["expected_system_fingerprint"]:
            raise ValueError("teacher identity drift")
    return {"records": len(rows), "quality_counts": dict(counts), "source_group_split_leaks": 0,
            "raw_prompt_only": True, "llm_human_agreement": "NOT RUN", "head_accuracy": "NOT RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
