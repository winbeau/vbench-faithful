"""Validate human-reviewed clean-frame prompts and freeze their canonical hashes."""
import argparse
import json
from pathlib import Path

from .subject_artifacts import artifact_path, read_jsonl, safe_output, write_jsonl


def freeze(reviewed: Path, dataset: Path, output: Path) -> int:
    from subject_consistency.localizer import prompt_sha256, validate_prompt

    expected = {}
    for entry in read_jsonl(dataset / "index.jsonl"):
        manifest = json.loads(artifact_path(dataset, entry["manifest"]).read_text())
        if manifest["status"] == "accepted":
            expected[manifest["base"]["video_uid"]] = manifest
    rows = read_jsonl(reviewed)
    if len(rows) != len(expected) or {row["video_uid"] for row in rows} != set(expected):
        raise ValueError("provide exactly one human-reviewed prompt for every accepted source clip")
    frozen = []
    for row in rows:
        value = {**row, "sha256": prompt_sha256(row)}
        validate_prompt(value)
        manifest = expected[row["video_uid"]]
        if row["source_frame_sha256"] != manifest["variants"]["clean"][0]["sha256"]:
            raise ValueError("reviewed frame does not match this dataset")
        if row["image_size"] != manifest["shape"][1:3]:
            raise ValueError("reviewed image size does not match this dataset")
        if row["phrase"] != manifest["base"]["subject_en"]:
            raise ValueError("use the official subject_en phrase for benchmark scoring")
        frozen.append(value)
    output = safe_output(output)
    if output.exists():
        raise FileExistsError(output)
    write_jsonl(output, sorted(frozen, key=lambda row: row["video_uid"]))
    return len(frozen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Frozen {freeze(args.reviewed, args.dataset, args.output)} human-reviewed prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
