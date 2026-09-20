"""Join official subjects and optionally extend the published 25 bases to 60.

Only source metadata, the existing prompt split, and a fixed hash are read.
No localization output or metric score enters the selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .common import ROOT, sha256_file
from .select_bases import load_manifest, load_prompts, ranked_pool
from .subject_artifacts import read_jsonl, safe_output, write_jsonl


def prepare(annotations: Path, count: int = 25) -> list[dict]:
    if not 25 <= count <= 60:
        raise ValueError("base count must be between 25 and 60")
    official = {}
    for row in json.loads(annotations.read_text()):
        prompt, subject = row["prompt_en"], row.get("subject_en")
        if prompt in official and official[prompt] != subject:
            raise ValueError(f"conflicting official subject for {prompt}")
        official[prompt] = subject
    bases = [row for row in read_jsonl(ROOT / "configs/counterfactual/bases_published.jsonl")
             if row["dimension"] == "subject_consistency"]
    used = {row["prompt_id"] for row in bases}
    pool = ranked_pool("subject_consistency", load_prompts(), load_manifest())
    pool.sort(key=lambda row: hashlib.sha256(f"20260920:{row['prompt_id']}".encode()).hexdigest())
    for row in pool:
        if len(bases) >= count:
            break
        if row["prompt_id"] not in used:
            bases.append(row)
            used.add(row["prompt_id"])
    if len(bases) != count:
        raise ValueError("insufficient unique source prompts")
    digest = sha256_file(annotations)
    for row in bases:
        row["subject_en"] = official.get(row["prompt_en"])
        row["subject_source_sha256"] = digest
    return bases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = safe_output(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = prepare(args.annotations, args.count)
    write_jsonl(output, rows)
    print(f"Frozen {len(rows)} unique source prompts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
