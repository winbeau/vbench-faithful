"""Reuse MovieGen and existing LoRA training prompts without importing labels."""
import argparse
import json
from pathlib import Path

from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json, write_jsonl
from .subject_semantics import prepare_prompt_pool


def source_records(moviegen: Path, training_files: list[Path]) -> tuple[list[dict], list[dict]]:
    records, sources = [], []
    text = moviegen / "benchmark/MovieGenVideoBench.txt"
    audio = moviegen / "benchmark/MovieGenAudioBenchSfx.jsonl"
    expected = {text: "ae02a4bbebcb55067dbeea369e43d52a4817d57fd0ac223071b9aa236ac027c4",
                audio: "6e86e6ef12bb6c8f0f1410262a626386d4a539b0000075fc73ba345d02e38acc"}
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"MovieGen input differs from the pinned revision: {path}")
    # Both audio manifests share the same video prompts; no duplicate pool.
    for path, prompts in ((text, [s.strip() for s in text.read_text().splitlines() if s.strip()]),
                          (audio, [row["video_prompt"] for row in read_jsonl(audio)])):
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "moviegen"})
        for prompt in prompts:
            import hashlib
            group = hashlib.sha256(" ".join(prompt.casefold().split()).encode()).hexdigest()
            records.append({"prompt": prompt, "source": "MovieGenBench@bab7753fce1108def167a1efb4c1ae1d69b8f03a",
                            "source_group": f"moviegen:{group}", "seen_in_training": False})
    for path in training_files:
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "existing_model_training"})
        for row in read_jsonl(path):
            prompt = row.get("input", {}).get("prompt", row.get("prompt"))
            group = row.get("group_id")
            if not isinstance(prompt, str) or not group or not row.get("source"):
                raise ValueError(f"training source lacks prompt/group provenance: {path}")
            records.append({"prompt": prompt, "source": row["source"],
                            "source_group": f"{row['source']}:{group}", "seen_in_training": True})
    return records, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moviegen", type=Path, required=True)
    parser.add_argument("--training-prompts", type=Path, nargs="*", default=[])
    parser.add_argument("--review-size", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, sources = source_records(args.moviegen, args.training_prompts)
    pool, review = prepare_prompt_pool(records, review_size=args.review_size)
    output = new_output(args.output)
    write_jsonl(output / "prompts.jsonl", pool)
    write_jsonl(output / "human_review.pending.jsonl", review)
    write_json(output / "manifest.json", {"sources": sources, "unique_prompts": len(pool),
        "train": sum(row["split"] == "train" for row in pool), "test": sum(row["split"] == "test" for row in pool),
        "human_review_reserved": len(review), "human_review_completed": 0,
        "split_before_rewriting": True, "grouping": "source group union normalized prompt equality",
        "vocabulary_sha256": sha256_file(ROOT / "configs/subject-repair/vocabulary.json"),
        "prompts_sha256": sha256_file(output / "prompts.jsonl"),
        "human_review_pending_sha256": sha256_file(output / "human_review.pending.jsonl"),
        "llm_human_agreement": "NOT RUN", "head_accuracy": "NOT RUN", "training": "NOT RUN"})
    print(f"Prepared {len(pool)} prompts and {len(review)} pending human reviews in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
