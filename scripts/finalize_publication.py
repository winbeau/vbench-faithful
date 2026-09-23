#!/usr/bin/env python3
"""Publish the dataset card/catalog only after complete remote verification."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.verification.read_text())
    if report["errors"] or len(report["dimensions"]) != 16 or not report["status"].startswith("verified"):
        raise ValueError("A complete successful remote verification is required")
    catalog_file = args.stage / "catalog.json"
    catalog = json.loads(catalog_file.read_text())
    counts = Counter(e["dimension"] for e in report["counterfactual_experiments"])
    for dimension in catalog["dimensions"]:
        dimension["status"] = "published_with_source_reference_issues" if dimension["reference_issues"] else "published_verified"
        dimension["counterfactual_experiments"] = [e for e in report["counterfactual_experiments"]
                                                   if e["dimension"] == dimension["dimension"]]
    catalog.update(publication_status="published_verified_with_documented_upstream_issues",
                   verified_data_revision=report["verified_revision"],
                   verified_at=datetime.now(timezone.utc).isoformat(),
                   original_entries=report["original_entries"], unique_original_paths=report["unique_original_paths"],
                   annotation_rows=report["annotation_rows"], human_pairs=report["human_pairs"],
                   counterfactual_experiments=len(report["counterfactual_experiments"]),
                   code_repository="https://github.com/winbeau/vbench-repair",
                   code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    catalog_file.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    provenance = args.stage / "provenance/dataset-verification.json"
    provenance.parent.mkdir(exist_ok=True)
    shutil.copy2(args.verification, provenance)
    table = ["| Dimension | Original files | Annotation rows | Pairs with exact media references | CF experiments |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for d in report["dimensions"]:
        table.append(f"| {d['dimension']} | {d['origin_files']:,} | {d['annotation_rows']:,} | "
                     f"{d['usable_exact_pairs']:,}/{d['human_pairs']:,} | {counts[d['dimension']]} |")
    text = (args.stage / "README.md").read_text()
    text = text.replace("Publication is in progress. See `catalog.json` and `provenance/` for exact coverage.",
        f"Publication has been verified against the pinned data revision in `provenance/dataset-verification.json`. "
        f"The archive contains {report['original_entries']:,} dimension-local original entries "
        f"({report['unique_original_paths']:,} distinct source paths), {report['annotation_rows']:,} annotation rows, "
        f"{report['human_pairs']:,} normalized unordered comparisons, and "
        f"{len(report['counterfactual_experiments'])} versioned counterfactual experiment archives. "
        "The upstream reference issues below remain documented and unresolved.")
    if "## Coverage\n" in text:
        text = text.split("## Coverage\n", 1)[0].rstrip() + "\n"
    text += "\n## Coverage\n\n" + "\n".join(table) + "\n\n"
    text += ("Original counts include shared source videos in separate dimension directories. "
             "Pair usability here means both media references resolve exactly; reverse-label consistency "
             "is recorded separately in `reciprocal_labels`. Counterfactual archive members include "
             "originals, controls, generated clips, frame arrays, previews and records, so their member "
             "counts must not be interpreted as unique derived-video counts.\n\n"
             "## Download and integrity\n\n"
             "```bash\nHF_ENDPOINT=https://hf-mirror.com hf download xju-arlab/vbench-repair \\\n"
             "  --repo-type dataset --include 'dimensions/scene/**' --local-dir vbench-repair\n```\n\n"
             "If the mirror fails, retry with `HF_ENDPOINT=https://huggingface.co`. Original videos are "
             "directly accessible at the repository-relative paths in `origin/manifest.jsonl` and "
             "`human_preference/pairs.jsonl`. For one counterfactual experiment, download every "
             "`data-*.tar` shard and extract them into one empty directory for that experiment. "
             "`files.jsonl` maps each source-relative member to its archive and SHA-256. Keep different "
             "experiment versions in separate extraction directories.\n\n"
             "Verification checks every original file's remote size/hash, raw annotation byte identity, "
             "all shard hashes and index coverage, and downloaded original/complete-shard samples. "
             "Historical absolute machine paths are provenance; use the archive-relative indexes for access.\n")
    (args.stage / "README.md").write_text(text)
    if args.publish:
        from huggingface_hub import HfApi, CommitOperationAdd
        files = [args.stage / "README.md", catalog_file, provenance]
        result = HfApi(endpoint=args.endpoint).create_commit(repo_id=report["repo_id"], repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=p.relative_to(args.stage).as_posix(), path_or_fileobj=p) for p in files],
            commit_message="docs: finalize verified sixteen-dimension dataset catalog and archive coverage")
        print(json.dumps({"final_publication_revision": result.oid, "verified_data_revision": report["verified_revision"]}))


if __name__ == "__main__":
    main()
