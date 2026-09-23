#!/usr/bin/env python3
"""Verify published original videos, counterfactual shards, labels and models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(args):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint=args.endpoint)
    info = api.dataset_info(args.repo_id)
    files = {x.path: x for x in api.list_repo_tree(args.repo_id, repo_type="dataset", revision=info.sha,
             recursive=True) if hasattr(x, "size")}
    errors, original_rows, summaries, experiments, archive_samples = [], [], [], [], []
    metadata_checked = 0
    for local in sorted(args.stage.rglob("*")):
        if not local.is_file() or ".cache" in local.parts:
            continue
        name = local.relative_to(args.stage).as_posix()
        item = files.get(name)
        if item is None or item.size != local.stat().st_size:
            errors.append({"path": name, "reason": "metadata_missing_or_size"})
            continue
        content = local.read_bytes()
        if item.lfs:
            identical = item.lfs.sha256 == hashlib.sha256(content).hexdigest()
        else:
            identical = item.blob_id == hashlib.sha1(
                f"blob {len(content)}\0".encode() + content).hexdigest()
        if not identical:
            errors.append({"path": name, "reason": "metadata_content"})
        metadata_checked += 1
    for manifest in sorted(args.stage.glob("dimensions/*/origin/manifest.jsonl")):
        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        original_rows.extend(rows)
        dimension = manifest.parents[1].name
        for row in rows:
            item = files.get(row["path"])
            if item is None:
                errors.append({"path": row["path"], "reason": "missing"})
            elif item.size != row["bytes"]:
                errors.append({"path": row["path"], "reason": "size"})
            elif not row["sha256"] or not item.lfs or item.lfs.sha256 != row["sha256"]:
                errors.append({"path": row["path"], "reason": "sha256"})
        hp = manifest.parents[1] / "human_preference"
        raw = next(p for p in hp.glob("*.json") if p.name not in {"sources.json"})
        relative = raw.relative_to(args.stage).as_posix()
        downloaded = Path(hf_hub_download(args.repo_id, relative, repo_type="dataset", revision=info.sha,
                                        endpoint=args.endpoint))
        if sha(downloaded) != sha(raw):
            errors.append({"path": relative, "reason": "annotation_bytes_changed"})
        pairs = [json.loads(l) for l in (hp / "pairs.jsonl").read_text().splitlines() if l.strip()]
        issues = [json.loads(l) for l in (hp / "reference_issues.jsonl").read_text().splitlines() if l.strip()]
        for pair in pairs:
            if pair["usable_exact_pair"]:
                for key in ["video_a", "video_b"]:
                    if pair[key] not in files:
                        errors.append({"path": pair[key], "reason": "missing_pair_video"})
        summaries.append({"dimension": dimension, "annotation_rows": len(json.loads(raw.read_text())),
                          "human_pairs": len(pairs), "usable_exact_pairs": sum(p["usable_exact_pair"] for p in pairs),
                          "source_reference_issues": len(issues), "origin_files": len(rows),
                          "origin_bytes": sum(r["bytes"] for r in rows)})
    for name in sorted(files):
        if "/counterfactual/" not in name or not name.endswith("/manifest.json"):
            continue
        p = Path(hf_hub_download(args.repo_id, name, repo_type="dataset", revision=info.sha, endpoint=args.endpoint))
        record = json.loads(p.read_text())
        if record.get("schema") != "vbench-repair-counterfactual/1":
            continue
        for shard in record["archives"]:
            item = files.get(shard["path"])
            if item is None or item.size != shard["bytes"] or not item.lfs or item.lfs.sha256 != shard["sha256"]:
                errors.append({"path": shard["path"], "reason": "archive_identity"})
        index_name = name.rsplit("/", 1)[0] + "/files.jsonl"
        p = Path(hf_hub_download(args.repo_id, index_name, repo_type="dataset", revision=info.sha, endpoint=args.endpoint))
        members = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if len(members) != record["source_files"] or sum(m["bytes"] for m in members) != record["source_bytes"]:
            errors.append({"path": index_name, "reason": "member_coverage"})
        keys = {(m["archive"], m["member"]) for m in members}
        if len(keys) != len(members):
            errors.append({"path": index_name, "reason": "duplicate_members"})
        archive_names = {s["path"] for s in record["archives"]}
        for member in members:
            path = PurePosixPath(member["member"])
            if path.is_absolute() or ".." in path.parts or member["archive"] not in archive_names:
                errors.append({"path": index_name, "reason": "unsafe_or_unassigned_member", "member": str(path)})
        for shard in record["archives"]:
            indexed = [m for m in members if m["archive"] == shard["path"]]
            if len(indexed) != shard["files"]:
                errors.append({"path": shard["path"], "reason": "per_shard_member_count"})
            if any(Path(m["member"]).suffix.lower() in {".mp4", ".gif"} for m in indexed):
                archive_samples.append((shard, indexed))
        experiments.append({"dimension": record["dimension"], "version": record["version"],
                            "manifest": name, "source_files": len(members), "source_bytes": record["source_bytes"],
                            "video_members": sum(Path(m["member"]).suffix.lower() in {".mp4", ".gif", ".webm"} for m in members),
                            "shards": len(record["archives"]), "archive_bytes": sum(s["bytes"] for s in record["archives"])})
    expected = set()
    for spec in args.spec:
        for exp in json.loads(spec.read_text())["experiments"]:
            expected.add((exp["dimension"], exp["version"]))
    actual = {(e["dimension"], e["version"]) for e in experiments}
    for dimension, version in sorted(expected - actual):
        errors.append({"dimension": dimension, "version": version, "reason": "missing_experiment_manifest"})
    if len(summaries) != 16:
        errors.append({"reason": "dimension_coverage", "actual": len(summaries)})
    # Download byte-exact samples from both source storage forms.
    samples = []
    for git_backed in [True, False]:
        row = next(r for r in original_rows if bool(r.get("git_blob_sha1")) == git_backed)
        p = Path(hf_hub_download(args.repo_id, row["path"], repo_type="dataset", revision=info.sha, endpoint=args.endpoint))
        ok = sha(p) == row["sha256"]
        samples.append({"path": row["path"], "sha256_verified": ok})
        if not ok:
            errors.append({"path": row["path"], "reason": "download_hash"})
    # Inspect every byte/member in one downloaded media-bearing shard.
    if archive_samples:
        shard, indexed = min(archive_samples, key=lambda pair: pair[0]["bytes"])
        p = Path(hf_hub_download(args.repo_id, shard["path"], repo_type="dataset", revision=info.sha, endpoint=args.endpoint))
        expected_members = {m["member"]: m for m in indexed}
        actual_members, ok = set(), sha(p) == shard["sha256"]
        with tarfile.open(p, "r") as tf:
            for member in tf:
                actual_members.add(member.name)
                expected_member = expected_members.get(member.name)
                if not member.isfile() or expected_member is None:
                    ok = False
                    continue
                with tf.extractfile(member) as stream:
                    h = hashlib.sha256()
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        h.update(block)
                ok &= member.size == expected_member["bytes"] and h.hexdigest() == expected_member["sha256"]
        ok &= actual_members == set(expected_members)
        samples.append({"path": shard["path"], "sha256_verified": bool(ok), "members_verified": len(actual_members)})
        if not ok:
            errors.append({"path": shard["path"], "reason": "archive_download_members"})
    report = {"schema": "vbench-repair-verification/1", "repo_id": args.repo_id, "verified_revision": info.sha,
              "verified_at": datetime.now(timezone.utc).isoformat(), "private": info.private,
              "metadata_files_verified": metadata_checked,
              "status": "verified_with_documented_upstream_reference_issues" if not errors else "failed",
              "errors": errors, "dimensions": summaries, "original_entries": len(original_rows),
              "unique_original_paths": len({r["source_path"] for r in original_rows}),
              "unique_original_sha256": len({r["sha256"] for r in original_rows}),
              "annotation_rows": sum(s["annotation_rows"] for s in summaries),
              "human_pairs": sum(s["human_pairs"] for s in summaries),
              "source_reference_issues": sum(s["source_reference_issues"] for s in summaries),
              "counterfactual_experiments": experiments, "download_samples": samples,
              "remote_files": len(files), "remote_logical_bytes": sum(x.size for x in files.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "verified_revision", "original_entries", "unique_original_paths",
                     "annotation_rows", "human_pairs", "source_reference_issues", "remote_files", "remote_logical_bytes"]}), flush=True)
    print(json.dumps({"experiments": len(experiments), "errors": errors[:20]}), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--spec", action="append", type=Path, default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-id", default="xju-arlab/vbench-repair")
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    raise SystemExit(run(p.parse_args()))
