#!/usr/bin/env python3
"""Build and publish a dimension-first, byte-preserving VBench 1.0 archive.

Requires huggingface_hub>=1.32 for cross-repository LFS copies. Raw annotations
are immutable. Unresolvable references are reported, never silently repaired.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.parse import quote

DIMENSIONS = {
    "Aesthetics.json": "aesthetic_quality",
    "Appearance_Style.json": "appearance_style",
    "Background_Consistency.json": "background_consistency",
    "Color.json": "color",
    "Dynamics_Degree.json": "dynamic_degree",
    "Human_Action.json": "human_action",
    "Motion_Smoothness.json": "motion_smoothness",
    "Multiplt_Object.json": "multiple_objects",
    "Object.json": "object_class",
    "Overall_Consistency.json": "overall_consistency",
    "Scene.json": "scene",
    "Spatial_Relationship.json": "spatial_relationship",
    "Subject_consistency.json": "subject_consistency",
    "Technical_Quality.json": "imaging_quality",
    "Temporal_Flickering.json": "temporal_flickering",
    "Temporal_Style.json": "temporal_style",
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe source path: {value!r}")
    return p.as_posix()


def origin_path(dimension, source):
    return f"dimensions/{dimension}/origin/videos/{safe_relative(source)}"


def normalize_pairs(rows, dimension, available):
    pairs, issues = [], []
    for index, row in enumerate(rows):
        videos = row["videos"]
        for model, path in videos.items():
            safe_relative(path)
            expected_model = "videocrafter" if model == "videocraft" else model
            if path not in available or path.split("/")[0] != expected_model:
                issues.append({"source_row": index, "generator_key": model,
                               "original_video_path": path,
                               "path_present": path in available,
                               "generator_matches_path": path.split("/")[0] == expected_model})
        for a, b in itertools.combinations(sorted(videos), 2):
            label = row.get("human_anno", {}).get(a, {}).get(b)
            reverse = row.get("human_anno", {}).get(b, {}).get(a)
            paths = [videos[a], videos[b]]
            resolved = [p in available and p.split("/")[0] ==
                        ("videocrafter" if m == "videocraft" else m)
                        for m, p in zip((a, b), paths)]
            pairs.append({"dimension": dimension, "source_row": index,
                          "prompt_en": row.get("prompt_en"), "generator_a": a,
                          "generator_b": b, "original_path_a": paths[0],
                          "original_path_b": paths[1],
                          "video_a": origin_path(dimension, paths[0]) if resolved[0] else None,
                          "video_b": origin_path(dimension, paths[1]) if resolved[1] else None,
                          "label_a_over_b": label, "label_b_over_a": reverse,
                          "reciprocal_labels": label is not None and reverse is not None
                          and abs(float(label) + float(reverse) - 1.0) < 1e-8,
                          "usable_exact_pair": all(resolved) and label is not None})
    return pairs, issues


def prepare(args):
    inventory = json.loads(args.inventory.read_text())
    available = {f["path"]: f for f in inventory["files"]}
    root = args.stage
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.annotations / "sources.json", provenance / "annotation-sources.json")
    shutil.copy2(args.annotations / "README.md", provenance / "upstream-annotation-README.md")
    write_json(provenance / "official-video-source.json", {
        "repo_id": inventory["repo_id"], "revision": inventory["revision"],
        "raw_annotations_unchanged": True,
        "background_note": "Raw Background_Consistency paths conflict with the released media. "
        "The scene video suite is included as a documented candidate suite, not as a repaired annotation mapping."})
    catalog, plan, all_issues = [], [], []
    for filename, dimension in DIMENSIONS.items():
        source = args.annotations / filename
        rows = json.loads(source.read_text())
        d = root / "dimensions" / dimension
        hp = d / "human_preference"
        hp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, hp / filename)
        pairs, issues = normalize_pairs(rows, dimension, available)
        write_jsonl(hp / "pairs.jsonl", pairs)
        write_jsonl(hp / "reference_issues.jsonl", issues)
        paths = {p for row in rows for p in row["videos"].values() if p in available}
        if dimension == "background_consistency":
            scene = json.loads((args.annotations / "Scene.json").read_text())
            paths.update(p for row in scene for p in row["videos"].values())
        manifest = []
        for p in sorted(paths):
            f = available[p]
            rec = {"dimension": dimension, "path": origin_path(dimension, p),
                   "source_path": p, "bytes": f["bytes"], "sha256": f["sha256"],
                   "git_blob_sha1": f["blob_id"] if f["sha256"] is None else None,
                   "source_repo": inventory["repo_id"], "source_revision": inventory["revision"],
                   "role": "candidate_scene_suite" if dimension == "background_consistency" else "exact_annotation_reference"}
            manifest.append(rec)
            plan.append(rec)
        write_jsonl(d / "origin" / "manifest.jsonl", manifest)
        summary = {"dimension": dimension, "annotation_file": filename,
                   "annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "annotation_rows": len(rows), "human_pairs": len(pairs),
                   "usable_exact_pairs": sum(p["usable_exact_pair"] for p in pairs),
                   "reference_issues": len(issues), "origin_files": len(manifest),
                   "origin_bytes": sum(x["bytes"] for x in manifest),
                   "status": "source_reference_issues" if issues else "ready_to_publish"}
        catalog.append(summary)
        all_issues.extend({"dimension": dimension, **i} for i in issues)
        (d / "README.md").write_text(
            f"# {dimension}\n\nOriginal annotation: `human_preference/{filename}` (unchanged).\n\n"
            f"{len(rows)} annotation rows; {len(pairs)} unordered comparisons; "
            f"{len(manifest)} original media files.\n\n"
            "Pair labels express preference for A over B: 1=A, 0=B, 0.5=tie. "
            "Original reverse labels are retained and checked separately. Paths in manifests "
            "are relative to the dataset root.\n\n"
            + ("**Upstream path/model inconsistencies remain unresolved.** The scene source suite "
               "is included, but it is not evidence that the broken human labels have been re-associated. "
               "Pairs with unresolved references have null video paths and usable_exact_pair=false.\n\n" if issues else "")
            + "Counterfactual experiments are stored under `counterfactual/<version>/`, "
            "with their original protocol and validation limitations.\n")
    write_json(root / "catalog.json", {"schema": "vbench-repair/1", "dimensions": catalog,
               "publication_status": "in_progress", "source_reference_issues": len(all_issues)})
    write_jsonl(provenance / "reference_issues.jsonl", all_issues)
    write_json(args.plan, {"schema": 1, "records": plan})
    print(json.dumps({"dimensions": len(catalog), "files": len(plan),
                      "unique_source_files": len({r['source_path'] for r in plan}),
                      "source_reference_issues": len(all_issues)}), flush=True)


def endpoint_for(args):
    import httpx
    endpoint = args.endpoint.rstrip("/")
    if args.allow_official_fallback and endpoint == "https://hf-mirror.com":
        try:
            response = httpx.get(endpoint + "/api/datasets/" + args.repo_id,
                                 follow_redirects=False, timeout=20)
            if response.status_code == 200:
                return endpoint
            print(json.dumps({"event": "mirror_fallback", "status": response.status_code,
                              "location": response.headers.get("location")}), flush=True)
        except httpx.HTTPError as exc:
            print(json.dumps({"event": "mirror_fallback", "error": type(exc).__name__}), flush=True)
        return "https://huggingface.co"
    return endpoint


def publish(args):
    import httpx
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, get_token
    endpoint = endpoint_for(args)
    api = HfApi(endpoint=endpoint)
    plan_bytes = args.plan.read_bytes()
    records = json.loads(plan_bytes)["records"]
    identity = hashlib.sha256(plan_bytes).hexdigest()
    state = json.loads(args.state.read_text()) if args.state.exists() else {"plan_sha256": identity, "commits": {}}
    if state["plan_sha256"] != identity:
        raise ValueError("Resume plan changed; use a fresh state file")
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    token = get_token()
    client = httpx.Client(headers={"Authorization": f"Bearer {token}"} if token else {},
                          follow_redirects=True, timeout=90, limits=httpx.Limits(max_connections=args.workers))

    def acquire(row):
        path = cache / safe_relative(row["source_path"])
        def valid(p):
            if not p.is_file() or p.stat().st_size != row["bytes"]:
                return False
            raw = p.read_bytes()
            return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == row["git_blob_sha1"]
        if valid(path):
            return path
        if args.local_videos and valid(args.local_videos / row["source_path"]):
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.local_videos / row["source_path"], path)
            return path
        url = endpoint + "/datasets/" + row["source_repo"] + "/resolve/" + row["source_revision"] + "/" + quote(row["source_path"], safe="/")
        for attempt in range(4):
            try:
                response = client.get(url)
                response.raise_for_status()
                raw = response.content
                digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                if len(raw) != row["bytes"] or digest != row["git_blob_sha1"]:
                    raise ValueError("Source blob identity mismatch: " + row["source_path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".partial")
                tmp.write_bytes(raw)
                tmp.replace(path)
                return path
            except (httpx.HTTPError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))

    for dimension in DIMENSIONS.values():
        subset = [r for r in records if r["dimension"] == dimension]
        for kind in ["lfs", "git"]:
            selected = [r for r in subset if (r["sha256"] is not None) == (kind == "lfs")]
            for offset in range(0, len(selected), args.batch_size):
                batch = selected[offset:offset + args.batch_size]
                key = f"{dimension}/{kind}/{offset}"
                if key in state["commits"]:
                    continue
                if kind == "lfs":
                    ops = [CommitOperationCopy(src_path_in_repo=r["source_path"], path_in_repo=r["path"],
                           src_revision=r["source_revision"], src_repo_id=r["source_repo"], src_repo_type="dataset") for r in batch]
                else:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        files = list(pool.map(acquire, batch))
                    ops = [CommitOperationAdd(path_in_repo=r["path"], path_or_fileobj=p) for r, p in zip(batch, files)]
                for attempt in range(4):
                    try:
                        result = api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops,
                            commit_message=f"data({dimension}): archive original videos {kind} {offset}-{offset + len(batch)}", num_threads=args.workers)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(5 * (attempt + 1))
                state["commits"][key] = {"sha": result.oid, "files": len(batch), "endpoint": endpoint}
                write_json(args.state, state)
                print(json.dumps({"event": "committed", "batch": key, "files": len(batch), "sha": result.oid}), flush=True)
        # Fill SHA-256 for the small Git-backed originals after exact blob verification.
        manifest = args.stage / "dimensions" / dimension / "origin" / "manifest.jsonl"
        complete_rows = []
        for row in subset:
            item = dict(row)
            if item["sha256"] is None:
                p = acquire(row)
                item["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            complete_rows.append(item)
        write_jsonl(manifest, complete_rows)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.stage / "dimensions" / dimension,
                          path_in_repo="dimensions/" + dimension,
                          commit_message=f"data({dimension}): add original annotations and integrity manifests")
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.stage / "provenance",
                      path_in_repo="provenance", commit_message="data: record official provenance and unresolved source references")
    print(json.dumps({"event": "origins_published", "commits": len(state["commits"])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "publish"])
    p.add_argument("--annotations", type=Path)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--state", type=Path, default=Path("output/origin-upload-state.json"))
    p.add_argument("--cache", type=Path, default=Path("output/origin-upload-cache"))
    p.add_argument("--local-videos", type=Path)
    p.add_argument("--repo-id", default="xju-arlab/vbench-repair")
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    p.add_argument("--allow-official-fallback", action="store_true")
    p.add_argument("--batch-size", type=int, default=300)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        p.error("batch-size and workers must be positive")
    if args.command == "prepare":
        if args.annotations is None or args.inventory is None:
            p.error("prepare requires --annotations and --inventory")
        prepare(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
