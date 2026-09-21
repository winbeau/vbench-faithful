"""Read-only media/runtime inventory for the already frozen Object/Color runs.

Run on the H100 host that holds the videos. No models are loaded and no
construction or scoring is repeated. Only the requested output directory is
written; all media paths and model assets are read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.upstream import verify_upstream


def inventory(root: Path, upstream: Path) -> dict:
    state = verify_upstream(upstream)
    media, cache, construction = [], {}, []
    for dimension in ("object_class", "color"):
        family = root / "families" / dimension
        frozen = json.loads((family / "freeze.json").read_text())
        if sha256_file(family / "metadata.json") != frozen["metadata_sha256"]:
            raise ValueError("frozen input metadata changed")
        for row in json.loads((family / "metadata.json").read_text())["videos"]:
            path = family / "videos" / row["video"]
            resolved = path.resolve(strict=True)
            if resolved not in cache:
                probe = json.loads(subprocess.check_output([
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration",
                    "-of", "json", str(resolved)], text=True))
                stream = probe["streams"][0]
                cache[resolved] = {"sha256": sha256_file(resolved), "stream": stream,
                    "media_seconds": float(probe["format"]["duration"])}
            actual = cache[resolved]
            expected = row.get("video_sha256", row["source_video_sha256"])
            if actual["sha256"] != expected:
                raise ValueError(f"frozen media changed: {row['query_uid']}")
            if dimension == "color" and row["family"] == "color_visibility_denominator":
                replay = resolved.with_name(resolved.stem + "-replay" + resolved.suffix)
                replay_hash = sha256_file(replay)
                if replay_hash != actual["sha256"] or row["construction_proof"]["outside_mask_changed_pixels"] != 0:
                    raise ValueError("construction replay or pixel assertion changed")
                construction.append({"query_uid": row["query_uid"], "path": str(resolved),
                    "replay_path": str(replay), "sha256": actual["sha256"], "replay_sha256": replay_hash,
                    "outside_mask_changed_pixels": 0})
            media.append({"dimension": dimension, "base_id": row["base_id"],
                "query_uid": row["query_uid"], "video_uid": row["video_uid"],
                "split": row["split"], "family": row["family"],
                "path": str(path), "resolved_path": str(resolved), **actual})
    frozen_color = json.loads((root / "families/color/freeze.json").read_text())
    if frozen_color["scorer_received_masks"]:
        raise ValueError("construction masks leaked to scorer")
    for row in frozen_color["accepted"]:
        path = root / "construction/color" / (row["base_id"] + "-masks.npy")
        if sha256_file(path) != row["mask_sha256"]:
            raise ValueError("construction mask changed")
        if any(row["zero_endpoint_locator"]["target_present"]):
            raise ValueError("unqualified zero endpoint included")
    versions = {}
    for name in ("torch", "torchvision", "transformers", "detectron2", "numpy", "decord"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"media": media, "unique_media_files": len(cache), "construction_replays": construction,
        "construction_mask_hashes_verified": len(frozen_color["accepted"]),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "versions": versions, "upstream": asdict(state),
        "containing_checkout_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_file_sha256": {str(p): sha256_file(p) for group in ("packages", "metrics")
                               for p in sorted(Path(group).glob("*/src/**/*.py"))},
        "gpu_inventory_csv": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"], text=True),
        "scope": "post-run read-only inventory; recorded score run.json supplies actual devices, source hashes and times"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = inventory(args.root, args.upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"queries": len(result["media"]), "unique_media_files": result["unique_media_files"]}))


if __name__ == "__main__":
    main()
