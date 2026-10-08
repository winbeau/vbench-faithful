#!/usr/bin/env python3
"""Complete frozen Object/Color endpoint scoring with cached prompt-only LoRA.

No training, cohort selection by scores, or change to a historical output. The
cached text predictions are reused verbatim; GRiT is run freshly for both arms.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PINS = {
    "object_class": {
        "metadata": "7ef574dc3647355e682f91be83fd45e82496e3b876cfef3d3ba76d8ef3c8c936",
        "compiled": "1476b75427e0ba80f8e2430514599a4d20194ac99af86b8e18b488d152609646",
        "pairs": 14,
    },
    "color": {
        "metadata": "3e9ac091d96a491144b6d96a2835fd3c058201db8294b2e3afa031fe8e15c86d",
        "compiled": "adc01664745bc1279f23a326f9fce545d3d921aae3c030f4e9b530b4e308255e",
        "pairs": 5,
    },
}
GRIT_SHA = "53b6e9b3fd948eac55b574c9b6f94ad0743dff46ba449df7ac2d33009ee92ef1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def select_endpoints(rows, dimension, expected_pairs):
    """Select the previously frozen primary test endpoints, without scores."""
    grouped = defaultdict(dict)
    for row in rows:
        if row.get("split") != "test":
            continue
        if dimension == "object_class":
            arm = {"canonical": "base", "uppercase": "cf"}.get(row.get("variant"))
        else:
            if row.get("family") != "color_visibility_denominator":
                continue
            arm = {1.0: "base", 0.0: "cf"}.get(row.get("visible_fraction"))
        if arm is None:
            continue
        group = grouped[row["base_id"]]
        if arm in group:
            raise ValueError("duplicate endpoint")
        group[arm] = row
    if len(grouped) != expected_pairs:
        raise ValueError("frozen cohort size changed")
    selected, pairs = [], []
    for base_id, group in sorted(grouped.items()):
        if set(group) != {"base", "cf"}:
            raise ValueError("missing endpoint; do not silently drop pairs")
        base, cf = group["base"], group["cf"]
        if base["prompt"] != cf["prompt"]:
            raise ValueError("this completion requires unchanged original prompts")
        if dimension == "object_class":
            a = base["dimension_metadata"][dimension]["object"]
            b = cf["dimension_metadata"][dimension]["object"]
            if b != a.upper() or base["source_video_sha256"] != cf["source_video_sha256"]:
                raise ValueError("not the frozen metadata uppercase intervention")
        elif base["dimension_metadata"] != cf["dimension_metadata"]:
            raise ValueError("color visibility must not change the text target")
        selected.extend((base, cf))
        pairs.append({"base_id": base_id, "base_query_uid": base["query_uid"],
                      "cf_query_uid": cf["query_uid"]})
    if len({r["query_uid"] for r in selected}) != len(selected):
        raise ValueError("duplicate query IDs")
    if len({r["video"] for r in selected}) != len(selected):
        raise ValueError("duplicate filenames would overwrite metadata lookup")
    return selected, pairs


def prepare(source, output, dimension):
    family = source / "families" / dimension
    meta = family / "metadata.json"
    compiled = source / "compiled" / f"{dimension}-lora.json"
    pins = PINS[dimension]
    if digest(meta) != pins["metadata"] or digest(compiled) != pins["compiled"]:
        raise ValueError("frozen metadata / compiled LoRA SHA mismatch")
    rows, pairs = select_endpoints(json.loads(meta.read_text())["videos"], dimension, pins["pairs"])
    predictions = json.loads(compiled.read_text())
    if predictions["input_fields"] != ["prompt"]:
        raise ValueError("LoRA input contract is not prompt-only")
    available = {r["prompt"] for r in predictions["records"]}
    if any(row["prompt"] not in available for row in rows):
        raise ValueError("missing frozen text prediction")
    target = output / "families" / dimension
    (target / "videos").mkdir(parents=True, exist_ok=False)
    media = []
    for row in rows:
        path = (family / "videos" / row["video"]).resolve(strict=True)
        expected = row["source_video_sha256"] if dimension == "object_class" else row["video_sha256"]
        if digest(path) != expected:
            raise ValueError(f"media SHA mismatch: {row['query_uid']}")
        (target / "videos" / row["video"]).symlink_to(path)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,nb_frames",
            "-of", "json", str(path)], text=True))
        media.append({"query_uid": row["query_uid"], "path": str(path), "sha256": expected, "probe": probe})
    save(target / "metadata.json", {"videos": rows})
    save(target / "freeze.json", {
        "metadata_sha256": digest(target / "metadata.json"), "parent_metadata_sha256": pins["metadata"],
        "parent_freeze_sha256": digest(family / "freeze.json"), "pairs": pairs,
        "selection": "all frozen primary test endpoints; no score-based filtering",
        "n_queries": len(rows), "compiled_sha256": pins["compiled"],
        "semantic_execution": "reused frozen prompt-only selected-step300 LoRA predictions",
        "visual_execution": "pending fresh GRiT inference", "frozen_before_scores": True,
    })
    save(target / "media.json", media)
    (output / "compiled").mkdir(exist_ok=True)
    shutil.copyfile(compiled, output / "compiled" / compiled.name)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dimension", choices=list(PINS), required=True)
    args = parser.parse_args()
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe must be on PATH before preparing inputs")
    if digest(args.checkpoint) != GRIT_SHA:
        raise ValueError("unexpected GRiT weights")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("isolate one physical GPU before launching")
    prepare(args.source, args.output, args.dimension)
    for backend, compiler in (("official", "metadata"), ("repair", "lora")):
        command = [sys.executable, "-u", "-B", "-m", "scripts.object_color_experiments", "score",
                   "--dimension", args.dimension, "--output", str(args.output),
                   "--upstream", str(args.upstream), "--checkpoint", str(args.checkpoint),
                   "--variant", backend, "--compiler", compiler]
        start = time.time()
        result = subprocess.run(command, check=False)
        save(args.output / "execution" / f"{args.dimension}-{backend}.json", {
            "command": command, "returncode": result.returncode, "started_unix": start,
            "wall_seconds": time.time() - start, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "runner_sha256": digest(__file__), "semantic_execution": "cached LoRA, not new Qwen inference",
            "visual_execution": "fresh GPU inference", "grit_sha256": GRIT_SHA,
        })
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
