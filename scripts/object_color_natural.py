"""Run frozen Object/Color backends on the VBench paper's unedited videos."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from vbench_audit_core.contracts import run_batch_contract, run_batch_sharded
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.outputs import write_results
from vbench_audit_core.provenance import collect_provenance
from vbench_audit_core.schemas import VideoResult
from scripts.object_color_natural_media import save


def cohort(root, dimension):
    rows = json.loads((root / (dimension + "-manifest.json")).read_text())["rows"]
    if len({r["video_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate frozen video_uid")
    return rows


def cached_evaluator(backend, videos, metadata, device, config):
    """Common batch evaluator with UID checkpoints and no metric formulas."""
    import random
    import numpy as np
    import torch
    random.seed(20260920)
    np.random.seed(20260920)
    torch.manual_seed(20260920)
    dimension = config["experiment"]["dimension"]
    module = importlib.import_module(dimension + ".metric")
    checkpoint_root = Path(config["experiment"]["checkpoints"])
    context = config["experiment"]["context_sha256"]
    found, pending = {}, []
    for video in videos:
        uid = metadata[video.name]["video_uid"]
        path = checkpoint_root / (uid + ".json")
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved["context_sha256"] != context or saved["result"]["metric"]["video_uid"] != uid:
                raise ValueError("checkpoint context or UID mismatch")
            raw = saved["result"]
            raw["video"] = str(video)
            found[str(video)] = VideoResult(**raw)
        else:
            pending.append(video)
    chunk_size = config["experiment"]["chunk_size"]
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        rows = run_batch_contract(module.evaluate_batch, backend, chunk, metadata,
                                  device=device, config=config)
        for row in rows:
            uid = metadata[Path(row.video).name]["video_uid"]
            save(checkpoint_root / (uid + ".json"), {"context_sha256": context,
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": device, "result": asdict(row)})
            found[row.video] = row
        print(json.dumps({"dimension": dimension, "backend": backend,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "completed_in_worker": len(found), "worker_inputs": len(videos),
            "last_chunk_status": dict(Counter(r.status for r in rows))}), flush=True)
    return [found[str(video)] for video in videos]


def score(args):
    root = args.root.resolve()
    rows = cohort(root, args.dimension)
    ledger_path = root / "media-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger["errors"] or ledger["materialized_videos"] != ledger["expected_videos"]:
        raise ValueError("all official media must be available before scoring")
    assets = {r["relative_path"]: r for r in ledger["files"]}
    inputs = root / "inputs" / args.dimension
    inputs.mkdir(parents=True, exist_ok=True)
    metadata, videos = {}, []
    for row in rows:
        raw = root / "media" / row["relative_video_path"]
        if sha256_file(raw) != assets[row["relative_video_path"]]["sha256"]:
            raise ValueError("official media checksum mismatch")
        video = inputs / (row["video_uid"] + raw.suffix)
        if video.is_symlink():
            if video.resolve() != raw.resolve():
                raise ValueError("existing input link points to different media")
        elif video.exists():
            raise ValueError("input view must be a link preserving native media")
        else:
            video.symlink_to(raw)
        videos.append(video)
        metadata[video.name] = {"video": video.name, "video_uid": row["video_uid"],
            "prompt": row["prompt"], "dimension_metadata": {
                args.dimension: row["official_auxiliary_info"]}}
    save(root / "inputs" / (args.dimension + "-metadata.json"), {"videos": list(metadata.values())})
    module = importlib.import_module(args.dimension + ".metric")
    backend = "vbench" if args.backend == "official" else "audit"
    destination = root / "scores" / args.dimension / args.backend
    if (destination / "run.json").exists():
        raise FileExistsError("completed result exists; use its verified result or a new run root")
    model = {"grit": {"checkpoint": str(args.checkpoint)},
             "labels": {"path": str(args.vocabulary.resolve())},
             "runtime": {"upstream_root": str(args.upstream)},
             "prompt_compiler": {"kind": "deterministic"}}
    context = {"dimension": args.dimension, "backend": backend, "model": model,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "vocabulary_sha256": sha256_file(args.vocabulary),
        "manifest_sha256": sha256_file(root / (args.dimension + "-manifest.json")),
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "media_ledger_sha256": sha256_file(ledger_path),
        "source_files": {str(p): sha256_file(p) for folder in ("packages", "metrics")
                         for p in sorted(Path(folder).glob("*/src/**/*.py"))},
        "runner_sha256": sha256_file(Path(__file__))}
    checkpoint_root = root / "checkpoints" / args.dimension / args.backend
    context_path = checkpoint_root / "context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("existing checkpoint context changed")
    save(context_path, context)
    config = {"model": model,
              "runtime": {"audit_variant": "repair", "seed": 20260920,
                          "evidence_dir": str(root / "evidence" / args.dimension / args.backend)},
              "experiment": {"dimension": args.dimension, "chunk_size": args.chunk_size,
                    "checkpoints": str(checkpoint_root / "records"),
                    "context_sha256": sha256_file(context_path)}}
    visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
    if not visible or len(set(visible)) != len(visible):
        raise ValueError("set explicit unique available physical CUDA_VISIBLE_DEVICES")
    started = time.time()
    results = run_batch_sharded(cached_evaluator, backend, videos, metadata,
                                list(range(len(visible))), config=config,
                                label="official-natural-" + args.dimension)
    if {r.metric.get("video_uid") for r in results} != {r["video_uid"] for r in rows}:
        raise ValueError("recovered batch lacks exact UID coverage")
    summary = module.summarize(backend, results)
    wall = time.time() - started
    write_results(destination, results, summary, {
        "dimension": args.dimension, "backend": args.backend, "config": config,
        "started_unix": started, "finished_unix": time.time(), "wall_seconds": wall,
        "physical_gpus": visible, "scheduling": last_schedule(),
        "python": platform.python_version(), "status": summary.status,
        "provenance": collect_provenance(videos,
            root / "inputs" / (args.dimension + "-metadata.json"), None,
            upstream_status="verified_by_metric_backend"),
        "context_sha256": sha256_file(context_path),
        "coverage": {"expected": len(rows), "retained": len(results),
                     "status_counts": dict(Counter(r.status for r in results))}})
    print(json.dumps({"dimension": args.dimension, "backend": args.backend,
                      "wall_seconds": wall, "summary": summary.to_dict()}), flush=True)
    if any(r.status == "failed" for r in results):
        raise RuntimeError("runtime failures retained; inspect them before the next dimension")


def compile_prompts(args):
    from vbench_audit_models.labels import LabelVocabulary
    from vbench_audit_models.qwen import QwenPromptRouter
    vocabulary = LabelVocabulary.from_file(args.vocabulary)
    dimensions = ("object_class", "color")
    router = QwenPromptRouter.from_local(args.base, {d: args.adapters / d for d in dimensions})
    prompts = json.loads((args.root / "official-prompts.json").read_text())
    started = time.time()
    for dimension in dimensions:
        for mode in ("base", "lora"):
            destination = args.root / "compiled" / (dimension + "-" + mode + ".json")
            if destination.exists():
                raise FileExistsError(destination)
            previous_path = args.existing_compiled / (dimension + "-" + mode + ".json")
            previous = json.loads(previous_path.read_text())
            if previous["model"] != router.provenance:
                raise ValueError("cached prompt model identity differs from loaded base/adapters")
            known = {r["prompt"]: r for r in previous["records"]}
            compiled = []
            for i, prompt in enumerate(prompts[dimension], 1):
                item = known.get(prompt)
                if item is None:
                    item = router.compile(dimension, prompt, vocabulary, mode=mode)
                compiled.append(item)
                if i % 10 == 0:
                    print(json.dumps({"dimension": dimension, "mode": mode,
                                      "completed": i, "total": len(prompts[dimension])}), flush=True)
            save(destination, {"dimension": dimension, "mode": mode, "input_fields": ["prompt"],
                "model": router.provenance, "vocabulary_sha256": vocabulary.provenance["sha256"],
                "records": compiled, "cache_source_sha256": sha256_file(previous_path),
                "cached_prompts": len(set(prompts[dimension]) & known.keys()),
                "trained_in_this_run": False})
    save(args.root / "compiled" / "timing.json", {"started_unix": started,
        "wall_seconds_excluding_model_load": time.time() - started,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "new_training": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "compile"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dimension", choices=("object_class", "color"))
    parser.add_argument("--backend", choices=("official", "repair_deterministic"))
    parser.add_argument("--upstream", type=Path, default=Path("/root/wenbiao_zhao/VBench"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth"))
    parser.add_argument("--vocabulary", type=Path, default=Path("configs/four_dimension/object_color_vocabulary.json"))
    parser.add_argument("--chunk-size", type=int, default=192)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--existing-compiled", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if "output" not in args.root.resolve().parts:
        raise ValueError("experiment output must be under output/")
    {"score": score, "compile": compile_prompts}[args.command](args)


if __name__ == "__main__":
    main()
