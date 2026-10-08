#!/usr/bin/env python3
"""Isolated legacy-vision worker; paper algorithms are imported without new formulas."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from paper_common import ROOT, TASKS, configure_imports, digest, load_assets, result, write_json


def evidence(rows, dim, assets):
    """Identical decode/model calls to the pinned semantic evidence collector."""
    import hashlib
    import torch
    import torchvision.transforms as tv
    from vbench.utils import load_video, tag2text_transform
    from cache_backend_outputs import initialize_umt
    device = torch.device("cuda:0")
    task = TASKS[dim]
    if task == "scene":
        from vbench.third_party.tag2Text.tag2text import tag2text_caption
        model = tag2text_caption(pretrained=assets["tag2text"], image_size=384, vit="swin_b").to(device).eval()
        transform = tag2text_transform(384)
    elif task == "action":
        model, transform, classes = initialize_umt(argparse.Namespace(umt_weights=assets["umt"]), device)
    else:
        from vbench_audit_models.grit import GritEvidenceModel
        # These two dimensions use the same ObjectDet predictor, native frames
        # and tuple conversion as object_class. Prompt-dependent scoring stays
        # in each dimension; Color retains its separate DenseCap namespace.
        model = GritEvidenceModel("object_class", Path(assets["grit"]),
                                  device=str(device), upstream=Path(assets["vbench"]))
    completed, cache = [], {}
    for row in rows:
        if row["video"] in cache:
            completed.append({**cache[row["video"]], "id": row["id"], "prompt": row["prompt"]})
            continue
        item = {"id": row["id"], "prompt": row["prompt"], "video": row["video"], "video_sha256": row["video_sha256"]}
        try:
            if task == "scene":
                frames = load_video(row["video"], num_frames=16, return_tensor=False, width=384, height=384)
                tensor = torch.cat([transform(frame).unsqueeze(0) for frame in frames]).to(device)
                with torch.no_grad():
                    captions, _ = model.generate(tensor, tag_input=None, return_tag_predict=True)
                item["frame_captions"] = list(captions)
            elif task == "action":
                inputs = load_video(row["video"], transform, num_frames=16).unsqueeze(0)
                with torch.no_grad():
                    scores, ids = torch.topk(torch.sigmoid(model(inputs.to(device))), 5, dim=1)
                item["top5"] = [{"class_id": int(i), "label": classes[str(i)], "score": float(s), "rounded_score": round(float(s), 4)}
                                for i, s in zip(ids.squeeze().tolist(), scores.squeeze().tolist())]
                item["model_input_sha256"] = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
            else:
                video = load_video(row["video"], num_frames=16)
                _, _, h, w = video.size()
                item["original_frame_size"] = [int(w), int(h)]
                if min(h, w) > 768:
                    scale = 720. / min(h, w)
                    video = tv.Resize(size=(int(scale * h), int(scale * w)))(video)
                frames = video.permute(0, 2, 3, 1).numpy()
                item["frame_size"] = [int(frames.shape[2]), int(frames.shape[1])]
                detections, labels = [], []
                with torch.no_grad():
                    for frame in frames:
                        captions = model.caption(frame)
                        detections.append([{"label": x[0], "box": [float(v) for v in x[1][:4]]} for x in captions])
                        labels.append(sorted(set(str(v) for v in captions[0][2])) if captions else [])
                item.update(frame_detections=detections, frame_labels=labels)
            if task != "action":
                item["frame_sha256"] = [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
            item["status"] = "ok"
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        completed.append(item)
        cache[row["video"]] = item
    payload = {"rows": completed, "source": "pinned cache_backend_outputs.py visual protocol", "fresh_visual_inference": True}
    if task in {"spatial", "objects"} and model.cache is not None:
        payload["run_shared_inference"] = {"hits": model.cache.hits, "misses": model.cache.misses,
                                          "scope": "current evaluation only", "cache": str(model.cache.root)}
        payload["fresh_visual_inference"] = model.cache.misses > 0
    return payload


def official_submodules(dim):
    from vbench.utils import init_submodules
    # Missing assets must be prepared explicitly through the mirror-aware setup.
    # Prevent upstream's hard-coded wget/git calls from silently downloading.
    original_run = subprocess.run
    def offline_run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and Path(str(command[0])).name in {"wget", "curl", "git"}:
            raise FileNotFoundError("Missing offline VBench dependency; run restore_paper_runtime.py first")
        return original_run(command, *args, **kwargs)
    subprocess.run = offline_run
    try:
        modules = init_submodules([dim], local=True, read_frame=False)
    finally:
        subprocess.run = original_run
    return modules[dim]


def official(rows, dim, assets, output):
    import torch
    submodules = official_submodules(dim)
    full_info = [{"prompt_en": row["prompt"], "dimension": [dim], "video_list": [row["video"]],
                  "auxiliary_info": row["auxiliary_info"]} for row in rows]
    path = output.with_name("official-input.json")
    write_json(path, full_info)
    if dim in {"aesthetic_quality", "imaging_quality", "temporal_flickering", "appearance_style"}:
        adapter = importlib.import_module(dim + ".official")
        aggregate, returned = adapter.compute(str(path), torch.device("cuda:0"), submodules)
    else:
        module = importlib.import_module("vbench." + dim)
        aggregate, returned = getattr(module, "compute_" + dim)(str(path), torch.device("cuda:0"), submodules)
    return {"rows": score_rows(rows, returned, "official"), "official_aggregate": float(aggregate),
            "upstream_commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"}


def score_rows(rows, returned, implementation):
    by_video = defaultdict(list)
    for item in returned:
        by_video[str(Path(item["video_path"]).resolve())].append(item)
    results = []
    for row in rows:
        candidates = by_video[row["video"]]
        if not candidates:
            results.append(result(row, None, "dropped_by_" + implementation,
                                  reason="upstream omitted input" if implementation == "official" else "implementation omitted input"))
        else:
            item = candidates.pop(0)
            results.append(result(row, float(item["video_results"]), **{implementation: item}))
    return results


def accelerated(rows, dim, assets):
    import torch
    # Dimensions are enabled individually after their implementation/validation.
    if dim not in {"appearance_style", "imaging_quality", "aesthetic_quality", "temporal_style"}:
        raise ValueError(f"No accelerated implementation for {dim}")
    adapter = importlib.import_module(dim + ".accelerated")
    diagnostics = {}
    extra = {"diagnostics": diagnostics} if dim == "temporal_style" else {}
    aggregate, returned = adapter.compute(rows, torch.device("cuda:0"), official_submodules(dim), **extra)
    return {"rows": score_rows(rows, returned, "accelerated"), "aggregate": float(aggregate),
            "implementation": "project accelerated", "aggregation": "vbench_native_reducer",
            "reference_upstream_commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", **diagnostics}


def repair(rows, dim, assets, output, compiled):
    import torch
    device = torch.device("cuda:0")
    if dim == "dynamic_degree":
        import io
        import math
        import hashlib
        import numpy as np
        from dynamic_degree.learned_probe import MotionProbe
        from vbench_audit_models.vjepa import FrozenVJEPA
        from scripts.counterfactual.official_video_jitter import native_video
        root = Path(assets["dynamic_model"])
        model = MotionProbe().eval().requires_grad_(False)
        model.load_state_dict(torch.load(root / "aligned.pt", map_location="cpu", weights_only=True), strict=True)
        model = model.to(device)
        config = json.loads((root / "configs/vjepa-probe-v1.json").read_text())
        encoder = FrozenVJEPA(assets["vjepa_source"], root / "backbone/vjepa2_1_vitb_dist_vitG_384.pt", config)
        results = []
        for row in rows:
            try:
                frames, pts, fps = native_video(Path(row["video"]), 1e-6)
                if len(frames) != 16 or abs(fps - 8) > 1e-6:
                    raise ValueError("Paper Dynamic requires native 16 frames at 8 FPS; input not silently resampled")
                features = encoder.encode(frames)
                with torch.inference_mode():
                    latent = float(model(torch.from_numpy(features).to(device, torch.float32)[None])[0])
                stream = io.BytesIO(); np.save(stream, features, allow_pickle=False)
                results.append(result(row, 1 / (1 + math.exp(-latent)), latent=latent,
                                      feature_sha256=hashlib.sha256(stream.getvalue()).hexdigest(), method="aligned-v1"))
            except Exception as exc:
                results.append(result(row, None, "failed", error=str(exc)))
        return {"rows": results, "encoder": encoder.identity}
    if dim == "subject_consistency":
        from subject_consistency.hybrid import load_hybrid_provider
        from subject_consistency.metric import build_dino_config, evaluate_masked_batch, set_seed
        from subject_consistency.models import OfficialDinoPatchExtractor
        set_seed(42)
        provider = load_hybrid_provider(Path(assets["maskrcnn"]), Path(assets["mobilesam"]), device)
        dino = build_dino_config(assets["dino_repo"], assets["dino"])
        extractor = OfficialDinoPatchExtractor(device, dino, Path(assets["vbench"]), frame_batch_size=16)
        results = []
        for row in rows:
            video = Path(row["video"])
            item = evaluate_masked_batch([video], {video.name: row}, device, dino, provider, extractor=extractor,
                                        phrase_field="subject_en", instance_mode="union", missing_policy="exclude",
                                        encoding_mode="preencode_crop")[0]
            results.append(result(row, item["score"], item["status"], diagnostics=item.get("diagnostics"), error=item.get("error")))
        return {"rows": results, "method": "hybrid_v5_exclude"}
    if dim == "background_consistency":
        from background_consistency.metric import evaluate_batch
        config = {"model": {"clip": {"checkpoint": assets["clip"], "sha256": digest(assets["clip"])},
                  "localizer": {"detector_checkpoint": assets["maskrcnn"], "detector_sha256": digest(assets["maskrcnn"]),
                                "sam_checkpoint": assets["mobilesam"], "sam_sha256": digest(assets["mobilesam"]), "threshold": .8, "size": 512}},
                  "runtime": {"audit_variant": "patch_frame_calibrated", "seed": 0}}
    else:
        module = importlib.import_module(dim + ".metric")
        evaluate_batch = module.evaluate_batch
        config = {"model": {"grit": {"checkpoint": assets["grit"]},
                  "labels": {"path": str(ROOT / "configs/four_dimension/object_color_vocabulary.json")},
                  "prompt_compiler": {"kind": "lora", "path": str(compiled)}, "runtime": {"upstream_root": assets["vbench"]}},
                  "runtime": {"audit_variant": "repair", "evidence_dir": str(output.parent / "instance-evidence")}}
    # A metadata key is a filename in the existing backend. Group duplicate names
    # into distinct calls, preserving every query rather than silently overwriting.
    buckets = []
    for row in rows:
        for batch in buckets:
            if all(Path(old["video"]).name != Path(row["video"]).name for old in batch):
                batch.append(row); break
        else:
            buckets.append([row])
    results = []
    for batch in buckets:
        paths = [Path(r["video"]) for r in batch]
        returned = evaluate_batch("audit", paths, {p.name: r for p, r in zip(paths, batch)}, str(device), config)
        if len(returned) != len(batch):
            raise ValueError("Backend changed input denominator")
        for row, item in zip(batch, returned):
            results.append(result(row, item.score, item.status, diagnostics=item.metric, error=item.error))
    return {"rows": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--mode", choices=["origin", "repair", "evidence", "accelerated"], required=True)
    parser.add_argument("--compiled", type=Path)
    args = parser.parse_args()
    assets = load_assets(args.assets)
    configure_imports(assets)
    rows = json.loads(args.input.read_text())
    for row in rows:
        if digest(row["video"]) != row["video_sha256"]:
            raise ValueError("Video changed after input validation")
    from vbench_audit_core.upstream import verify_upstream
    verify_upstream(assets["vbench"])
    sys.path.insert(0, assets["vbench"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import cv2
    torch.set_num_threads(3); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("An explicit CUDA runtime is required")
    if args.mode == "evidence":
        payload = evidence(rows, args.dimension, assets)
    elif args.mode == "origin":
        payload = official(rows, args.dimension, assets, args.output)
    elif args.mode == "accelerated":
        payload = accelerated(rows, args.dimension, assets)
    else:
        payload = repair(rows, args.dimension, assets, args.output, args.compiled)
    payload.update(dimension=args.dimension, mode=args.mode, torch=torch.__version__,
                   worker_sha256=digest(__file__), input_manifest_sha256=digest(args.input))
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
