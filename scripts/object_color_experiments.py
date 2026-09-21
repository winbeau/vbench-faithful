#!/usr/bin/env python3
"""Frozen family preparation and execution through the public metric contract."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.artifacts import write_json_artifact
from vbench_audit_core.contracts import run_batch_contract
from vbench_audit_core.outputs import write_results
from scripts.object_color import save


def locate(args):
    import torch
    from vbench_audit_core.upstream import import_official_module
    from vbench_audit_models.coco import CocoInstanceLocator
    from vbench_audit_models.labels import LabelVocabulary
    torch.set_num_threads(2)
    module, state = import_official_module(args.dimension, args.upstream)
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    protocol = json.loads((args.config / "object_color_protocol.json").read_text())
    model = CocoInstanceLocator(args.locator, device="cuda:0", expected_sha256=args.locator_sha256,
                                threshold=protocol["sampling"]["locator_score_min"])
    sources = json.loads((args.output / "source_manifest.json").read_text())["rows"]
    selected = [r for r in sources if r["dimension"] == args.dimension and r["split"] != "train"]
    destination = args.output / "localization" / args.dimension
    rows = []
    for source in selected:
        path = destination / (source["source_group"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["source"]["sha256"] != source.get("sha256") or row["locator"] != model.provenance:
                raise ValueError("localization resume input/model mismatch")
        else:
            row = {"source": source, "locator": model.provenance, "upstream": asdict(state)}
            if source["status"] != "available":
                row.update({"status": "rejected", "reason": source["status"]})
            else:
                frames = module.load_video(source["video"], num_frames=16, return_tensor=False)
                detections = model.instances_for(frames)
                target = source["gold"]["object"]
                matches = [[r for r in records if vocabulary.object(r["label"]) == vocabulary.object(target)] for records in detections]
                fraction = sum(bool(r) for r in matches) / len(matches)
                row.update({"detections": detections, "target_boxes": matches, "present_fraction": fraction,
                            "status": "accepted" if fraction >= protocol["sampling"]["target_presence_min_fraction"] else "rejected",
                            "reason": None if fraction >= protocol["sampling"]["target_presence_min_fraction"] else "independent_locator_presence_gate",
                            "ground_truth_status": "automatic_confirmation_only; absence not human-confirmed"})
                observed = {vocabulary.object(r["label"]) for records in detections for r in records}
                candidates = sorted(vocabulary.objects, key=lambda c: hashlib.sha256((source["source_group"]+c).encode()).hexdigest())
                row["absent_control"] = next((c for c in candidates if c not in observed and c != vocabulary.object(target)), None)
            save(path, row)
        rows.append(row)
        print(json.dumps({"source_group": source["source_group"], "status": row["status"], "present_fraction": row.get("present_fraction")}), flush=True)
    save(destination / "manifest.json", {"rows": rows, "status_counts": dict(Counter(r["status"] for r in rows)),
         "protocol_sha256": sha256_file(args.config / "object_color_protocol.json"), "no_grit_score_selection": True})


def prepare_object(args):
    rows = json.loads((args.output / "localization/object_class/manifest.json").read_text())["rows"]
    aliases = json.loads((args.config / "object_color_aliases.json").read_text())["object_aliases"]
    root = args.output / "families/object_class"
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=False)
    queries, rejections = [], []
    for row in rows:
        source = row["source"]
        if row["status"] != "accepted":
            rejections.append({"base_id": source["source_group"], "reason": row["reason"], "split": source["split"]})
            continue
        target = source["gold"]["object"]
        variations = [("canonical", target), ("uppercase", target.upper())]
        if aliases[target]:
            variations.append(("synonym", aliases[target][0]))
        if row["absent_control"]:
            variations.append(("absent", row["absent_control"]))
        for variant, label in variations:
            video = videos / f"video_{len(queries):05d}.mp4"
            video.symlink_to(source["video"])
            queries.append({"video": video.name, "video_uid": source["sha256"] + ":" + variant,
                "base_id": source["source_group"], "query_uid": source["source_group"] + ":" + variant,
                "family": "object_absent_control" if variant == "absent" else "object_label_invariance",
                "variant": variant, "split": source["split"], "prompt": source["prompt"],
                "dimension_metadata": {"object_class": {"object": label}},
                "source_video_sha256": source["sha256"], "source_group": source["source_group"]})
    save(root / "metadata.json", {"videos": queries})
    save(root / "freeze.json", {"metadata_sha256": sha256_file(root / "metadata.json"),
        "localization_sha256": sha256_file(args.output / "localization/object_class/manifest.json"),
        "aliases_sha256": sha256_file(args.config / "object_color_aliases.json"),
        "protocol_sha256": sha256_file(args.config / "object_color_protocol.json"),
        "rejections": rejections, "n_input_bases": len(rows), "n_queries": len(queries), "frozen_before_scores": True})


def prepare_color(args):
    import numpy as np
    import torch
    from vbench_audit_core.upstream import import_official_module
    from vbench_audit_models.coco import CocoInstanceLocator
    from vbench_audit_models.labels import LabelVocabulary
    from scripts.object_color_construction import ConstructionMasks, occlude, encode_lossless
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    construction = args.config / "object_color_construction.json"
    declaration = json.loads(construction.read_text())
    if declaration["scorer_mask_input"] is not False or declaration["levels"] != [1., .75, .5, .25, 0.]:
        raise ValueError("unexpected frozen construction protocol")
    module, state = import_official_module("color", args.upstream)
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    aliases = json.loads((args.config / "object_color_aliases.json").read_text())["object_aliases"]
    locator = CocoInstanceLocator(args.locator, device="cuda:0", expected_sha256=args.locator_sha256, threshold=.7)
    maker = ConstructionMasks(args.segformer, args.sam_checkpoint, args.sam_source, device="cuda:0")
    inputs = json.loads((args.output / "localization/color/manifest.json").read_text())["rows"]
    root = args.output / "families/color"
    root.mkdir(parents=True, exist_ok=False)
    construction_root = args.output / "construction/color"
    construction_root.mkdir(parents=True, exist_ok=False)
    save(construction_root / "protocol.json", {"declaration": declaration,
        "protocol_sha256": sha256_file(construction), "models": maker.provenance, "locator": locator.provenance})
    queries, rejected, accepted = [], [], []
    for base_index, row in enumerate(inputs):
        source = row["source"]
        base = source["source_group"]
        proof = {"base_id": base, "split": source["split"], "source": source,
                 "localization_status": row["status"], "status": "rejected"}
        try:
            if row["status"] != "accepted" or row["present_fraction"] != 1:
                raise ValueError("independent source presence not 16/16; cannot name endpoint 100 percent")
            frames = module.load_video(source["video"], num_frames=16, return_tensor=False)
            masks, mask_evidence = maker.for_frames(frames, source["gold"]["object"], row["target_boxes"])
            mask_path = construction_root / (base + "-masks.npy")
            np.save(mask_path, masks, allow_pickle=False)
            proof.update({"mask_sha256": sha256_file(mask_path), "mask_evidence": mask_evidence,
                          "seed": int(base, 16) % 2**32})
            zero, _ = occlude(frames, masks, 0, seed=proof["seed"])
            zero_detections = locator.instances_for(zero)
            target = vocabulary.object(source["gold"]["object"])
            residual = [any(vocabulary.object(x["label"]) == target for x in objects) for objects in zero_detections]
            proof["zero_endpoint_locator"] = {"detections": zero_detections, "target_present": residual}
            if any(residual):
                raise ValueError("independent locator still detects target after full occlusion")
            base_queries = []
            for level_index, level in enumerate(declaration["levels"]):
                edited, evidence = occlude(frames, masks, level, seed=proof["seed"])
                video = root / "videos" / f"video_{base_index*6+level_index:05d}.mp4"
                staged = construction_root / (base + f"-{int(level*100):03}.mp4")
                encoded = encode_lossless(edited, staged)
                replay = construction_root / (base + f"-{int(level*100):03}-replay.mp4")
                repeated = encode_lossless(edited, replay)
                decoded = module.load_video(str(staged), num_frames=16, return_tensor=False)
                if encoded["sha256"] != repeated["sha256"] or not np.array_equal(decoded, edited):
                    raise AssertionError("lossless VBench decoder or encoded SHA replay failed")
                if not np.array_equal(decoded[masks == 0], frames[masks == 0]):
                    raise AssertionError("post-decode pixels outside mask changed")
                base_queries.append({"video": video.name, "video_uid": base + ":visibility:" + str(level),
                    "query_uid": base + ":visibility:" + str(level), "base_id": base, "source_group": base,
                    "split": source["split"], "family": "color_visibility_denominator", "visible_fraction": level,
                    "prompt": source["prompt"], "dimension_metadata": {"color": source["gold"]},
                    "source_video_sha256": source["sha256"], "video_sha256": encoded["sha256"],
                    "media_seconds": encoded["media_seconds"], "construction_proof": evidence,
                    "constructed_video": str(staged.resolve())})
            # Frozen lexical control. Failure of Official invariance is reported,
            # never avoided by selecting a convenient synonym after scoring.
            obj, col = source["gold"]["object"], source["gold"]["color"]
            prompt = source["prompt"]
            replacement = "grey" if col == "gray" else (aliases[obj][0] if aliases[obj] else None)
            for query in base_queries:
                destination = root / "videos" / query["video"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(query["constructed_video"])
            if replacement:
                changed = prompt.replace(col, replacement) if col == "gray" else prompt.replace(obj, replacement)
                video = root / "videos" / f"video_{base_index*6+5:05d}.mp4"
                video.symlink_to((root / "videos" / base_queries[0]["video"]).resolve())
                base_queries.append({**base_queries[0], "video": video.name, "prompt": changed,
                    "query_uid": base + ":synonym", "video_uid": base + ":synonym",
                    "family": "color_synonym_control"})
            queries.extend(base_queries)
            proof["status"] = "accepted"
            accepted.append(proof)
        except Exception as exc:
            proof["reason"] = f"{type(exc).__name__}: {exc}"
            rejected.append(proof)
        save(construction_root / (base + ".json"), proof)
        print(json.dumps({"base_id": base, "status": proof["status"], "reason": proof.get("reason")}), flush=True)
    save(root / "metadata.json", {"videos": queries})
    save(root / "freeze.json", {"metadata_sha256": sha256_file(root / "metadata.json"),
        "construction_protocol_sha256": sha256_file(construction), "n_input_bases": len(inputs),
        "n_accepted_bases": len(accepted), "n_queries": len(queries), "rejections": rejected,
        "accepted": accepted, "frozen_before_scores": True, "scorer_received_masks": False})


def score(args):
    root = args.output / "families" / args.dimension
    metadata = json.loads((root / "metadata.json").read_text())["videos"]
    if args.canonical_only:
        metadata = [r for r in metadata if (r.get("variant") == "canonical" if args.dimension == "object_class"
                    else r["family"] == "color_visibility_denominator" and r["visible_fraction"] == 1)]
    module = importlib.import_module(args.dimension + ".metric")
    videos = [root / "videos" / row["video"] for row in metadata]
    entries = {r["video"]: r for r in metadata}
    run_name = args.variant if args.variant == "official" or args.compiler == "metadata" else args.variant + "-" + args.compiler
    if args.canonical_only:
        run_name += "-canonical"
    run = args.output / "scores" / args.dimension / run_name
    if run.exists():
        raise FileExistsError(run)
    start = time.time()
    config = {"model": {"grit": {"checkpoint": str(args.checkpoint)},
               "labels": {"path": str(args.config / "object_color_vocabulary.json")},
               "runtime": {"upstream_root": str(args.upstream)},
               "prompt_compiler": {"kind": args.compiler}},
              "runtime": {"evidence_dir": str(args.output / "evidence" / args.dimension / run_name),
                          "audit_variant": args.variant}}
    if args.compiler in {"base", "lora"}:
        config["model"]["prompt_compiler"]["path"] = str(args.output / "compiled" / f"{args.dimension}-{args.compiler}.json")
    backend = "vbench" if args.variant == "official" else "audit"
    results = run_batch_contract(module.evaluate_batch, backend, videos, entries, device="cuda:0", config=config)
    summary = module.summarize(backend, results)
    write_results(run, results, summary, {"config": config, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "started_unix": start, "wall_seconds": time.time()-start,
                  "family_freeze_sha256": sha256_file(root / "freeze.json"),
                  "code_sha256": {str(p): sha256_file(p) for p in [Path(__file__), *Path("metrics").glob(args.dimension.replace("_", "-")+"/src/**/*.py")]},
                  "status": summary.status, "provenance": [r.metric.get("provenance") for r in results if r.metric.get("provenance")]})
    print(json.dumps(summary.to_dict()), flush=True)


def parity(args):
    family = args.output / "families" / args.dimension
    entries = json.loads((family / "metadata.json").read_text())["videos"]
    selected = [r for r in entries if (r.get("variant") == "canonical" if args.dimension == "object_class"
                                     else r["family"] == "color_visibility_denominator" and r["visible_fraction"] == 1)][:4]
    if len(selected) != 4:
        raise ValueError("four actual inputs required for four-isolated-shard parity")
    root = args.output / "cli_parity" / args.dimension
    root.mkdir(parents=True, exist_ok=False)
    video_dir = root / "videos"
    video_dir.mkdir()
    for entry in selected:
        (video_dir / entry["video"]).symlink_to((family / "videos" / entry["video"]).absolute())
    save(root / "metadata.json", {"videos": selected})
    config = root / "models.toml"
    config.write_text('[runtime]\nupstream_root = '+json.dumps(str(args.upstream))+ '\n[grit]\ncheckpoint = '
        +json.dumps(str(args.checkpoint))+'\n[labels]\npath = '+json.dumps(str((args.config / "object_color_vocabulary.json").resolve()))
        +'\n[prompt_compiler]\nkind = "metadata"\n')
    started = time.time()
    command = [sys.executable, "-m", args.dimension+".cli", "--both", "--video-dir", str(video_dir),
               "--metadata", str(root / "metadata.json"), "--model-config", str(config), "--gpu", "0,1,2,3",
               "--output", str(root / "output")]
    completed = subprocess.run(command, capture_output=True, text=True)
    (root / "stdout.log").write_text(completed.stdout)
    (root / "stderr.log").write_text(completed.stderr)
    checks = []
    for backend, reference in (("vbench", "official"), ("audit", "repair")):
        refs = {r["query_uid"]: r for r in json.loads((args.output / "scores" / args.dimension / reference / "results.json").read_text())}
        paths = list((root / "output" / args.dimension.replace("_", "-") / backend).glob("*/results.json"))
        if len(paths) != 1:
            raise ValueError("public CLI did not write exactly one result run")
        results = json.loads(paths[0].read_text())
        if len(results) != 4 or len({r["video_uid"] for r in results}) != 4:
            raise ValueError("CLI four-shard coverage/UID mismatch")
        for row in results:
            expected = refs[row["query_uid"]]
            if row["status"] != expected["status"]:
                raise ValueError("CLI status differs from batch reference")
            error = abs(row["score"]-expected["score"]) if row["score"] is not None else None
            if error is not None and error > 1e-6:
                raise ValueError("CLI score parity failed")
            worker = row["worker_device"]
            if worker["logical_device"] != "cuda:0" or "," in worker["cuda_visible_devices"]:
                raise ValueError("worker did not isolate one physical CUDA device")
            checks.append({"backend": backend, "query_uid": row["query_uid"], "abs_error": error, "worker": worker})
        if len({r["worker_device"]["cuda_visible_devices"] for r in results}) != 4:
            raise ValueError("four requested shards did not use four distinct GPUs")
    save(root / "parity.json", {"command": command, "return_code": completed.returncode,
        "physical_gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"), "four_card_wall_seconds": time.time()-started,
        "checks": checks, "scope": "public CLI versus direct common batch; not independent frozen E0 replication"})
    print(json.dumps({"dimension": args.dimension, "n": len(checks), "max_abs_error": max(r["abs_error"] or 0 for r in checks)}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["locate", "prepare-object", "prepare-color", "score", "parity"])
    parser.add_argument("--dimension", choices=["object_class", "color"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/four_dimension"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--locator", type=Path)
    parser.add_argument("--locator-sha256")
    parser.add_argument("--segformer", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variant", default="repair", choices=["official", "repair", "legacy", "binding", "binding_lexical"])
    parser.add_argument("--compiler", default="metadata", choices=["metadata", "deterministic", "base", "lora"])
    parser.add_argument("--canonical-only", action="store_true", help="prompt-only head ablation on original queries; excludes metadata-only label interventions")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    {"locate": locate, "prepare-object": prepare_object, "prepare-color": prepare_color, "score": score, "parity": parity}[args.command](args)


if __name__ == "__main__":
    main()
