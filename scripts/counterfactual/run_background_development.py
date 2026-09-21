"""Real background scoring: dev-only natural cohort or cached region diagnostic.

The region cohort is labelled as reused SUBJECT-source diagnostic material.
All scoring masks were independently produced on the actual variants, never
copied from construction or a different variant. Natural videos use a new
class-agnostic localizer; neither path reads human preference labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def score_record(encoder, frames, masks, output, identity, *, provenance):
    from background_consistency.algorithms import score_views
    from background_consistency.runtime import encode_views
    views, fractions = encode_views(encoder, frames, masks)
    path = output / "features" / f"{identity}.npz"
    write_npz(path, **{k: v.detach().cpu().numpy() for k, v in views.items()},
              **{k+"_background_fraction": v for k, v in fractions.items()})
    scores = score_views(views, fractions)
    return {"scores": {k: {"status": "succeeded", "score": v} for k, v in scores.items()},
            "features": {"path": str(path.relative_to(output)), "sha256": sha256_file(path)},
            "background_fraction": {k: v.tolist() for k, v in fractions.items()},
            "scoring_mask": provenance, "num_frames": len(frames)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("natural_dev", "region_diagnostic"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/background-repair/development_protocol_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cached-scores", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")
    config = ROOT / "configs/background-repair"
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    manifest_path = config / protocol.get("manifest_file", "natural1720_manifest.jsonl")
    if sha256_file(manifest_path) != protocol["natural_manifest_sha256"]:
        raise ValueError("natural manifest changed")
    if args.mode == "natural_dev":
        entries = [r for r in read_jsonl(manifest_path) if r["split"] == "dev"]
        if len(entries) != 680:
            raise ValueError("expected the entire 680-video development population")
    else:
        if not args.dataset or not args.cached_scores:
            parser.error("region diagnostic needs dataset and cached-scores")
        entries = read_jsonl(args.dataset / "index.jsonl")
        cache = {}
        for directory in sorted(args.cached_scores.glob("shard*")):
            if not directory.is_dir():
                continue
            run = json.loads((directory / "run.json").read_text())
            if not run.get("completed") or sha256_file(directory / "scores.jsonl") != run["scores_sha256"]:
                raise ValueError("incomplete or altered independent scoring-mask run")
            if run["dataset_index_sha256"] != sha256_file(args.dataset / "index.jsonl"):
                raise ValueError("independent scoring masks belong to a different immutable dataset")
            for row in read_jsonl(directory / "scores.jsonl"):
                bid = row["base"]["base_id"]
                if bid in cache:
                    raise ValueError("duplicate cached base")
                cache[bid] = (directory, row)
        if len(entries) != 288 or set(cache) != {r["base_id"] for r in entries}:
            raise ValueError("region diagnostic must retain all 288 candidate records")

    import numpy as np
    import torch
    from background_consistency.models import build_model
    from background_consistency.runtime import build_foreground_provider

    deterministic_setup()
    torch.set_num_threads(2)
    output = new_output(args.output)
    started = time.time()
    encoder = build_model(protocol["clip"], device=args.device)
    provider = build_foreground_provider(protocol["localizer"], device=args.device) if args.mode == "natural_dev" else None
    selected = entries[args.shard_index::args.num_shards]
    run = {"mode": args.mode, "protocol_sha256": sha256_file(protocol_path), "started_unix": started,
           "video_count": len(selected), "shard_index": args.shard_index, "num_shards": args.num_shards,
           "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": args.device,
           "encoder": encoder.provenance, "localizer": provider.provenance if provider else "independent per-variant cached MobileSAM",
           "cohort": "official background dev" if provider else "reused subject-source diagnostic; no background test claim",
           "torch_version": torch.__version__, "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p)
             for directory in (ROOT/"metrics/background-consistency/src", ROOT/"packages/audit-models/src", ROOT/"scripts/counterfactual")
             for p in directory.rglob("*.py")}}
    write_json(output / "run.json", run)
    write_json(output / "protocol.json", protocol)
    failures = completed = 0
    for index, entry in enumerate(selected):
        row = dict(entry)
        try:
            if args.mode == "natural_dev":
                path = args.video_root / entry["relative_video_path"]
                row["video_sha256"] = sha256_file(path)
                frames = encoder.decode(path)
                masks = provider.masks_for(frames)
                row["localizer_diagnostics"] = provider.last_diagnostics
                mask_path = output / "scoring_masks" / f"{entry['video_uid']}.npz"
                write_npz(mask_path, masks=masks, metadata_json=json.dumps({**provider.provenance,
                    "source_video_sha256": row["video_sha256"],
                    "frame_array_sha256": hashlib.sha256(frames.to(torch.uint8).numpy().tobytes()).hexdigest()}, sort_keys=True))
                row.update(score_record(encoder, frames, masks, output, entry["video_uid"],
                           provenance={"path": str(mask_path.relative_to(output)), "sha256": sha256_file(mask_path)}))
                row["upstream_score"] = encoder.upstream_score(path)
                row["parity_absolute_error"] = abs(row["scores"]["official"]["score"] - row["upstream_score"])
                if row["parity_absolute_error"] > 1e-6:
                    raise ValueError("official real-video parity failed")
                # Preserve actual media duration, including variable-delay GIFs.
                if path.suffix == ".gif":
                    from PIL import Image, ImageSequence
                    with Image.open(path) as image:
                        row["media_duration_seconds"] = sum(f.info.get("duration", 100) for f in ImageSequence.Iterator(image)) / 1000
                else:
                    import decord
                    reader = decord.VideoReader(str(path), num_threads=1)
                    row["media_duration_seconds"] = len(reader) / reader.get_avg_fps()
            else:
                source_path = artifact_path(args.dataset, entry["manifest"])
                if sha256_file(source_path) != entry["manifest_sha256"]:
                    raise ValueError("construction manifest changed")
                source = json.loads(source_path.read_text())
                row.update(base=source["base"], construction_status=source["status"],
                           rejection_reasons=source["rejection_reasons"], variants={})
                if source["status"] != "accepted":
                    row["status"] = "construction_rejected"
                else:
                    directory, cached = cache[entry["base_id"]]
                    if cached["status"] != "completed" or set(cached["variants"]) != set(source["variants"]):
                        raise ValueError("independent mask cache has incomplete variant coverage")
                    for variant, files in source["variants"].items():
                        frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0, 3, 1, 2)
                        ref = cached["variants"][variant]["scoring_mask_file"]
                        path = artifact_path(directory, ref["path"])
                        if sha256_file(path) != ref["sha256"]:
                            raise ValueError("scoring-mask hash mismatch")
                        with np.load(path, allow_pickle=False) as data:
                            meta = json.loads(str(data["metadata_json"]))
                            if (meta.get("role") != "scoring" or meta.get("construction_masks_reused") is not False
                                or meta["source_frame_sha256"] != files[0]["sha256"]
                                or meta["weights_sha256"] != protocol["localizer"]["sam_sha256"]):
                                raise ValueError("scoring mask provenance does not match actual variant")
                            masks = data["masks"][:, 0]
                        identity = entry["base_id"] + "__" + variant.replace("/", "__")
                        row["variants"][variant] = score_record(encoder, frames, masks, output, identity,
                            provenance={"path": str(path), "sha256": ref["sha256"], "source_scores_sha256": sha256_file(directory/"scores.jsonl")})
                    original = args.video_root / source["base"]["relative_video_path"]
                    row["upstream_score"] = encoder.upstream_score(original)
                    row["parity_absolute_error"] = abs(row["variants"]["clean"]["scores"]["official"]["score"] - row["upstream_score"])
                    if row["parity_absolute_error"] > 1e-6:
                        raise ValueError("region clean decode/origin parity failed")
            if row.get("status") != "construction_rejected":
                row["status"] = "completed"
                completed += 1
        except Exception as exc:
            row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
            failures += 1
        with (output / "scores.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        progress = {"processed": index + 1, "total": len(selected), "scored": completed,
                    "failed": failures, "elapsed_seconds": time.time()-started}
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    run.update(completed=True, finished_unix=time.time(), wall_seconds=time.time()-started,
               runtime_failures=failures, scores_sha256=sha256_file(output/"scores.jsonl"))
    write_json(output/"run.json", run)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
