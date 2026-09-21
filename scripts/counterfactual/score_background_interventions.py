"""Score every native background intervention with independent localization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .run_background_development import score_record
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--natural-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")
    config = ROOT/"configs/background-repair"
    protocol_path = config/"development_protocol_v2.json"
    protocol = json.loads(protocol_path.read_text())
    dataset_run = json.loads((args.dataset/"run.json").read_text())
    if not dataset_run.get("completed") or dataset_run["index_sha256"] != sha256_file(args.dataset/"index.jsonl"):
        raise ValueError("dataset is incomplete or modified")
    if dataset_run["protocol_sha256"] != sha256_file(config/"construction_dev_v1.json"):
        raise ValueError("construction protocol changed")
    entries = read_jsonl(args.dataset/"index.jsonl")
    if len(entries) != 680 or len({r["video_uid"] for r in entries}) != 680:
        raise ValueError("expected every native background dev candidate")
    cache = {}
    for directory in sorted(args.natural_run.glob("shard*")):
        if not directory.is_dir():
            continue
        run = json.loads((directory/"run.json").read_text())
        if (not run.get("completed") or run["protocol_sha256"] != sha256_file(protocol_path)
            or sha256_file(directory/"scores.jsonl") != run["scores_sha256"]):
            raise ValueError("natural scoring cache is incomplete or uses another protocol")
        for source in ("metrics/background-consistency/src/background_consistency/algorithms.py",
                       "metrics/background-consistency/src/background_consistency/models.py",
                       "metrics/background-consistency/src/background_consistency/runtime.py",
                       "packages/audit-models/src/vbench_audit_models/foreground.py"):
            if run["source_sha256"][source] != sha256_file(ROOT/source):
                raise ValueError("cached clean representation or localizer implementation changed")
        for row in read_jsonl(directory/"scores.jsonl"):
            if row["video_uid"] in cache:
                raise ValueError("duplicate natural cache identity")
            cache[row["video_uid"]] = (directory, row)
    if set(cache) != {r["video_uid"] for r in entries}:
        raise ValueError("natural cache and construction input identities differ")

    import numpy as np
    import torch
    from background_consistency.models import build_model
    from background_consistency.runtime import build_foreground_provider
    deterministic_setup(); torch.set_num_threads(2)
    output = new_output(args.output)
    started = time.time()
    encoder = build_model(protocol["clip"], device=args.device)
    provider = build_foreground_provider(protocol["localizer"], device=args.device)
    selected = entries[args.shard_index::args.num_shards]
    run = {"started_unix": started, "video_count": len(selected), "protocol_sha256": sha256_file(protocol_path),
           "dataset_index_sha256": sha256_file(args.dataset/"index.jsonl"), "shard_index": args.shard_index,
           "num_shards": args.num_shards, "device": args.device, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "encoder": encoder.provenance, "localizer": provider.provenance,
           "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for d in
               (ROOT/"metrics/background-consistency/src", ROOT/"packages/audit-models/src", ROOT/"scripts/counterfactual") for p in d.rglob("*.py")}}
    write_json(output/"run.json", run)
    failures = accepted = 0
    for index, entry in enumerate(selected):
        row = {**entry, "variants": {}}
        try:
            path = artifact_path(args.dataset, entry["manifest"])
            if sha256_file(path) != entry["manifest_sha256"]:
                raise ValueError("construction manifest hash mismatch")
            source = json.loads(path.read_text())
            row.update(base=source["base"], construction_status=source["status"])
            if source["status"] != "accepted":
                row["status"] = "construction_rejected" if source["status"] == "construction_rejected" else "construction_failed"
            else:
                accepted += 1
                for variant, files in source["variants"].items():
                    frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0, 3, 1, 2)
                    frame_hash = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
                    if variant == "clean":
                        directory, cached = cache[entry["video_uid"]]
                        if cached["status"] != "completed" or cached["video_sha256"] != source["video_sha256"]:
                            raise ValueError("cached clean video identity mismatch")
                        ref = cached["scoring_mask"]
                        mask_path = artifact_path(directory, ref["path"])
                        if sha256_file(mask_path) != ref["sha256"]:
                            raise ValueError("cached clean scoring mask changed")
                        with np.load(mask_path, allow_pickle=False) as data:
                            meta = json.loads(str(data["metadata_json"]))
                            if (meta["frame_array_sha256"] != frame_hash or meta["role"] != "scoring"
                                or meta["construction_masks_reused"] is not False):
                                raise ValueError("clean cache does not match the exact PNG tensor")
                        row["variants"][variant] = {"scores": cached["scores"], "background_fraction": cached["background_fraction"],
                            "localizer_diagnostics": cached["localizer_diagnostics"], "source_scores": str(directory/"scores.jsonl"),
                            "source_scores_sha256": sha256_file(directory/"scores.jsonl"), "scoring_mask": {"path": str(mask_path), "sha256": ref["sha256"]}}
                    else:
                        masks = provider.masks_for(frames)
                        identity = entry["video_uid"]+"__"+variant.replace("/", "__")
                        mask_path = output/"scoring_masks"/f"{identity}.npz"
                        write_npz(mask_path, masks=masks, metadata_json=json.dumps({**provider.provenance,
                            "frame_array_sha256": frame_hash, "construction_manifest_sha256": entry["manifest_sha256"]}, sort_keys=True))
                        row["variants"][variant] = score_record(encoder, frames, masks, output, identity,
                            provenance={"path": str(mask_path.relative_to(output)), "sha256": sha256_file(mask_path)})
                        row["variants"][variant]["localizer_diagnostics"] = provider.last_diagnostics
                row["status"] = "completed"
        except Exception as exc:
            row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
            failures += 1
        with (output/"scores.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
        progress = {"processed": index+1, "total": len(selected), "accepted": accepted,
                    "failures": failures, "elapsed_seconds": time.time()-started}
        write_json(output/"progress.json", progress); print(json.dumps(progress), flush=True)
    run.update(completed=True, runtime_failures=failures, finished_unix=time.time(), scores_sha256=sha256_file(output/"scores.jsonl"))
    write_json(output/"run.json", run)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
