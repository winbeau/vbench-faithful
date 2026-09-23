"""Evaluate all cached local proposals on disjoint same-region RGB witnesses.

One complete adjacent-phase source shard per worker. No additional global
search or model inference, no motion suppression, no final Dynamic score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.appearance_context import context_masks, evidence, rankings
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import resolve_media, load_masks
from .score_static_jitter import source_hashes
from .static_jitter import digest


def mask_hash(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def witness_groups(pair, masks, shape):
    regions = np.concatenate((masks, np.ones((1, *shape), bool)))
    groups = {}
    for p, point in enumerate(pair["source_points"]):
        for ref in point["regional_supports"]:
            core, annuli = context_masks(regions[ref["region"]], point["xy"], point["size"], ref["factor"])
            keys = [mask_hash(m) for m in (core, *annuli)]
            if keys[0] != ref["support_sha256"]:
                raise ValueError("source support differs from bound appearance evidence")
            group = groups.setdefault(tuple(keys), {"masks": (core, *annuli), "references": []})
            group["references"].append({"source_point": p, **ref})
    return groups


def inspect_pair(frames, masks, pair, update=lambda _: None):
    groups = witness_groups(pair, masks, frames.shape[1:3])
    images = frames.astype(np.float64) / 255.
    core_cache, witness_cache, records = {}, {}, []
    chunks = {"group": [], "displacements": []}
    for role in ("core", "inner", "outer"):
        for value in ("ncc", "zero_ncc", "mse", "zero_mse", "overlap"):
            chunks[role + "_" + value] = []
    offset = 0
    for index, (keys, group) in enumerate(sorted(groups.items())):
        hypotheses = pair["supports"][keys[0]]["hypotheses"]
        displacements = np.array([h["displacement_pixels"] for h in hypotheses], int).reshape(-1, 2)
        zero = np.flatnonzero(np.all(displacements == 0, axis=1))
        baseline = int(zero[0]) if len(zero) else len(displacements)
        if not len(zero):
            displacements = np.concatenate((displacements, [[0, 0]]))
        if keys[0] not in core_cache:
            core_cache[keys[0]] = evidence(images[0], images[1], group["masks"][0], displacements)
            measured = core_cache[keys[0]]["ncc"][:len(hypotheses)]
            if not np.allclose(measured, [h["correlation"] for h in hypotheses], rtol=1e-10, atol=1e-10):
                raise ValueError("cached core proposals do not reproduce native pixels")
        checked = [core_cache[keys[0]]]
        for ring in (1, 2):
            cache_key = (keys[0], keys[ring])
            if cache_key not in witness_cache:
                witness_cache[cache_key] = evidence(images[0], images[1], group["masks"][ring], displacements)
            checked.append(witness_cache[cache_key])
        count = len(displacements)
        record = {"core_key": keys[0], "inner_key": keys[1], "outer_key": keys[2], "references": group["references"],
                  "support_pixels": [int(m.sum()) for m in group["masks"]], "path_start": offset, "candidate_count": count,
                  "parent_hypotheses": len(hypotheses), "zero_control_index": baseline,
                  "rankings": rankings(checked[0], checked[1:]), "score": None}
        records.append(record)
        chunks["group"].append(np.full(count, index, int)); chunks["displacements"].append(displacements)
        for role, values in zip(("core", "inner", "outer"), checked):
            for name, values in values.items():
                chunks[role + "_" + name].append(values)
        offset += count
        update({"groups_completed": index + 1, "groups_expected": len(groups), "candidate_comparisons": offset})
    arrays = {name: np.concatenate(values) if values else np.empty((0, 2) if name == "displacements" else (0,),
                                                               int if name in {"displacements", "group"} else float) for name, values in chunks.items()}
    result = {"start": pair["start"], "lag": pair["lag"], "seconds": pair["seconds"], "status": "diagnostic_only", "score": None,
              "source_queries": len(pair["source_points"]), "groups": records, "candidate_comparisons": offset,
              "warning": "witness pixels were not source templates; common-region translation failure is not a motion veto or jitter label"}
    return result, arrays


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-shard", "source-execution-root", "manifest", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    source, execution, manifest, regions, output = map(Path, (args.source_shard, args.source_execution_root, args.manifest, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh witness output required")
    identity = json.loads((source / "provenance.json").read_text())
    state = json.loads((source / "runtime.json").read_text())
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    if (state["status"] != "finished" or state["failed"] or state["completed"] != state["expected"]
            or len(rows) != state["completed"] or [r["candidate_id"] for r in rows] != identity["sharding"]["assigned"]
            or identity["search"] != "ambiguity" or identity["lags"] != [1]
            or digest(source / "diagnostics.jsonl") != identity["diagnostics_sha256"]
            or digest(manifest) != identity["manifest_sha256"] or digest(regions / "provenance.json") != identity["region_provenance_sha256"]):
        raise ValueError("complete bound adjacent-phase appearance/SAM inputs required")
    for name, sha in {**identity["code_files"], **identity["helper_sha256"], "scripts/counterfactual/probe_local_appearance.py": identity["script_sha256"]}.items():
        if digest(execution / name) != sha:
            raise ValueError("parent execution snapshot changed")
    ledger = {r["candidate_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    paths = {r["candidate_id"]: resolve_media(ledger[r["candidate_id"]], args.video_root) for r in rows}
    root = Path(__file__).resolve().parents[2]
    cv2.setNumThreads(1)
    output.mkdir(parents=True); (output / "evidence").mkdir()
    provenance = {k: identity[k] for k in ("manifest_sha256", "review_sha256", "input_sha256", "region_provenance_sha256", "region_cache_sha256", "sharding", "starts", "lags")}
    provenance.update(status="diagnostic_only", score=None, formal_acceptance="NOT EVALUATED",
                      source_provenance_sha256=digest(source / "provenance.json"), source_diagnostics_sha256=identity["diagnostics_sha256"],
                      config={"witness_radii": "(r,2r] and (2r,4r] clipped to same source SAM region; r=factor*SIFT_size",
                              "min_overlap": .8, "hypotheses": "all parent hypotheses and explicit zero control", "ranking": "core, min(core,inner), min(core,outer), min(core,inner,outer)",
                              "score": None, "note": "source witness pixels are excluded from templates, not statistically independent images"},
                      script_sha256=digest(Path(__file__)), code_files=source_hashes(root),
                      helper_sha256={"scripts/counterfactual/probe_native_region_motion.py": digest(root / "scripts/counterfactual/probe_native_region_motion.py")},
                      environment={"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu"}, evidence_sha256={})
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0, "failed": 0, "pairs_completed": 0,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def update(fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)); update({})
    with (output / "diagnostics.jsonl").open("x") as handle:
        for original in rows:
            key = original["candidate_id"]
            record = {k: original[k] for k in ("candidate_id", "base_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, pairs=[])
            try:
                frames, times, sampling = decode_video(paths[key], TrajectoryConfig(sample_fps=8, max_side=512))
                if (identity["input_sha256"][key] != ledger[key]["sha256"] or list(frames.shape) != ledger[key]["decoded_shape"]
                        or not np.array_equal(times, ledger[key]["pts"]) or original["score"] is not None or original["status"] != "diagnostic_only"
                        or not np.array_equal(times, original["sampling"]["timestamps"])
                        or [(p["start"], p["lag"]) for p in original["pairs"]] != [(i, 1) for i in range(len(frames) - 1)]):
                    raise ValueError("native geometry, times or complete phase contract changed")
                masks = load_masks(regions / "evidence" / f"{key}.npz", identity["region_cache_sha256"][key], frames, times)
                record["sampling"] = sampling
                for pair in original["pairs"]:
                    start = pair["start"]
                    update({"candidate_id": key, "start": start, "groups_completed": 0})
                    result, arrays = inspect_pair(frames[start:start + 2], masks[start], pair, update)
                    arrays["timestamps"] = times[start:start + 2]
                    path = output / "evidence" / f"{key}_start{start:02d}.npz"
                    np.savez_compressed(path, **arrays)
                    provenance["evidence_sha256"][path.name] = digest(path)
                    record["pairs"].append(result)
                    update({"pairs_completed": runtime["pairs_completed"] + 1})
                    print(json.dumps({"candidate_id": key, "start": start, "groups": len(result["groups"]), "comparisons": result["candidate_comparisons"]}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}"); runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            update({"completed": runtime["completed"] + 1})
    update({"status": "finished", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    provenance["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
