"""Translation/affine refinement of ALL proposals in declared support scales.

Input proposals come from the complete same-video local-appearance diagnostic.
This is still a correspondence experiment, not a Dynamic score or invariance
claim. Models see only the two RGB frames, source support and proposed transform.
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

from dynamic_degree.local_appearance import feature_support, refine_local_appearance
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import load_masks, resolve_media
from .score_static_jitter import source_hashes
from .static_jitter import digest


def declared_supports(pair, masks, factors):
    """Deduplicate computation only; all point/region memberships stay in refs."""
    all_masks = np.concatenate((masks, np.ones((1, *masks.shape[1:]), bool)))
    selected = {}
    for point_index, point in enumerate(pair["source_points"]):
        for ref in point["regional_supports"]:
            if ref["factor"] not in factors:
                continue
            key = ref["support_sha256"]
            if key not in selected:
                mask = feature_support(all_masks[ref["region"]], point["xy"], point["size"], ref["factor"])
                if hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() != key:
                    raise ValueError("source support changed")
                selected[key] = {"mask": mask, "references": []}
            selected[key]["references"].append({"point": point_index, **ref})
    return selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "manifest", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--factors", nargs="+", type=int, required=True, choices=[1, 2, 4])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.shards or len(args.factors) != len(set(args.factors)):
        p.error("valid sharding and distinct support factors required")
    source, manifest, regions, output = map(Path, (args.source_run, args.manifest, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh output required")
    rows, identities = [], []
    shard_paths = sorted(source.glob("shard-*"))
    for shard in shard_paths:
        identity = json.loads((shard / "provenance.json").read_text())
        runtime = json.loads((shard / "runtime.json").read_text())
        if (runtime["status"] != "finished" or runtime["failed"] or runtime["completed"] != runtime["expected"]
                or digest(shard / "diagnostics.jsonl") != identity["diagnostics_sha256"]
                or digest(regions / "provenance.json") != identity["region_provenance_sha256"]
                or digest(manifest) != identity["manifest_sha256"]):
            raise ValueError("complete hash-bound proposals/regions/media required")
        batch = list(map(json.loads, (shard / "diagnostics.jsonl").read_text().splitlines()))
        if [r["candidate_id"] for r in batch] != identity["sharding"]["assigned"]:
            raise ValueError("incomplete source shard")
        rows.extend(batch); identities.append(identity)
    if not identities:
        raise ValueError("no complete source shards")
    identity = identities[0]
    rows.sort(key=lambda r: r["candidate_id"])
    common = {k: v for k, v in identity.items() if k not in {"sharding", "diagnostics_sha256", "resolved_media"}}
    if (any({k: v for k, v in i.items() if k not in {"sharding", "diagnostics_sha256", "resolved_media"}} != common for i in identities)
            or sorted(i["sharding"]["shard"] for i in identities) != list(range(identity["sharding"]["shards"]))
            or [r["candidate_id"] for r in rows] != sorted(identity["sharding"]["selected_cohort"])):
        raise ValueError("source cohort/protocol coverage differs")
    full_ids = [r["candidate_id"] for r in rows]
    assigned = rows[args.shard::args.shards]
    if not assigned:
        raise ValueError("empty shard")
    ledger = {r["candidate_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    if any(ledger[key]["sha256"] != identity["input_sha256"][key] for key in full_ids):
        raise ValueError("media identity differs")
    paths = {r["candidate_id"]: resolve_media(ledger[r["candidate_id"]], args.video_root) for r in assigned}
    root = Path(__file__).resolve().parents[2]
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    provenance = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                  "source_provenance_sha256": {s.name: digest(s / "provenance.json") for s in shard_paths},
                  "source_diagnostics_sha256": {s.name: digest(s / "diagnostics.jsonl") for s in shard_paths},
                  "manifest_sha256": digest(manifest), "input_sha256": identity["input_sha256"],
                  "region_provenance_sha256": identity["region_provenance_sha256"],
                  "region_cache_sha256": identity["region_cache_sha256"],
                  "factors": args.factors, "models": ["translation", "affine"], "max_iterations": 30,
                  "min_fixed_visible_overlap": .8, "starts": identity["starts"], "lags": identity["lags"],
                  "sharding": {"shard": args.shard, "shards": args.shards, "selected_cohort": full_ids,
                               "assigned": [r["candidate_id"] for r in assigned]},
                  "script_sha256": digest(Path(__file__)), "code_files": source_hashes(root),
                  "helper_sha256": {"scripts/counterfactual/probe_native_region_motion.py": digest(root / "scripts/counterfactual/probe_native_region_motion.py")},
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu"},
                  "scope": "declared DEV pairs/scales only, same RGB evidence used to fit and evaluate; not certified motion"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(assigned), "completed": 0, "failed": 0,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def update(**fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    update()
    with (output / "diagnostics.jsonl").open("x") as out:
        for prior in assigned:
            key = prior["candidate_id"]
            record = {k: prior[k] for k in ("candidate_id", "base_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, pairs=[])
            try:
                frames, times, _ = decode_video(paths[key], TrajectoryConfig(sample_fps=8, max_side=512))
                if list(frames.shape) != ledger[key]["decoded_shape"] or not np.array_equal(times, ledger[key]["pts"]):
                    raise ValueError("native geometry/time changed")
                masks = load_masks(regions / "evidence" / f"{key}.npz", identity["region_cache_sha256"][key], frames, times)
                for pair in prior["pairs"]:
                    start, target = pair["start"], pair["start"] + pair["lag"]
                    supports = declared_supports(pair, masks[start], args.factors)
                    results = {}
                    for index, (support_key, support) in enumerate(supports.items()):
                        hypotheses = []
                        for candidate in pair["supports"][support_key]["hypotheses"]:
                            hypotheses.append({"initial": candidate["displacement_pixels"],
                                               "refinements": {model: refine_local_appearance(frames[start], frames[target], support["mask"],
                                                                candidate["displacement_pixels"], model=model)
                                                               for model in provenance["models"]}})
                        results[support_key] = {"references": support["references"], "hypotheses": hypotheses}
                        update(candidate_id=key, start=start, lag=pair["lag"], supports_completed=index + 1, supports_expected=len(supports))
                    record["pairs"].append({"start": start, "lag": pair["lag"], "seconds": pair["seconds"], "supports": results})
                    print(json.dumps({"candidate_id": key, "supports": len(supports)}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            out.write(json.dumps(record, allow_nan=False) + "\n"); out.flush()
            update(completed=runtime["completed"] + 1)
    update(status="finished", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    provenance["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
