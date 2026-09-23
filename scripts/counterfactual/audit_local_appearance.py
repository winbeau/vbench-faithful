"""Verify the full declared local-appearance probe and expose its ambiguity.

Counts/closure are not correctness labels. No candidate is silently discarded,
and the diagnostic deliberately has no Dynamic Degree score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.local_appearance import feature_support
from dynamic_degree.sparse_structure import extract_features
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_local_appearance import feature_groups
from .probe_native_region_motion import load_masks, resolve_media
from .static_jitter import digest


def verify_pair(pair, frame, masks, *, factors=(1, 2, 4)):
    expected = feature_groups(extract_features(frame, "sift"))
    points = pair["source_points"]
    if len(points) != len(expected):
        raise ValueError("not every automatic point was retained")
    all_masks = np.concatenate((masks, np.ones((1, *frame.shape[:2]), bool)))
    if pair["regions"] != len(masks) or pair["whole_frame_control_region"] != len(masks):
        raise ValueError("missing original region or full-image control")
    used = set()
    for wanted, point in zip(expected, points):
        if any(point[k] != wanted[k] for k in wanted):
            raise ValueError("automatic feature position/scale/orientation group changed")
        x, y = np.rint(point["xy"]).astype(int)
        refs = []
        for region in np.flatnonzero(all_masks[:, y, x]):
            for factor in factors:
                mask = feature_support(all_masks[region], point["xy"], point["size"], factor)
                key = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
                support = pair["supports"][key]
                if support["score"] is not None or support["support_pixels"] != int(mask.sum()):
                    raise ValueError("support geometry/null score differs")
                refs.append({"region": int(region), "whole_frame_control": bool(region == len(masks)),
                             "factor": factor, "support_sha256": key})
                used.add(key)
        if refs != point["regional_supports"]:
            raise ValueError("point did not retain all containing regions and all declared scales")
    if used != set(pair["supports"]):
        raise ValueError("unreferenced or missing support")


def support_statistics(supports):
    """One entry per unique source position/scale in a declared region/factor.

    Reverse rank zero and any reverse hypothesis are kept distinct. Even exact
    closure is only self-consistency, not proof of correct physical identity.
    """
    top, gaps, closures, alternative_closures = [], [], [], []
    for support in supports:
        hypotheses = support["hypotheses"]
        if not hypotheses:
            continue
        best = hypotheses[0]
        top.append(best["displacement_pixels"])
        if len(hypotheses) >= 2:
            gaps.append(best["correlation"] - hypotheses[1]["correlation"])
        reverse = best["reverse_candidates"]
        if reverse:
            closures.append(reverse[0]["closure_error_pixels"])
            alternative_closures.append(min(h["closure_error_pixels"] for h in reverse))
    d = np.asarray(top, float).reshape(-1, 2)
    length = np.linalg.norm(d, axis=1)
    return {"source_points": len(supports), "with_forward_hypothesis": len(top),
            "missing": len(supports) - len(top), "with_reverse_hypothesis": len(closures),
            "best_reverse_exact_closure": sum(v == 0 for v in closures),
            "any_reverse_exact_closure": sum(v == 0 for v in alternative_closures),
            "top1_mean_displacement_pixels": d.mean(axis=0).tolist() if len(d) else None,
            "top1_mean_length_pixels": float(length.mean()) if len(d) else None,
            "top1_length_quantiles_pixels": np.quantile(length, [0, .25, .5, .75, 1]).tolist() if len(d) else None,
            "median_top1_top2_correlation_gap": float(np.median(gaps)) if gaps else None,
            "scope": "adaptive points and a single frame pair; closure is not motion correctness"}


def summarize_pair(pair):
    grouped = defaultdict(list)
    for point in pair["source_points"]:
        for ref in point["regional_supports"]:
            grouped[(ref["region"], ref["factor"], ref["whole_frame_control"])].append(pair["supports"][ref["support_sha256"]])
    return {"start": pair["start"], "lag": pair["lag"], "seconds": pair["seconds"],
            "points": len(pair["source_points"]), "unique_supports": len(pair["supports"]),
            "regions": [{"region": key[0], "factor": key[1], "whole_frame_control": key[2],
                         **support_statistics(value)} for key, value in sorted(grouped.items())]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "execution-root", "manifest", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    p.add_argument("--no-display", action="store_true", help="verify all records without creating a very tall all-phase montage")
    args = p.parse_args(argv)
    source, execution, manifest, regions, output = map(Path, (args.source_run, args.execution_root, args.manifest, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh analysis required")
    ledger = {r["candidate_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    shards = sorted(source.glob("shard-*"))
    if not shards:
        raise ValueError("no shards found")
    records, identities, runtimes = [], [], []
    for path in shards:
        identity = json.loads((path / "provenance.json").read_text())
        runtime = json.loads((path / "runtime.json").read_text())
        if (runtime["status"] != "finished" or runtime["completed"] != runtime["expected"]
                or digest(path / "diagnostics.jsonl") != identity["diagnostics_sha256"]
                or digest(manifest) != identity["manifest_sha256"]
                or digest(regions / "provenance.json") != identity["region_provenance_sha256"]):
            raise ValueError("running/partial/changed source")
        for name, sha in {**identity["code_files"], **identity["helper_sha256"],
                          "scripts/counterfactual/probe_local_appearance.py": identity["script_sha256"]}.items():
            if digest(execution / name) != sha:
                raise ValueError("execution snapshot changed")
        rows = list(map(json.loads, (path / "diagnostics.jsonl").read_text().splitlines()))
        if ([r["candidate_id"] for r in rows] != identity["sharding"]["assigned"]
                or sum(r["status"] == "failed" for r in rows) != runtime["failed"]
                or sum(len(r["pairs"]) for r in rows) != runtime["pairs_completed"]):
            raise ValueError("shard records and actual completed work disagree")
        identities.append(identity); runtimes.append(runtime); records.extend(rows)
    reference = identities[0]
    varying = {"sharding", "diagnostics_sha256", "resolved_media"}
    common = {k: v for k, v in reference.items() if k not in varying}
    if (any({k: v for k, v in i.items() if k not in varying} != common for i in identities)
            or sorted(i["sharding"]["shard"] for i in identities) != list(range(reference["sharding"]["shards"]))
            or sorted(r["candidate_id"] for r in records) != sorted(reference["sharding"]["selected_cohort"])):
        raise ValueError("shard identity/selection coverage differs")
    stats, signatures, controls, display = [], {}, [], []
    for record in sorted(records, key=lambda r: r["candidate_id"]):
        key = record["candidate_id"]
        row = ledger[key]
        if record["score"] is not None or reference["input_sha256"][key] != row["sha256"]:
            raise ValueError("score or video identity differs")
        path = resolve_media(row, args.video_root)
        frames, times, _ = decode_video(path, TrajectoryConfig(sample_fps=8, max_side=512))
        if list(frames.shape) != row["decoded_shape"] or not np.array_equal(times, row["pts"]):
            raise ValueError("native media decoding differs")
        masks = load_masks(regions / "evidence" / f"{key}.npz", reference["region_cache_sha256"][key], frames, times)
        wanted = [(s, l) for s in reference["starts"] for l in reference["lags"]]
        if record["status"] != "failed" and [(p["start"], p["lag"]) for p in record["pairs"]] != wanted:
            raise ValueError("missing declared native phases")
        for pair in record["pairs"]:
            verify_pair(pair, frames[pair["start"]], masks[pair["start"]], factors=reference["support_factors"])
        stats.append({k: record[k] for k in ("candidate_id", "base_id", "family", "seed", "status")}
                     | {"pairs": [summarize_pair(pair) for pair in record["pairs"]]})
        signature = hashlib.sha256(json.dumps(record["pairs"], sort_keys=True).encode()).hexdigest()
        if record["family"] == "original":
            signatures[record["base_id"]] = signature
        elif record["family"] == "encoding_control":
            controls.append({"candidate_id": key, "all_pair_fields_exact": signatures.get(record["base_id"]) == signature})
        if args.no_display:
            continue
        for pair in record["pairs"]:
            cells = []
            start, target = pair["start"], pair["start"] + pair["lag"]
            for factor in (None, *reference["support_factors"]):
                canvas = cv2.resize(frames[start if factor is None else target], (512, 512), interpolation=cv2.INTER_NEAREST)
                if factor is not None:
                    for point in pair["source_points"]:
                        ref = next(r for r in point["regional_supports"] if r["whole_frame_control"] and r["factor"] == factor)
                        hypotheses = pair["supports"][ref["support_sha256"]]["hypotheses"]
                        if not hypotheses:
                            continue
                        proposal = hypotheses[0]
                        scale = 512 / np.array(frames.shape[2:0:-1])
                        a = tuple(np.rint(np.array(point["xy"]) * scale).astype(int))
                        b = tuple(np.rint((np.array(point["xy"]) + proposal["displacement_pixels"]) * scale).astype(int))
                        back = proposal["reverse_candidates"]
                        color = (0, 255, 255) if back and back[0]["closure_error_pixels"] == 0 else (255, 60, 60)
                        cv2.arrowedLine(canvas, a, b, color, 1, cv2.LINE_AA, tipLength=.15)
                panel = np.full((562, 512, 3), 255, np.uint8)
                label = f"{key} t={start}" if factor is None else f"t={target} local radius={factor}*SIFT size"
                cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
                cv2.putText(panel, "all points; cyan=exact reverse closure (NOT truth)", (8, 41), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1, cv2.LINE_AA)
                panel[50:] = canvas
                cells.append(panel)
            display.append(np.concatenate(cells, axis=1))
    output.mkdir(parents=True)
    figure_sha = None
    if display:
        image = output / "all_points.png"
        if not cv2.imwrite(str(image), cv2.cvtColor(np.concatenate(display), cv2.COLOR_RGB2BGR)):
            raise RuntimeError("diagnostic display write failed")
        figure_sha = digest(image)
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
              "source_provenance_sha256": {s.name: digest(s / "provenance.json") for s in shards},
              "source_diagnostics_sha256": {s.name: digest(s / "diagnostics.jsonl") for s in shards},
              "script_sha256": digest(Path(__file__)), "runtimes": runtimes, "videos": stats,
              "encoding_controls": controls, "figure_sha256": figure_sha,
              "scope": "declared DEV frame pairs only; adaptive-point averages are not scores or an invariance test"}
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"videos": len(stats), "controls": controls, "runtime_failures": sum(t["failed"] for t in runtimes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
