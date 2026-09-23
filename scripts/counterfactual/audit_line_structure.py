"""Check full line/junction cohort, native geometry, pixel evidence and identity.

All accepted line correlations are remeasured from RGB. One predetermined
source line per frame pair is compared with EVERY target line independently;
this is not an exhaustive independent ranking audit of all rejected queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.trajectory import TrajectoryConfig, decode_video
from .probe_native_region_motion import resolve_media
from .review_selection import select_reviewed_candidates
from .static_jitter import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True), message)


def literal_strip(frame, line, width):
    a, b = np.asarray(line, float)
    vector = b - a
    normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
    xy = ((a + b)[None, None] / 2 + np.linspace(-.75, .75, 32)[:, None, None] * vector
          + np.linspace(-3 * width, 3 * width, 9)[None, :, None] * normal)
    valid = ((xy[..., 0] >= 0) & (xy[..., 0] <= frame.shape[1] - 1)
             & (xy[..., 1] >= 0) & (xy[..., 1] <= frame.shape[0] - 1))
    pixels = cv2.remap(frame.astype(float) / 255, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    return pixels, valid


def literal_ncc(a, ma, b, mb):
    common = ma & mb
    overlap = float(common.mean())
    if common.sum() < 3 or overlap < .8:
        return np.nan, overlap
    x, y = a[common].copy(), b[common].copy()
    floor = 128 * np.finfo(float).eps * max(1, (x * x).sum(), (y * y).sum())
    x -= x.mean(axis=0); y -= y.mean(axis=0)
    ex, ey = (x * x).sum(), (y * y).sum()
    return (float(np.clip((x * y).sum() / np.sqrt(ex * ey), -1, 1)) if min(ex, ey) > floor else np.nan), overlap


def checked_frame(frame, saved):
    detected, widths, precision, _ = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    lines = np.empty((0, 2, 2)) if detected is None else detected.reshape(-1, 2, 2).astype(float)
    widths = np.empty(0) if widths is None else widths.ravel()
    precision = np.empty(0) if precision is None else precision.ravel()
    close(np.asarray(saved["lines"]).reshape(-1, 2, 2), lines, "line detector/geometry mismatch")
    close(saved["widths"], widths, "line widths mismatch"); close(saved["precision"], precision, "line precision mismatch")
    # Independent batched infinite-line solve followed by finite extent checks.
    ii, jj = np.triu_indices(len(lines), 1)
    d, e = lines[ii, 1] - lines[ii, 0], lines[jj, 1] - lines[jj, 0]
    rhs = lines[jj, 0] - lines[ii, 0]
    determinant = d[:, 1] * e[:, 0] - d[:, 0] * e[:, 1]
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    rank = abs(determinant) > 128 * np.finfo(float).eps * lengths[ii] * lengths[jj]
    t = np.divide(-rhs[:, 0] * e[:, 1] + rhs[:, 1] * e[:, 0], determinant,
                  out=np.full(len(ii), np.nan), where=rank)
    u = np.divide(d[:, 0] * rhs[:, 1] - d[:, 1] * rhs[:, 0], determinant,
                  out=np.full(len(ii), np.nan), where=rank)
    xy = lines[ii, 0] + t[:, None] * d
    keep = (rank & (t >= -widths[ii] / lengths[ii]) & (t <= 1 + widths[ii] / lengths[ii])
            & (u >= -widths[jj] / lengths[jj]) & (u <= 1 + widths[jj] / lengths[jj])
            & (xy >= 0).all(axis=1) & (xy <= [frame.shape[1] - 1, frame.shape[0] - 1]).all(axis=1))
    indices = np.flatnonzero(keep)
    require(len(saved["junctions"]) == len(indices), "junction completeness mismatch")
    for j, k in zip(saved["junctions"], indices):
        require(j["lines"] == [int(ii[k]), int(jj[k])], "junction identities mismatch")
        close(j["xy"], xy[k], "junction location mismatch")
        close(j["parameters"], [t[k], u[k]], "junction parameters mismatch")
        close(j["sine_angle"], abs(determinant[k]) / (lengths[ii[k]] * lengths[jj[k]]), "junction angle mismatch")
    return [literal_strip(frame, line, width) for line, width in zip(lines, widths)]


def audit_video(record, frames, times):
    require(record["score"] is None and record["status"] == "diagnostic_only", "not a diagnostic record")
    close(record["sampling"]["timestamps"], times, "timestamp mismatch")
    require(record["decoded_shape"] == list(frames.shape) and len(record["frames"]) == len(frames), "frame inventory mismatch")
    expected = [(start, lag) for lag in (1, 2, 3, 4) for start in range(len(frames) - lag)]
    require([(p["start"], p["lag"]) for p in record["pairs"]] == expected, "phase/pair completeness mismatch")
    strips = [checked_frame(f, saved) for f, saved in zip(frames, record["frames"])]
    counts = {"frames": len(frames), "pairs": len(expected), "detected_lines": sum(len(f["lines"]) for f in record["frames"]),
              "junctions": sum(len(f["junctions"]) for f in record["frames"]), "source_line_queries": 0, "matched_lines": 0,
              "source_junction_queries": 0, "matched_junctions": 0, "identity_supported_lines": 0, "identity_supported_junctions": 0,
              "line_ncc_rechecks": 0, "full_target_rank_queries": 0, "full_target_rank_comparisons": 0}
    speed = {"raw": [], "identity_supported": []}
    for pair in record["pairs"]:
        start, end = pair["start"], pair["start"] + pair["lag"]
        close(pair["seconds"], times[end] - times[start], "pair time mismatch")
        source, target = record["frames"][start], record["frames"][end]
        candidates, matches = pair["lines"]["candidates"], pair["lines"]["matches"]
        require([c["source_key"] for c in candidates] == list(range(len(source["lines"]))), "all source lines must remain")
        counts["source_line_queries"] += len(candidates)
        for candidate in candidates:
            choices = candidate["hypotheses"]
            require(len(choices) <= 3 and len({h["target_key"] for h in choices}) == len(choices), "distinct target hypotheses required")
            require(all(0 <= h["target_key"] < len(target["lines"]) and .8 <= h["common_overlap"] <= 1
                        and np.isfinite(h["correlation"]) and -1 <= h["correlation"] <= 1 for h in choices), "invalid hypotheses")
            require(all(a["correlation"] >= b["correlation"] for a, b in zip(choices, choices[1:])), "unordered hypotheses")
        require(len({m["source_key"] for m in matches}) == len(matches) == len({m["target_key"] for m in matches}), "one-to-one line match failure")
        counts["matched_lines"] += len(matches)
        for match in matches:
            i, j = match["source_key"], match["target_key"]
            require(0 <= i < len(candidates) and candidates[i]["hypotheses"], "matched source missing")
            choice = candidates[i]["hypotheses"][0]
            require(all(match[k] == v for k, v in choice.items()) and match["ratio"] == candidates[i]["ratio"]
                    and 0 <= match["ratio"] < .75 and 0 <= match["reverse_ratio"] < .75, "line acceptance mismatch")
            a = np.asarray(source["lines"][i]); b = np.asarray(target["lines"][j])
            tb, tm = strips[end][j]
            if match["reversed"]:
                b, tb, tm = b[::-1], tb[::-1, ::-1], tm[::-1, ::-1]
            close(match["source_endpoints"], a, "source endpoints mismatch"); close(match["target_endpoints"], b, "target endpoints mismatch")
            close(match["endpoint_displacements"], b - a, "endpoint displacement mismatch")
            close(match["source_xy"], a.mean(axis=0), "source center mismatch"); close(match["target_xy"], b.mean(axis=0), "target center mismatch")
            corr, common = literal_ncc(*strips[start][i], tb, tm)
            close(match["correlation"], corr, "accepted native NCC mismatch"); close(match["common_overlap"], common, "accepted visibility mismatch")
            counts["line_ncc_rechecks"] += 1
        if candidates:
            # Fixed index arithmetic, not selected by motion, seed or success.
            i = (37 * start + 11 * pair["lag"]) % len(candidates)
            options = []
            for j, (b, mb) in enumerate(strips[end]):
                plain, po = literal_ncc(*strips[start][i], b, mb)
                reverse, ro = literal_ncc(*strips[start][i], b[::-1, ::-1], mb[::-1, ::-1])
                flip = (reverse if np.isfinite(reverse) else -2) > (plain if np.isfinite(plain) else -2)
                value, overlap = (reverse, ro) if flip else (plain, po)
                if np.isfinite(value):
                    options.append({"target_key": j, "correlation": value, "common_overlap": overlap, "reversed": flip})
            options.sort(key=lambda o: -o["correlation"])
            choices = candidates[i]["hypotheses"]
            require(len(choices) == min(3, len(options)), "top-three completeness mismatch")
            for actual, expected_choice in zip(choices, options[:3]):
                require(actual["target_key"] == expected_choice["target_key"] and actual["reversed"] == expected_choice["reversed"], "global target ranking mismatch")
                close(actual["correlation"], expected_choice["correlation"], "global NCC mismatch")
                close(actual["common_overlap"], expected_choice["common_overlap"], "global visibility mismatch")
            if len(options) < 2:
                require(candidates[i]["ratio"] is None, "missing competitor must not be unique")
            else:
                d1, d2 = [max(0., 2 - 2 * c["correlation"]) for c in options[:2]]
                # Near exact copies sqrt(1-NCC) amplifies harmless roundoff.
                # Verify the defining squared-distance equation instead.
                if d2 > 1e-12:
                    close(candidates[i]["ratio"] ** 2 * d2, d1, "global ratio mismatch")
                else:
                    close(candidates[i]["ratio"], 1., "ambiguous duplicate ratio mismatch")
            counts["full_target_rank_queries"] += 1
            counts["full_target_rank_comparisons"] += len(strips[end])
        mapping = {m["source_key"]: m["target_key"] for m in matches}
        targets = {tuple(j["lines"]): i for i, j in enumerate(target["junctions"])}
        observed = {m["source_key"]: m for m in pair["junctions"]["matches"]}
        missing = {m["source_key"]: m["reason"] for m in pair["junctions"]["missing"]}
        require(len(observed) == len(pair["junctions"]["matches"]) and len(missing) == len(pair["junctions"]["missing"])
                and not set(observed) & set(missing) and set(observed) | set(missing) == set(range(len(source["junctions"]))), "junction coverage mismatch")
        counts["source_junction_queries"] += len(source["junctions"]); counts["matched_junctions"] += len(observed)
        for i, node in enumerate(source["junctions"]):
            if not all(k in mapping for k in node["lines"]):
                require(missing.get(i) == "missing_line_correspondence", "wrong missing-line reason"); continue
            j = targets.get(tuple(sorted(mapping[k] for k in node["lines"])))
            if j is None:
                require(missing.get(i) == "target_intersection_not_visible", "wrong missing-intersection reason"); continue
            require(i in observed and observed[i]["target_key"] == j, "junction mapping mismatch")
            other, actual = target["junctions"][j], observed[i]
            require(actual["source_lines"] == node["lines"] and actual["target_lines"] == other["lines"], "junction parent mismatch")
            close(actual["source_xy"], node["xy"], "junction source mismatch"); close(actual["target_xy"], other["xy"], "junction target mismatch")
            close(actual["displacement_pixels"], np.subtract(other["xy"], node["xy"]), "junction displacement mismatch")
        if pair["lag"] == 1:
            passed = {s["source_key"] for s in pair["junctions"]["identity_support"] if s["supported_without_conflict"]}
            for m in observed.values():
                value = float(np.linalg.norm(m["displacement_pixels"]) / min(frames.shape[1:3]) / pair["seconds"])
                speed["raw"].append(value)
                if m["source_key"] in passed:
                    speed["identity_supported"].append(value)
    # Independent index composition over all available third frames.
    for kind in ("lines", "junctions"):
        edges = {}
        for p in record["pairs"]:
            a, b = p["start"], p["start"] + p["lag"]
            mapping = {m["source_key"]: m["target_key"] for m in p[kind]["matches"]}
            require(len(mapping) == len(set(mapping.values())), "duplicate target identity")
            edges[a, b] = mapping; edges[b, a] = {v: k for k, v in mapping.items()}
        for p in record["pairs"]:
            require(len(p[kind]["matches"]) == len(p[kind]["identity_support"]), "temporal identity length mismatch")
            a, b = p["start"], p["start"] + p["lag"]
            for m, saved in zip(p[kind]["matches"], p[kind]["identity_support"]):
                good, bad = [], []
                for third in range(len(frames)):
                    x, y = edges.get((a, third), {}).get(m["source_key"]), edges.get((b, third), {}).get(m["target_key"])
                    if x is not None and y is not None:
                        (good if x == y else bad).append(third)
                expected_support = {"source_key": m["source_key"], "target_key": m["target_key"], "confirmed_frames": good,
                                    "conflicting_frames": bad, "supported_without_conflict": bool(good) and not bad}
                require(saved == expected_support, "third-frame identity mismatch")
                counts["identity_supported_" + kind] += int(saved["supported_without_conflict"])
    counts["adjacent_conditional_junction_speed"] = {
        k: {"samples": len(v), "mean": float(np.mean(v)) if v else None} for k, v in speed.items()}
    counts["warning"] = "conditional matched junctions, not physical-motion truth, video coverage, invariance or a score"
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-shard", action="append", required=True)
    for name in ("execution-root", "manifest", "review", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    root, execution, manifest, review, output = Path(__file__).resolve().parents[2], *map(Path, (args.execution_root, args.manifest, args.review, args.output))
    if output.exists():
        raise FileExistsError("fresh audit output required")
    ledger = select_reviewed_candidates(list(map(json.loads, manifest.read_text().splitlines())), review, manifest, root)
    all_ids = [r["candidate_id"] for r in ledger]
    rows = {r["candidate_id"]: r for r in ledger}
    report = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "videos": {}, "runtime": [],
              "source_provenance_sha256": {}, "script_sha256": digest(Path(__file__)), "encoding_controls": [],
              "not_verified": ["all rejected query rankings", "semantic/physical identity", "invariance", "natural motion noninferiority", "final score"]}
    cv2.setNumThreads(1)
    scientific = {}
    for source in map(Path, args.source_shard):
        ident = json.loads((source / "provenance.json").read_text()); state = json.loads((source / "runtime.json").read_text())
        require(state["status"] == "finished" and not state["failed"] and state["completed"] == state["expected"], "incomplete source run")
        require(ident["manifest_sha256"] == digest(manifest) and ident["review_sha256"] == digest(review)
                and ident["input_sha256"] == {k: r["sha256"] for k, r in rows.items()} and ident["sharding"]["full_cohort"] == all_ids
                and ident["diagnostics_sha256"] == digest(source / "diagnostics.jsonl"), "source identity mismatch")
        require(ident["config"]["protocol"] == "native_line_junction_dev_v1" and ident["config"]["lags"] == [1, 2, 3, 4], "unsupported protocol")
        for name, sha in {**ident["code_files"], **ident["helper_sha256"], "scripts/counterfactual/probe_line_structure.py": ident["script_sha256"]}.items():
            require(digest(execution / name) == sha, "execution snapshot changed")
        assigned, pair_count = [], 0
        with (source / "diagnostics.jsonl").open() as handle:
            for line in handle:
                record = json.loads(line); key = record["candidate_id"]
                require(key in rows and key not in report["videos"], "duplicate or unexpected source")
                row = rows[key]
                require(all(record[k] == row[k] for k in ("base_id", "prompt_id", "family", "seed")), "cohort label mismatch")
                frames, times, _ = decode_video(resolve_media(row, args.video_root), TrajectoryConfig(sample_fps=8, max_side=512))
                require(list(frames.shape) == row["decoded_shape"] and np.array_equal(times, row["pts"]), "native input mismatch")
                result = audit_video(record, frames, times)
                report["videos"][key] = {k: row[k] for k in ("base_id", "family", "seed")}
                report["videos"][key].update(result)
                fields = {k: record[k] for k in ("frames", "pairs", "sampling", "decoded_shape", "media_duration_seconds")}
                scientific[key] = hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()
                assigned.append(key); pair_count += result["pairs"]
                print(json.dumps({"candidate_id": key, "checked": result["pairs"], "accepted_line_ncc": result["line_ncc_rechecks"]}), flush=True)
        require(assigned == ident["sharding"]["assigned"] == all_ids[ident["sharding"]["shard"]::ident["sharding"]["shards"]]
                and len(assigned) == state["completed"] and pair_count == state["pairs_completed"], "sharding/runtime mismatch")
        report["runtime"].append(state); report["source_provenance_sha256"][str(source)] = digest(source / "provenance.json")
    require(set(report["videos"]) == set(all_ids), "full cohort required")
    for row in ledger:
        if row["family"] != "original":
            continue
        control = next(r for r in ledger if r["base_id"] == row["base_id"] and r["family"] == "encoding_control")
        require(scientific[row["candidate_id"]] == scientific[control["candidate_id"]], "encoding control differs")
        report["encoding_controls"].append([row["candidate_id"], control["candidate_id"]])
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
