"""DEV-only flow/trajectory description, NOT a Repair or a classifier.

CPU replay of immutable dense caches. Every reviewed original/control/CF is
retained. No new model inference, filtering, threshold search, or final scores.
Temporal diagnostics and spatial predictability are deliberately separate:
periodic real motion and smooth coordinate jitter can share either property.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from dynamic_degree.dense_paths import local_dense_paths
from dynamic_degree.local_trajectory import LocalTrajectoryConfig
from dynamic_degree.trajectory import decode_video
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def interval_owners(timestamps, windows, span):
    """Each lag interval once, all start phases, maximal interior margin.

    Unlike independently averaging overlapping windows, this does not multiply
    middle-of-video observations. Uncovered intervals are explicit -1 values.
    """
    t = np.asarray(timestamps, float)
    if t.ndim != 1 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("strictly increasing finite timestamps required")
    if not isinstance(span, int) or not 0 < span < len(t):
        raise ValueError("span must be a positive integer shorter than the video")
    owners = np.full(len(t) - span, -1, int)
    margin = np.full(len(owners), -np.inf)
    for k, (a, b) in enumerate(windows):
        if not 0 <= a < b <= len(t):
            raise ValueError("invalid window bounds")
        starts = np.arange(a, max(a, b - span))
        value = np.minimum(t[starts] - t[a], t[b - 1] - t[starts + span])
        replace = value > margin[starts]
        owners[starts[replace]] = k
        margin[starts[replace]] = value[replace]
    return owners


def temporal_diagnostics(paths, timestamps, short_side):
    """Path length/net displacement and reversal, with explicit missingness.

    'reliable_only' is a CONDITIONAL subset, never an unbiased full-video score.
    Lag k compares endpoint distance to the arc along k adjacent transitions.
    Reliability requires every intervening transition, not just the endpoints.
    """
    t = np.asarray(timestamps, float)
    windows = paths["windows"]
    count = len(paths["queries"])
    if not np.isfinite(short_side) or short_side <= 0 or not count:
        raise ValueError("positive geometry and nonempty grid required")
    for k, (a, b) in enumerate(windows):
        xy = paths[f"window_{k}_tracks"]
        mask = paths[f"window_{k}_reliable_pair"]
        if xy.shape != (b - a, count, 2) or mask.shape != (b - a - 1, count) or not np.isfinite(xy).all():
            raise ValueError("invalid local trajectory geometry/evidence")
    result = {"lags": {}}
    for lag in range(1, min(4, len(t) - 1) + 1):
        owner = interval_owners(t, windows, lag)
        sums = {name: dict(net=0., arc=0., duration=0., count=0) for name in ("all", "reliable_only")}
        durations = []
        for start, k in enumerate(owner):
            if k < 0:
                continue
            j = start - windows[k][0]
            xy = paths[f"window_{k}_tracks"][j:j + lag + 1]
            reliable = paths[f"window_{k}_reliable_pair"][j:j + lag].all(axis=0)
            net = np.linalg.norm(xy[-1] - xy[0], axis=-1)
            arc = np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum(axis=0)
            dt = t[start + lag] - t[start]
            durations.append(float(dt))
            for name, keep in (("all", np.ones(count, bool)), ("reliable_only", reliable)):
                s = sums[name]
                s["net"] += float(net[keep].sum())
                s["arc"] += float(arc[keep].sum())
                s["duration"] += float(dt * keep.sum())
                s["count"] += int(keep.sum())
        result["lags"][str(lag)] = {
            "expected_intervals": len(owner), "covered_intervals": int((owner >= 0).sum()),
            "actual_durations_seconds": sorted(set(durations)),
            **{name: {"endpoint_speed": ratio(s["net"], s["duration"] * short_side),
                      "arc_speed": ratio(s["arc"], s["duration"] * short_side),
                      "net_to_arc": ratio(s["net"], s["arc"]),
                      "point_intervals": s["count"],
                      "point_interval_fraction": s["count"] / (len(owner) * count)}
               for name, s in sums.items()}}
    sums = {name: dict(dot=0., weight=0., reversed_weight=0., count=0) for name in ("all", "reliable_only")}
    owner = interval_owners(t, windows, 2)
    for start, k in enumerate(owner):
        if k < 0:
            continue
        j = start - windows[k][0]
        velocity = np.diff(paths[f"window_{k}_tracks"][j:j + 3], axis=0) / np.diff(t[start:start + 3])[:, None, None]
        dot = (velocity[0] * velocity[1]).sum(axis=-1)
        weight = np.linalg.norm(velocity[0], axis=-1) * np.linalg.norm(velocity[1], axis=-1)
        reliable = paths[f"window_{k}_reliable_pair"][j:j + 2].all(axis=0)
        for name, keep in (("all", np.ones(count, bool)), ("reliable_only", reliable)):
            s = sums[name]
            s["dot"] += float(dot[keep].sum())
            s["weight"] += float(weight[keep].sum())
            s["reversed_weight"] += float(weight[keep & (dot < 0)].sum())
            s["count"] += int(keep.sum())
    result["adjacent_velocity"] = {
        name: {"weighted_cosine": ratio(s["dot"], s["weight"]),
               "reversal_weight_fraction": ratio(s["reversed_weight"], s["weight"]),
               "point_interval_fraction": s["count"] / (len(owner) * count)}
        for name, s in sums.items()}
    return result


def spatial_predictability(vectors, queries, reliable, short_side, radii=(.125, .25, .5, None)):
    """Leave-center-out affine flow prediction: geometry, NOT object support.

    Predict held-out points from neighbors without their own displacement.
    This intentionally tests whether mere smoothness can distinguish jitter.
    Negative explained energy is retained; zero motion gives null, not 'perfect'.
    """
    v, q, mask = np.asarray(vectors, float), np.asarray(queries, float), np.asarray(reliable, bool)
    if (v.ndim != 3 or v.shape[-1] != 2 or q.shape != v.shape[1:] or mask.shape != v.shape[:2]
            or not np.isfinite(v).all() or not np.isfinite(q).all() or not np.isfinite(short_side) or short_side <= 0):
        raise ValueError("finite T,N,2 vectors, N,2 queries and T,N evidence required")
    results = {}
    for radius in radii:
        if radius is not None and (not np.isfinite(radius) or radius <= 0):
            raise ValueError("positive spatial radius required")
        by_support = {}
        for label, support in (("all", np.ones(mask.shape, bool)), ("reliable_only", mask)):
            error = energy = 0.
            tested = 0
            for i, point in enumerate(q):
                offsets = (q - point) / short_side
                eligible = np.ones(len(q), bool) if radius is None else np.linalg.norm(offsets, axis=1) <= radius
                eligible[i] = False
                design = np.c_[np.ones(len(q)), offsets]
                for k in range(len(v)):
                    keep = eligible & support[k]
                    if not support[k, i] or keep.sum() < 4 or np.linalg.matrix_rank(design[keep]) < 3:
                        continue
                    beta = np.linalg.lstsq(design[keep], v[k, keep], rcond=None)[0]
                    error += float(np.sum((v[k, i] - beta[0]) ** 2))
                    energy += float(np.sum(v[k, i] ** 2))
                    tested += 1
            by_support[label] = {"explained_energy": 1 - error / energy if energy > 0 else None,
                                 "tested_point_pair_fraction": tested / mask.size}
        results["global" if radius is None else str(radius)] = by_support
    return results


def field_frequency(vectors, timestamps):
    """Spectrum of a fixed-grid (Eulerian) velocity field, NOT object frequency.

    All phases including DC; Parseval-correct one-sided energy. Upper third of
    Nyquist is descriptive only, never a filter. Irregular sampling is explicit.
    """
    v, t = np.asarray(vectors, float), np.asarray(timestamps, float)
    if v.ndim != 3 or v.shape[-1] != 2 or len(v) != len(t) - 1 or not np.isfinite(v).all() or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("finite T-1,N,2 vectors and increasing T timestamps required")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-8):
        return {"status": "not_computed_irregular_sampling"}
    velocity = v / dt[:, None, None]
    transformed = np.fft.rfft(velocity, axis=0)
    energy = (abs(transformed) ** 2).sum(axis=(1, 2))
    weights = np.full(len(energy), 2.)
    weights[0] = 1.
    if len(v) % 2 == 0:
        weights[-1] = 1.
    energy *= weights
    freq = np.fft.rfftfreq(len(v), float(dt[0]))
    nyquist = .5 / dt[0]
    return {"status": "descriptive_only", "frequencies_hz": freq.tolist(),
            "energy_fraction": (energy / energy.sum()).tolist() if energy.sum() > 0 else None,
            "upper_third_energy_fraction": ratio(energy[freq >= nyquist * 2 / 3].sum(), energy.sum()),
            "dc_energy_fraction": ratio(energy[0], energy.sum()),
            "mean_square_velocity": float(energy.sum() / len(v) ** 2)}


def dense_magnitude_summary(forward, sampled_vectors):
    """Check whether a near-zero grid estimate merely missed dense motion.

    Pixel/transition units are explicit; no score or correspondence confidence
    is inferred from these numbers. Tiny full-field estimates can still be wrong.
    """
    f, v = np.asarray(forward, float), np.asarray(sampled_vectors, float)
    if (f.ndim != 4 or f.shape[-1] != 2 or v.ndim != 3 or v.shape[-1] != 2
            or len(f) != len(v) or not np.isfinite(f).all() or not np.isfinite(v).all()
            or min(f.shape[:-1]) < 1 or v.shape[1] < 1):
        raise ValueError("finite T,H,W,2 dense and T,N,2 sampled vectors required")
    magnitudes = np.linalg.norm(f, axis=-1).reshape(len(f), -1)
    k = max(1, int(.05 * magnitudes.shape[1]))
    top = np.partition(magnitudes, -k, axis=1)[:, -k:]
    return {"units": "scoring_pixels_per_adjacent_transition",
            "dense_mean": float(magnitudes.mean()),
            "sampled_grid_mean": float(np.linalg.norm(v, axis=-1).mean()),
            "dense_top5_mean": float(top.mean()), "dense_max": float(magnitudes.max()),
            "per_transition_dense_mean": magnitudes.mean(axis=1).tolist(),
            "per_transition_dense_top5_mean": top.mean(axis=1).tolist()}


def figures(output, key, frames, timestamps, paths, vectors, features):
    # Optional visualization dependency, not required to import/test diagnostics.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    indices = np.linspace(0, len(frames) - 1, 16).round().astype(int)
    for ax, i in zip(axes.flat, indices):
        ax.imshow(frames[i]); ax.set_title(f"t={timestamps[i]:.3f}s", fontsize=9); ax.axis("off")
    fig.suptitle(f"{key}: sampled video frames (display only)")
    fig.savefig(output / f"{key}_frames.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), layout="constrained")
    xy = paths["window_0_tracks"]
    reliable = paths["window_0_reliable_pair"]
    grid = int(round(np.sqrt(len(xy[0]))))
    ids = np.array([r * grid + c for r in (grid // 4, grid // 2, 3 * grid // 4)
                    for c in (grid // 4, grid // 2, 3 * grid // 4)])
    axes[0].imshow(frames[0])
    for i in ids:
        for j in range(len(xy) - 1):
            axes[0].plot(xy[j:j + 2, i, 0], xy[j:j + 2, i, 1],
                         color=f"C{i % 10}", linestyle="-" if reliable[j, i] else ":", linewidth=1.2)
        axes[0].scatter(*xy[0, i], s=7, color=f"C{i % 10}")
    axes[0].set_title("Fixed 9 queries, first window\ndotted = unverified correspondence", fontsize=9)
    axes[0].axis("off")
    for i in ids:
        axes[1].plot(timestamps[:len(xy)], xy[:, i, 0] - xy[0, i, 0], color=f"C{i % 10}", linewidth=.9)
    axes[1].set(xlabel="Time (s)", ylabel="Horizontal displacement (scoring px)", title="Advected paths, not ground truth")
    spectrum = features["eulerian_frequency"]
    if spectrum.get("energy_fraction") is not None:
        axes[2].bar(spectrum["frequencies_hz"], spectrum["energy_fraction"], width=.35)
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Fraction of energy", title="Fixed-grid velocity spectrum")
    axes[2].set_ylim(0, 1)
    radii = list(features["spatial_predictability"])
    for label in ("all", "reliable_only"):
        y = [features["spatial_predictability"][r][label]["explained_energy"] for r in radii]
        axes[3].plot(radii, y, ".-", label=label)
    axes[3].set(xlabel="Radius / short side", ylabel="Held-out explained energy", title="Local smoothness is not semantics")
    axes[3].legend(fontsize=8)
    fig.suptitle(key + " — diagnostics only; no Repair score")
    fig.savefig(output / f"{key}_trajectories.png", dpi=135); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--source-scores", required=True)
    parser.add_argument("--cache-provenance", help="earlier replay provenance binding the exact cache hashes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest, source_path = Path(args.manifest), Path(args.source_scores)
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], Path(args.review), manifest, root)
    source_rows = [json.loads(s) for s in source_path.read_text().splitlines()]
    source = {r["candidate_id"]: r for r in source_rows}
    source_identity = json.loads(source_path.with_name("provenance.json").read_text())
    if len(source) != len(source_rows) or set(source) != {r["candidate_id"] for r in rows}:
        raise ValueError("source must uniquely cover the entire reviewed cohort")
    if source_identity.get("repair_variant") != "dense-correspondence" or source_identity.get("review_sha256") != digest(Path(args.review)):
        raise ValueError("bound dense-correspondence source required")
    cache_identity = json.loads(Path(args.cache_provenance).read_text()) if args.cache_provenance else None
    if cache_identity and (cache_identity["source_scores_sha256"] != digest(source_path)
                           or set(cache_identity["cache_sha256"]) != set(source)):
        raise ValueError("earlier cache provenance must bind this exact source run/cohort")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    identity = {"source_model_identity": source_identity, "manifest_sha256": digest(manifest),
                "review_sha256": digest(Path(args.review)), "source_scores_sha256": digest(source_path),
                "source_provenance_sha256": digest(source_path.with_name("provenance.json")),
                "code_files": source_hashes(root), "script_sha256": digest(Path(__file__)),
                "input_sha256": {}, "cache_sha256": {}, "new_model_inference": False,
                "score_or_threshold_optimization": False, "formal_acceptance": "NOT EVALUATED"}
    identity["earlier_cache_provenance_sha256"] = digest(Path(args.cache_provenance)) if cache_identity else None
    runtime = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    records = []
    started = time.monotonic()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key = row["candidate_id"]
            prior = source[key]["repair"]
            video = Path(row["video"])
            if digest(video) != row["sha256"] or source[key]["input_sha256"] != row["sha256"]:
                raise ValueError("source video identity changed")
            config = LocalTrajectoryConfig(**prior["config"])
            frames, t, _ = decode_video(video, config)
            cache_path = source_path.parent / "evidence" / f"{key}.npz"
            if cache_identity and digest(cache_path) != cache_identity["cache_sha256"][key]:
                raise ValueError("cache differs from earlier immutable replay")
            with np.load(cache_path, allow_pickle=False) as z:
                if not np.array_equal(z["shape"], frames.shape) or not np.array_equal(z["timestamps"], t):
                    raise ValueError("cache geometry/timeline mismatch")
                vectors, queries = z["flow_vectors"], z["queries"]
                raw = np.linalg.norm(vectors, axis=-1).sum() / ((t[-1] - t[0]) * len(queries) * min(frames.shape[1:3]))
                if not np.isclose(raw, prior["ablations"]["raw_tracks"], rtol=1e-6, atol=1e-9):
                    raise ValueError("source raw-flow replay mismatch")
                mask = (z["inside_pair"] & z["visible_pair"] & (z["cycle_error"] <= config.cycle_error_max)
                        & (z["moving_error"] <= config.match_error_max))
                paths = local_dense_paths(frames, t, z["forward"], z["backward"], queries, config)
                dense_summary = dense_magnitude_summary(z["forward"], vectors)
            features = {"temporal": temporal_diagnostics(paths, t, min(frames.shape[1:3])),
                        "spatial_predictability": spatial_predictability(vectors, queries, mask, min(frames.shape[1:3])),
                        "eulerian_frequency": field_frequency(vectors, t), "dense_magnitude": dense_summary}
            record = {"candidate_id": key, "base_id": row["base_id"], "prompt_id": row["prompt_id"],
                      "family": row["family"], "seed": row["seed"], "status": "diagnostic_only", "score": None,
                      "original_reliability_status": prior["status"], "original_coverage": prior["coverage"],
                      "origin_boolean": source[key]["origin"]["score"], "frames": len(t),
                      "span_seconds": float(t[-1] - t[0]), **features}
            records.append(record)
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            identity["input_sha256"][key] = row["sha256"]
            identity["cache_sha256"][key] = digest(cache_path)
            if args.figures and row["family"] != "encoding_control":
                figures(output, key, frames, t, paths, vectors, features)
            print(json.dumps({"candidate_id": key, "status": "diagnostic_only"}), flush=True)
    table = []
    for r in records:
        f = {k: r[k] for k in ("candidate_id", "base_id", "family", "seed", "original_reliability_status")}
        f.update(arc_speed=r["temporal"]["lags"]["1"]["all"]["arc_speed"],
                 net_arc_lag2=r["temporal"]["lags"]["2"]["all"]["net_to_arc"],
                 net_arc_lag4=r["temporal"]["lags"]["4"]["all"]["net_to_arc"],
                 direction_cosine=r["temporal"]["adjacent_velocity"]["all"]["weighted_cosine"],
                 upper_band=r["eulerian_frequency"].get("upper_third_energy_fraction"),
                 local_affine=r["spatial_predictability"]["0.125"]["all"]["explained_energy"],
                 global_affine=r["spatial_predictability"]["global"]["all"]["explained_energy"])
        table.append(f)
    with (output / "cases.csv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    runtime.update(status="finished", completed=len(records), elapsed_seconds=time.monotonic() - started,
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
