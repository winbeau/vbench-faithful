"""Inspect native-coordinate tracks, including every regional alternative.

The optional source-key case is a disclosed post-hoc diagnostic, never a query
selection or scoring input. Point visibility is the model output, not truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import resolve_media
from .score_static_jitter import source_hashes
from .static_jitter import digest


def point_statistics(xy, visible, source_frame):
    d = np.asarray(xy)[source_frame + 1] - np.asarray(xy)[source_frame]
    magnitude = np.linalg.norm(d, axis=-1)
    mask = np.asarray(visible)[source_frame] & np.asarray(visible)[source_frame + 1]
    return {"queries": len(d), "all_query_step_mean_pixels": float(magnitude.mean()) if len(d) else None,
            "all_query_step_p95_pixels": float(np.quantile(magnitude, .95)) if len(d) else None,
            "model_visible_pair_fraction": float(mask.mean()) if len(d) else None,
            "visible_only_step_mean_pixels": float(magnitude[mask].mean()) if mask.any() else None,
            "scope": "single adjacent query-frame pair; adaptive feature positions, not a video score"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "output", "case-candidate"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--source-key", type=int, required=True)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    source, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise FileExistsError("fresh audit required")
    identity = json.loads((source / "provenance.json").read_text())
    runtime = json.loads((source / "runtime.json").read_text())
    if runtime["status"] != "finished" or runtime["completed"] != runtime["expected"] or runtime["failed"]:
        raise ValueError("complete nonfailed pilot required")
    request = identity["request"]
    expected = {r["candidate_id"]: r for r in request["videos"]}
    rows = [json.loads(s) for s in (source / "diagnostics.jsonl").read_text().splitlines()]
    if len(rows) != len(expected) or {r["candidate_id"] for r in rows} != set(expected):
        raise ValueError("missing or duplicate pilot inputs")
    query_frame = request["query_frame"]
    if args.case_candidate not in expected:
        raise ValueError("posthoc case outside this pilot")
    output.mkdir(parents=True)
    stats, anchors, originals, controls = [], [], {}, []
    for row in rows:
        key, info = row["candidate_id"], expected[row["candidate_id"]]
        path = source / "evidence" / f"{key}.npz"
        if digest(path) != identity["cache_sha256"][key] or row["score"] is not None:
            raise ValueError("changed evidence or score contract")
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        if not np.array_equal(arrays["queries_txy"], info["queries_txy"]):
            raise ValueError("queries differ from automatic preparation")
        if row["family"] == "original":
            originals[row["base_id"]] = arrays
        elif row["family"] == "encoding_control":
            base = originals.get(row["base_id"])
            controls.append({"candidate_id": key, "all_arrays_exact": base is not None
                             and set(base) == set(arrays) and all(np.array_equal(v, base[k]) for k, v in arrays.items())})
        anchor = None
        if key == args.case_candidate:
            anchor = next((i for i, group in enumerate(info["key_groups"]) if args.source_key in group), None)
            if anchor is None:
                raise ValueError("posthoc source key not part of automatic query set")
        record = {k: row[k] for k in ("candidate_id", "family", "queries", "status")}
        alternatives = []
        if "regions" not in row:
            alternatives.append((None, np.arange(row["queries"]), arrays["tracks"], arrays["visible"], None))
        else:
            record["insufficient_regions"] = [r["region"] for r in row["regions"] if r["status"] != "diagnostic_only"]
            for region in row["regions"]:
                if region["status"] == "diagnostic_only":
                    i = region["region"]
                    alternatives.append((i, arrays[f"region_{i}_query_indices"], arrays[f"region_{i}_tracks_native"],
                                         arrays[f"region_{i}_model_visible"], arrays[f"region_{i}_inside_observed_window"]))
        record["alternatives"] = []
        for region, ids, xy, visible, in_view in alternatives:
            record["alternatives"].append({"region": region, **point_statistics(xy, visible, query_frame)})
            if anchor is not None and anchor in ids:
                index = np.flatnonzero(ids == anchor).item()
                anchors.append({"region": region, "source_key": args.source_key, "query_index": anchor,
                                "query_frame_step_pixels": (xy[query_frame + 1, index] - xy[query_frame, index]).tolist(),
                                "native_xy": xy[:, index].tolist(), "model_visible": visible[:, index].tolist(),
                                "inside_observed_window": in_view[:, index].tolist() if in_view is not None else None})
        stats.append(record)
    if not anchors:
        raise ValueError("case has no retained track alternatives")
    info = expected[args.case_candidate]
    video = resolve_media({"video": info["video"], "sha256": info["sha256"]}, args.video_root)
    frames, times, _ = decode_video(video, TrajectoryConfig(sample_fps=8, max_side=512))
    if not np.array_equal(times, info["timestamps"]):
        raise ValueError("display must use original native frame times")
    cell, head = 256, 36
    sheet = np.full((len(anchors) * (cell + head), 4 * cell, 3), 255, np.uint8)
    shown = list(range(query_frame, min(query_frame + 4, len(frames))))
    for i, case in enumerate(anchors):
        for col, t in enumerate(shown):
            x, y = col * cell, i * (cell + head)
            title = f'region {case["region"]} t={t} visible={case["model_visible"][t]}'
            cv2.putText(sheet, title, (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
            view = cv2.resize(frames[t], (cell, cell), interpolation=cv2.INTER_NEAREST)
            xy = np.asarray(case["native_xy"][t]) * cell / np.array(frames.shape[2:0:-1])
            if np.isfinite(xy).all() and (xy >= 0).all() and (xy < cell).all():
                cv2.circle(view, tuple(np.rint(xy).astype(int)), 6, (255, 0, 255), 2, cv2.LINE_AA)
            sheet[y + head:y + head + cell, x:x + cell] = view
    image = output / "posthoc_anchor.png"
    if not cv2.imwrite(str(image), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("display write failed")
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
              "source_provenance_sha256": digest(source / "provenance.json"),
              "source_diagnostics_sha256": digest(source / "diagnostics.jsonl"),
              "script_sha256": digest(Path(__file__)), "code_files": source_hashes(Path(__file__).resolve().parents[2]),
              "runtime": runtime, "videos": stats, "encoding_controls": controls,
              "posthoc_case_candidate": args.case_candidate, "posthoc_source_key": args.source_key,
              "anchors": anchors, "figure_sha256": digest(image),
              "warning": "query-frame positions/visibility are forced by upstream; this is not certified physical tracking or a score"}
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"controls": controls, "anchors": [{k: a[k] for k in ("region", "query_frame_step_pixels")} for a in anchors]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
