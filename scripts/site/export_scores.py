#!/usr/bin/env python3
"""Export measured demonstration scores, retaining the initial Scene control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    mapping = {"scene": "scene", "subject": "subject_consistency", "spatial": "spatial_relationship"}
    cases, receipts = {}, []
    for key, dimension in mapping.items():
        run = "scene-synonym" if key == "scene" else "eval"
        cases[key] = {"dimension": dimension}
        for backend in ("origin", "repair"):
            path = args.evidence / run / dimension / f"{backend}.json"
            payload = read(path)
            rows = {row["id"]: row for row in payload["rows"]}
            assert set(rows) == {f"{key}:base", f"{key}:cf"}, path
            cases[key][backend] = [rows[f"{key}:{role}"]["score"] for role in ("base", "cf")]
            receipts.append({
                "result": str(path.relative_to(args.evidence)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fresh_inference": payload["execution"]["fresh_inference"],
                "rows": [{name: row.get(name) for name in ("id", "prompt", "video_sha256", "score", "status", "error")}
                         for row in rows.values()],
            })
    cases["scene"]["observation"] = "VBench follows the caption's exact word; Ours recognizes the equivalent scene description."
    cases["scene"]["note"] = "The street/road pair was selected after inspecting captions. The initial ocean/sea control is retained in the downloadable receipt."
    cases["subject"]["observation"] = "Both scores change slightly; Ours shows the smaller response to this background-only intervention."
    cases["subject"]["note"] = "The car is independently localized in each video. A small score change is not exact invariance."
    cases["spatial"]["observation"] = "VBench retains its maximum score after mirroring; Ours responds to the reversed relation."
    cases["spatial"]["note"] = "The query stays fixed. This tests directional sensitivity, not a preference for one camera orientation."
    controls = {}
    for backend in ("origin", "repair"):
        path = args.evidence / "eval" / "scene" / f"{backend}.json"
        controls[backend] = [{name: row[name] for name in ("id", "prompt", "video_sha256", "score", "status")}
                             for row in read(path)["rows"]]
    initial_summary = read(args.evidence / "eval" / "summary.json")
    scene_summary = read(args.evidence / "scene-synonym" / "summary.json")
    assert initial_summary["complete"] and scene_summary["complete"]
    identity = read(args.evidence / "identity.json")
    payload = {
        "schema": "vbench-site-scores/1", "cases": cases,
        "initial_scene_control": controls,
        "provenance": {
            **identity,
            "summary": "Fresh H100 inference on the displayed LTX clip and its constructed variants; all 16 per-query/backend records succeeded across the main run and Scene follow-up. The street/road illustration was chosen after inspecting the initial captions; the ocean/sea control is retained. These curated examples do not estimate failure prevalence or replace the frozen paper experiment.",
            "initial_run_seconds": initial_summary["elapsed_seconds"],
            "scene_followup_seconds": scene_summary["elapsed_seconds"],
            "initial_input_records": 6, "scene_followup_input_records": 2,
            "unique_source_videos": 1, "media_seconds": 121 / 24,
            "master_frame_shape": [1088, 1920, 3], "master_frames": 121, "master_fps": 24,
            "plan_sha256": hashlib.sha256((args.evidence / "eval" / "plan.json").read_bytes()).hexdigest(),
            "source_result_files": receipts,
        },
    }
    # Public receipts contain hashes and measured records, not private machine paths.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
