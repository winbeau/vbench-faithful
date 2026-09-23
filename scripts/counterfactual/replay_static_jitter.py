"""Development-only descriptor/aggregation comparison on cached model tracks.

No test manifest is accepted. New feature parameters recompute evidence from
the original decoded video, never from construction labels or clean pairs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from dynamic_degree.trajectory import TrajectoryConfig, correspondence_evidence, decode_video, score_evidence
from .analyze_static_jitter import load_scores
from .score_static_jitter import source_hashes
from .static_jitter import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--evidence-roots", nargs="+", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    if not rows or {r["split"] for r in rows} != {"dev"}:
        raise ValueError("cached development comparisons reject test data")
    scores = load_scores(args.scores)
    config = TrajectoryConfig.read(Path(args.config))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"manifest_sha256": digest(Path(args.manifest)), "config": asdict(config),
                  "config_sha256": digest(Path(args.config)), "model_predictions_reused": True,
                  "scores_sha256": {str(p): digest(Path(p)) for p in args.scores},
                  "code_files": source_hashes(Path(__file__).resolve().parents[2]),
                  "evidence_sha256": {}}
    with (output / "scores.jsonl").open("x") as handle:
        for row in rows:
            original = scores.get(row["candidate_id"])
            if original is None:
                continue  # This is explicitly a partial replay if source scoring is partial.
            old_config = original.get("repair", {}).get("config", {})
            if old_config.get("tracker_blur_sigma", 0.0) != config.tracker_blur_sigma:
                raise ValueError("changed model preprocessing requires new model inference, not cached tracks")
            paths = [Path(p) / f"{row['candidate_id']}.npz" for p in args.evidence_roots]
            paths = [p for p in paths if p.is_file()]
            if len(paths) != 1:
                raise ValueError(f"expected one model evidence cache for {row['candidate_id']}")
            path = paths[0]
            video = Path(args.video_root) / Path(row["video"]).name
            if digest(video) != row["sha256"] or original["input_sha256"] != row["sha256"]:
                raise ValueError("replay video identity mismatch")
            frames, timestamps, sampling = decode_video(video, config)
            with np.load(path, allow_pickle=False) as cached:
                if not np.array_equal(cached["shape"], frames.shape) or not np.allclose(cached["timestamps"], timestamps):
                    raise ValueError("cannot reuse tracks with a different sampling geometry")
                if len(cached["queries"]) != config.grid_size ** 2:
                    raise ValueError("cannot reuse tracks with a different query grid")
                tracks = {k: cached[k] for k in ("tracks", "visible", "reverse_tracks", "reverse_visible")}
            evidence = correspondence_evidence(frames, tracks, config)
            value = score_evidence(evidence, timestamps, *frames.shape[1:3], config)
            result = {**original, "candidate_id": row["candidate_id"], "repair": {
                **value, "video": str(video), "backend": "audit", "config": asdict(config),
                "sampling": sampling, "variant": "trajectory_development_replay", "model_cache": str(path)}}
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            provenance["evidence_sha256"][str(path)] = digest(path)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"replayed": len(provenance["evidence_sha256"])}))


if __name__ == "__main__":
    main()
