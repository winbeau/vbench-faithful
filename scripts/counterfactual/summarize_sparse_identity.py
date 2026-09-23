"""Verify complete sparse-identity diagnostics and audit the prompted DEV case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.native_region_motion import rank_with_common_sift
from dynamic_degree.sparse_structure import source_keys_in_mask
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from vbench_audit_models.sam_regions import unpack_masks
from .audit_prompted_regions import selected_prediction_indices
from .probe_native_region_motion import load_masks, resolve_media
from .score_static_jitter import source_hashes
from .static_jitter import digest


def method_counts(data):
    result = {"pairs": len(data["pairs"]), "matches": 0, "identity_supported_matches": 0,
              "contradicted_matches": 0, "unknown_matches": 0, "region_pairs": 0}
    for kind in ("raw", "temporal"):
        result[kind] = {"ranked": 0, "unranked": 0, "witnesses_0": 0, "witnesses_1": 0, "witnesses_ge2": 0}
    for pair in data["pairs"]:
        result["matches"] += len(pair["matches"])
        if len(pair["matches"]) != len(pair["identity_support"]):
            raise ValueError("every match needs an explicit identity record")
        for match, support in zip(pair["matches"], pair["identity_support"]):
            if any(match[k] != support[k] for k in ("source_key", "target_key")):
                raise ValueError("temporal support must refer to the exact paired feature IDs")
        for match in pair["identity_support"]:
            if match["supported_without_conflict"]:
                result["identity_supported_matches"] += 1
            elif match["conflicting_frames"]:
                result["contradicted_matches"] += 1
            else:
                result["unknown_matches"] += 1
        result["region_pairs"] += len(pair["regions"])
        for region in pair["regions"]:
            for kind in ("raw", "temporal"):
                row = region[kind + "_ranking"]
                if row["score"] is not None or row["reliability"] != "NOT CERTIFIED":
                    raise ValueError("diagnostic cannot be relabeled certified motion")
                result[kind]["unranked" if row["best_compatible_hypothesis"] is None else "ranked"] += 1
                count = row["unique_witness_locations"]
                result[kind][f"witnesses_{count}" if count < 2 else "witnesses_ge2"] += 1
    return result


def prompted_case(record, probe, regions, audit, video_roots, output):
    p_id = json.loads((probe / "provenance.json").read_text())
    req = p_id["request"]
    prior = json.loads((audit / "diagnostic.json").read_text())
    if (prior["proposal_source"] != "rgb-only" or prior["probe_provenance_sha256"] != digest(probe / "provenance.json")
            or digest(probe / "masks.npz") != p_id["masks_sha256"]
            or digest(regions / "provenance.json") != req["region_provenance_sha256"]
            or req["candidate_id"] != record["candidate_id"]):
        raise ValueError("same-video RGB-only prompted case binding required")
    frame_id, target_id, source_id = req["start"], req["target"], req["source_region"]
    video = resolve_media({"video": req["video"], "sha256": req["video_sha256"]}, video_roots)
    frames, times, _ = decode_video(video, TrajectoryConfig(sample_fps=8, max_side=512))
    for kind, index in (("source", frame_id), ("target", target_id)):
        if hashlib.sha256(frames[index].tobytes()).hexdigest() != req[kind + "_frame_sha256"]:
            raise ValueError("prompted case native pixels differ")
    masks = load_masks(regions / "evidence" / f'{req["candidate_id"]}.npz', req["cache_sha256"], frames, times)
    with np.load(probe / "masks.npz", allow_pickle=False) as z:
        recovered = unpack_masks(z["masks_packed"], z["image_shape"])
    selected = selected_prediction_indices(p_id, "rgb-only")
    if selected != prior["selected_prediction_indices"]:
        raise ValueError("prompt-source ablation changed")
    targets = np.concatenate((masks[target_id], recovered[selected]))
    results = {}
    cell, header = 512, 54
    sheet = np.full(((cell + header) * len(record["methods"]), 2 * cell, 3), 255, np.uint8)
    for row_index, (method, data) in enumerate(record["methods"].items()):
        pair = next(p for p in data["pairs"] if p["start"] == frame_id and p["start"] + p["lag"] == target_id)
        support = {s["source_key"]: s for s in pair["identity_support"]}
        keep = [m for m in pair["matches"] if support[m["source_key"]]["supported_without_conflict"]]
        base = prior["augmented"]
        source_mask = masks[frame_id][source_id]
        raw = rank_with_common_sift({**base, "sift_match_source_keys": source_keys_in_mask(pair["matches"], source_mask)},
                                    targets, pair["matches"])
        closed = rank_with_common_sift({**base, "sift_match_source_keys": source_keys_in_mask(keep, source_mask)}, targets, keep)
        witnesses = [m for m in pair["matches"] if m["source_key"] in raw["common_source_keys"]]
        results[method] = {"raw": raw, "temporal": closed,
                           "witnesses": [{**m, "identity": support[m["source_key"]]} for m in witnesses]}
        y = row_index * (cell + header)
        for col, (index, field) in enumerate(((frame_id, "source_xy"), (target_id, "target_xy"))):
            x = col * cell
            cv2.putText(sheet, f"{method} frame {index}: all common witnesses", (x + 8, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(sheet, "labels are key IDs, not verified physical identities", (x + 8, y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1, cv2.LINE_AA)
            view = cv2.resize(frames[index], (cell, cell), interpolation=cv2.INTER_NEAREST)
            for m in witnesses:
                xy = np.asarray(m[field]) * cell / np.array([frames.shape[2], frames.shape[1]])
                point = tuple(np.rint(xy).astype(int))
                cv2.circle(view, point, 7, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(view, str(m["source_key"]), (point[0] + 8, point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1, cv2.LINE_AA)
            sheet[y + header:y + header + cell, x:x + cell] = view
    image = output / "prompted_case_witnesses.png"
    if not cv2.imwrite(str(image), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("case display failed")
    result = {"status": "diagnostic_only", "score": None, "candidate_id": record["candidate_id"],
              "start": frame_id, "target": target_id, "source_region": source_id,
              "hypothesis_displacements": [h["displacement_pixels"] for h in prior["augmented"]["hypotheses"]],
              "methods": results, "probe_provenance_sha256": digest(probe / "provenance.json"),
              "rgb_audit_sha256": digest(audit / "diagnostic.json"), "figure_sha256": digest(image),
              "scope": "posthoc single DEV frame pair; all generated RGB-only target masks; no manual point selection"}
    (output / "prompted_case.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return digest(output / "prompted_case.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--output", required=True)
    for name in ("prompted-run", "region-run", "rgb-audit"):
        p.add_argument("--" + name)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    if any((args.prompted_run, args.region_run, args.rgb_audit)) and not all((args.prompted_run, args.region_run, args.rgb_audit)):
        p.error("prompted-run, region-run and rgb-audit must be supplied together")
    source, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise FileExistsError("fresh diagnostic summary required")
    identity = json.loads((source / "provenance.json").read_text())
    runtime = json.loads((source / "runtime.json").read_text())
    if (runtime["status"] != "finished" or runtime["completed"] != runtime["expected"]
            or digest(source / "diagnostics.jsonl") != identity["diagnostics_sha256"]):
        raise ValueError("complete hash-bound source required, no running/partial summary")
    candidate = None
    if args.prompted_run:
        request = json.loads((Path(args.prompted_run) / "provenance.json").read_text())["request"]
        candidate = request["candidate_id"]
        if (identity["input_sha256"].get(candidate) != request["video_sha256"]
                or identity["region_provenance_sha256"] != request["region_provenance_sha256"]):
            raise ValueError("prompted and temporal runs must use the same video and original SAM masks")
    summary, originals, controls, observed, case = [], {}, [], [], None
    output.mkdir(parents=True)
    with (source / "diagnostics.jsonl").open() as f:
        for row in map(json.loads, f):
            key = row["candidate_id"]
            if key in observed or key not in identity["input_sha256"] or row["score"] is not None:
                raise ValueError("cohort or null-score contract violated")
            observed.append(key)
            if row["status"] == "diagnostic_only":
                if set(row["methods"]) != set(identity["methods"]):
                    raise ValueError("successful record missing a planned witness method")
                count = len(row["sampling"]["timestamps"])
                expected = [(lag, start) for lag in identity["lags"] for start in range(count - lag)]
                for method in row["methods"].values():
                    if [(p["lag"], p["start"]) for p in method["pairs"]] != expected:
                        raise ValueError("successful record missing a native frame phase")
            entry = {k: row[k] for k in ("candidate_id", "base_id", "family", "seed", "status")}
            entry["methods"] = {method: method_counts(data) for method, data in row["methods"].items()}
            summary.append(entry)
            signature = hashlib.sha256(json.dumps(row["methods"], sort_keys=True, allow_nan=False).encode()).hexdigest()
            if row["family"] == "original":
                originals[row["base_id"]] = signature
            elif row["family"] == "encoding_control":
                controls.append({"candidate_id": key, "base_id": row["base_id"], "signature": signature})
            if key == candidate:
                case = prompted_case(row, Path(args.prompted_run), Path(args.region_run), Path(args.rgb_audit), args.video_root, output)
    if set(observed) != set(identity["input_sha256"]):
        raise ValueError("missing cohort records")
    if sum(m["pairs"] for r in summary for m in r["methods"].values()) != runtime["method_pairs_completed"]:
        raise ValueError("recorded phase count and runtime disagree")
    for control in controls:
        control["equal_to_original"] = control.pop("signature") == originals.get(control["base_id"])
    if args.prompted_run and case is None:
        raise ValueError("requested case not present")
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
              "source_provenance_sha256": digest(source / "provenance.json"),
              "source_diagnostics_sha256": identity["diagnostics_sha256"],
              "script_sha256": digest(Path(__file__)), "runtime": runtime, "videos": summary,
              "code_files": source_hashes(Path(__file__).resolve().parents[2]),
              "encoding_controls": controls, "prompted_case_sha256": case,
              "warning": "counts include nested regions and controls, not independent motion truth or valid scores"}
    (output / "counts.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"videos": len(summary), "controls": controls, "case_sha256": case}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
