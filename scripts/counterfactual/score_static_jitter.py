"""Resumable, provenance-checked scoring of a frozen static-jitter ledger.

The manifest is used by the runner only. The repair receives a video path,
its frozen config, and local model assets, never construction metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .static_jitter import digest


def source_hashes(root: Path) -> dict[str, str]:
    paths = []
    for pattern in ("metrics/dynamic-degree/src/**/*.py", "packages/audit-models/src/**/*.py",
                    "packages/audit-core/src/**/*.py", "scripts/counterfactual/*static_jitter.py",
                    "scripts/counterfactual/review_selection.py"):
        paths.extend(root.glob(pattern))
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_candidates(rows, *, stress_test=False):
    """An explicit DEV stress diagnostic never changes construction qualification."""
    if stress_test and any(r.get("split") != "dev" or
                           r.get("protocol") != "official-video-local-texture-jitter-v1" for r in rows):
        raise ValueError("stress scoring is restricted to development local texture jitter")
    selected = []
    for row in rows:
        if row["status"] == "qualified":
            selected.append(row)
        elif stress_test and row["status"] == "rejected":
            if not (row.get("sha256") and row.get("pixel_exact_to_intended")
                    and row.get("native_timeline_preserved")
                    and row.get("sampling_coordinates_in_bounds")
                    and row.get("minimum_warp_jacobian", 0) > 0
                    and row.get("intensity_noise_added") is False
                    and row.get("family") == "local_texture_alternating"):
                raise ValueError("stress scoring still requires verified media, no folds, and no additive noise")
            selected.append(row)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", help="relocated byte-identical videos, matched by filename and SHA256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--tracker-weight", required=True)
    parser.add_argument("--raft-weight", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--backend", choices=["both", "repair", "origin"], default="both")
    parser.add_argument("--repair-variant", choices=["trajectory", "local-trajectory", "dense-correspondence"], default="trajectory")
    parser.add_argument("--repair-flow-backend", choices=["official", "torchvision"], default="official")
    parser.add_argument("--repair-flow-weight", help="dense Repair-only local checkpoint; never replaces Origin's --raft-weight")
    parser.add_argument("--repair-flow-scale", type=float, default=1., help="torchvision-only internal spatial inference scale, in [1,4]")
    parser.add_argument("--review", help="hash-bound development user review; selection only, never supplied to the metric")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="development only: clean and strong seed 1701")
    parser.add_argument("--stress-test", action="store_true",
                        help="DEV ONLY: also score verified, nonfolding local warps rejected by quality gates; never requalify them")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        parser.error("invalid shard")
    if args.review and (args.smoke or args.stress_test):
        parser.error("review selects its exact cohort; cannot combine with smoke or stress-test")
    if (args.repair_flow_backend != "official" or args.repair_flow_weight) and (
            args.repair_variant != "dense-correspondence" or args.backend == "origin"):
        parser.error("Repair flow options require dense-correspondence and a Repair backend")
    if args.repair_flow_backend == "torchvision" and not args.repair_flow_weight:
        parser.error("torchvision requires its own explicit --repair-flow-weight")
    if not 1 <= args.repair_flow_scale <= 4 or (args.repair_flow_scale != 1 and args.repair_flow_backend != "torchvision"):
        parser.error("inference scale in [1,4] is available only for the torchvision Repair flow")
    import torch
    from dynamic_degree.trajectory import TrajectoryConfig, TrajectoryEvaluator
    if args.repair_variant == "local-trajectory":
        from dynamic_degree.local_trajectory import LocalTrajectoryConfig as TrajectoryConfig, LocalTrajectoryEvaluator as TrajectoryEvaluator
    if args.repair_variant == "dense-correspondence":
        from dynamic_degree.dense_correspondence import DenseCorrespondenceConfig as TrajectoryConfig, DenseCorrespondenceEvaluator
    from dynamic_degree.backends.vbench import OfficialDynamicEvaluator, official_result_payload

    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = TrajectoryConfig.read(Path(args.config))
    all_rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    if args.smoke and any(r["split"] != "dev" for r in all_rows):
        raise ValueError("smoke selection is allowed only on development data")
    root = Path(__file__).resolve().parents[2]
    if args.review:
        from .review_selection import select_reviewed_candidates
        rows = select_reviewed_candidates(all_rows, Path(args.review), Path(args.manifest), root)
    else:
        rows = select_candidates(all_rows, stress_test=args.stress_test)
    if args.smoke:
        rows = [r for r in rows if r["family"] == "clean" or (r["amplitude"] == 40 and r["seed"] == 1701)]
    rows = rows[args.shard::args.shards]
    out = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[2]
    uses_tracker = args.backend in {"repair", "both"} and args.repair_variant != "dense-correspondence"
    uses_dense = args.backend in {"repair", "both"} and args.repair_variant == "dense-correspondence"
    uses_raft = args.backend in {"origin", "both"} or (uses_dense and args.repair_flow_backend == "official")
    repair_flow_weight = Path(args.repair_flow_weight or args.raft_weight)
    repair_flow_identity = None
    if uses_dense:
        repair_flow_identity = {"backend": "official_raft", "updates": 20}
        if args.repair_flow_backend == "torchvision":
            from vbench_audit_models.torchvision_flow import source_identity

            repair_flow_identity = source_identity(args.repair_flow_scale)
        repair_flow_identity["weight_sha256"] = digest(repair_flow_weight)
    provenance = {
        "manifest_sha256": digest(Path(args.manifest)), "config": asdict(config),
        "config_sha256": digest(Path(args.config)), "code_files": source_hashes(root),
        "tracker_code": ({str(p.relative_to(args.tracker_root)): digest(p) for p in sorted(Path(args.tracker_root).glob("cotracker/**/*.py"))}
                         if uses_tracker else None),
        "tracker_weight_sha256": digest(Path(args.tracker_weight)) if uses_tracker else None,
        "raft_weight_sha256": digest(Path(args.raft_weight)) if uses_raft else None,
        "upstream_sha": (command_output(["git", "-C", args.upstream, "rev-parse", "HEAD"])
                         if uses_raft else None),
        "backend": args.backend, "shard": args.shard, "shards": args.shards, "smoke": args.smoke,
        "stress_test_includes_quality_rejections": args.stress_test,
        "repair_variant": args.repair_variant,
        "repair_flow": repair_flow_identity,
        "review_sha256": digest(Path(args.review)) if args.review else None,
        "video_sha256": {r["candidate_id"]: r["sha256"] for r in rows},
    }
    if out.exists():
        if not args.resume:
            raise FileExistsError("output exists: use --resume only with identical provenance")
        old = json.loads((out / "provenance.json").read_text())
        if old != provenance:
            raise ValueError("resume rejected: input/model/config/code identity changed")
    else:
        out.mkdir(parents=True)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    info = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(), "python": platform.python_version(), "torch": torch.__version__,
            "python_executable": sys.executable,
            "cuda": torch.version.cuda, "device": str(args.device),
            "gpu": torch.cuda.get_device_name(args.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "arguments": vars(args), "expected": len(rows)}
    (out / "runtime.json").write_text(json.dumps(info, indent=2))
    record_path = out / "scores.jsonl"
    completed = set()
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            item = json.loads(line)
            if item["candidate_id"] in completed:
                raise ValueError("duplicate candidate in score ledger")
            completed.add(item["candidate_id"])
    pending = [r for r in rows if r["candidate_id"] not in completed]
    repair = None
    if args.backend in {"repair", "both"} and pending:
        if args.repair_variant == "dense-correspondence":
            flow_model = None
            if args.repair_flow_backend == "torchvision":
                from vbench_audit_models.torchvision_flow import TorchvisionRaftFlowModel

                flow_model = TorchvisionRaftFlowModel(args.device, repair_flow_weight, inference_scale=args.repair_flow_scale)
            repair = DenseCorrespondenceEvaluator(config, repair_flow_weight, Path(args.upstream), args.device,
                                                   flow_model=flow_model)
        else:
            repair = TrajectoryEvaluator(config, Path(args.tracker_root), Path(args.tracker_weight), args.device)
    origin = (OfficialDynamicEvaluator(torch.device(args.device), Path(args.raft_weight), Path(args.upstream))
              if args.backend in {"origin", "both"} and pending else None)
    started = time.monotonic()
    failures = 0
    with record_path.open("a") as handle:
        for ordinal, row in enumerate(pending):
            video = Path(args.video_root) / Path(row["video"]).name if args.video_root else Path(row["video"])
            result = {"candidate_id": row["candidate_id"], "input_sha256": row["sha256"],
                      "construction_status": row["status"], "stress_test": args.stress_test,
                      "user_review_selected": bool(args.review)}
            video_started = time.monotonic()
            if digest(video) != row["sha256"]:
                raise ValueError(f"input identity changed: {video}")
            for backend, evaluator in (("repair", repair), ("origin", origin)):
                if evaluator is None:
                    continue
                try:
                    if backend == "repair":
                        value = evaluator.evaluate_video(video, evidence_path=out / "evidence" / f"{row['candidate_id']}.npz")
                    else:
                        official = evaluator.evaluate_video(video)
                        value = {"status": "succeeded", "score": float(official.official_video_boolean),
                                 "official_video_boolean": official.official_video_boolean,
                                 "diagnostics": official_result_payload(official)}
                        # Verify the actual reference infer path on the first item of every run.
                        if ordinal == 0:
                            reference = bool(evaluator.dynamic.infer(str(video)))
                            value["reference_infer_boolean"] = reference
                            if reference != official.official_video_boolean:
                                raise AssertionError("official adapter/reference parity failed")
                    result[backend] = value
                except Exception as exc:
                    result[backend] = {"status": "failed", "score": None, "error": f"{type(exc).__name__}: {exc}"}
                failures += result[backend]["status"] != "succeeded"
            result["seconds"] = time.monotonic() - video_started
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(json.dumps({"candidate_id": row["candidate_id"], "done": ordinal + 1, "pending": len(pending),
                "seconds": result["seconds"], **{b: result[b]["score"] for b in ("repair", "origin") if b in result}}), flush=True)
    info.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                elapsed_seconds=time.monotonic() - started, status="finished", completed=len(completed) + len(pending),
                failed_or_insufficient=failures)
    (out / "runtime.json").write_text(json.dumps(info, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
