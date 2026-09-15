"""Score counterfactual clips with Official and Repair backends.

One worker handles one (dimension, backend) pair on one logical GPU (`cuda:0`
inside a `CUDA_VISIBLE_DEVICES` mask).  It reads the counterfactual manifest,
reconstructs each clip's metadata from the frozen official annotations, scores
every clip, and appends one JSON line per (derived_id, backend) to a shard
results file.  Runs are resumable: a clip whose score already exists is skipped.

The backends are the metric packages' own `backends/vbench.py` (Official) and
`backends/audit.py` (Repair).  Weights are located through the same environment
variables the metric packages already honour; see `score_orchestrate.sh` for the
mapping on the H100 box.  Nothing here downloads a weight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]

DIMENSIONS = (
    "dynamics_degree",
    "subject_consistency",
    "human_action",
    "spatial_relationship",
    "scene",
    "multiplt_object",
    "motion_smoothness",
)

ANNOTATION_FILES = {
    "subject_consistency": "Subject_consistency.json",
    "human_action": "Human_Action.json",
    "spatial_relationship": "Spatial_Relationship.json",
    "scene": "Scene.json",
    "multiplt_object": "Multiplt_Object.json",
}


def _syspath() -> None:
    for rel in (
        "packages/audit-core/src",
        "packages/audit-models/src",
        "metrics/dynamic-degree/src",
        "metrics/subject-consistency/src",
        "metrics/human-action/src",
        "metrics/spatial-relationship/src",
        "metrics/scene/src",
        "metrics/multiple-objects/src",
        "metrics/motion-smoothness/src",
    ):
        path = ROOT / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    upstream = os.environ.get("VBENCH_AUDIT_UPSTREAM")
    if upstream and upstream not in sys.path:
        sys.path.insert(0, upstream)


def load_annotations(annotations_root: Path, dimension: str) -> dict[str, dict[str, Any]]:
    """Map prompt_en -> official annotation entry for one dimension.

    These live in the *source* VBench dataset, not in the derived counterfactual
    dataset, so the caller passes the source root explicitly.
    """
    name = ANNOTATION_FILES.get(dimension)
    if name is None:
        return {}
    path = annotations_root / "annotations" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry.get("prompt_en", "")): entry for entry in payload}


def metadata_item(dimension: str, row: dict[str, Any], annotation: dict[str, Any] | None) -> dict[str, Any]:
    """Build the metadata dict a backend's batch function expects for one clip."""
    prompt = row["prompt_en"]
    item: dict[str, Any] = {"prompt": prompt}
    if dimension == "human_action":
        prefix = "A person is "
        if not prompt.startswith(prefix):
            raise ValueError(f"human_action prompt lacks action: {prompt!r}")
        item["dimension_metadata"] = {"human_action": {"target_action": prompt[len(prefix) :]}}
    elif dimension == "spatial_relationship":
        ann = annotation or {}
        relation = (ann.get("relationship_en") or row.get("parsed", {}).get("relation") or "").strip()
        subject = (ann.get("object_a_en") or "").strip() or row.get("parsed", {}).get("subject", "")
        obj = (ann.get("object_b_en") or "").strip() or row.get("parsed", {}).get("object", "")
        item["dimension_metadata"] = {
            "spatial_relationship": {"object_a": subject, "object_b": obj, "relationship": relation}
        }
    elif dimension == "multiplt_object":
        ann = annotation or {}
        raw = str(ann.get("object_en") or "").strip()
        if not raw and "parsed" in row:
            raw = " and ".join(row["parsed"].get("targets", []))
        if not raw:
            raise ValueError(f"no object targets for {row.get('derived_id')}")
        # `_unwrap_object_metadata` accepts either shape; this matches the
        # official entry-builder in multiple_objects/backends/vbench.py.
        item["auxiliary_info"] = {
            "object": raw,
            "multiple_objects": {"object": raw},
        }
    elif dimension == "scene":
        # scene_label() reads the prompt; the base prompt is the target scene.
        item["auxiliary_info"] = {"scene": {"scene": prompt}}
    return item


def frame_evidence(diagnostics: Any) -> dict[str, Any] | None:
    """Compact per-clip evidence extracted from a backend's diagnostics.

    Without this the scored rows keep only a scalar, so a repair that floors at
    zero is indistinguishable from a repair that is systematically inverted:
    both look like a low mean.  The frame reasons separate them.
    """

    if not isinstance(diagnostics, dict):
        return None
    frames = diagnostics.get("frame_results")
    if frames is None:
        # Multiple Objects reports its per-frame evidence under `frames`.
        frames = diagnostics.get("frames")
    evidence: dict[str, Any] = {
        "detected_frame_count": diagnostics.get("detected_frame_count"),
        "scored_frame_count": diagnostics.get("scored_frame_count"),
        "valid_frame_count": diagnostics.get("valid_frame_count"),
        "missing_subject_count": diagnostics.get("missing_subject_count"),
        "missing_object_count": diagnostics.get("missing_object_count"),
        "ambiguous_role_count": diagnostics.get("ambiguous_role_count"),
        "aggregation_method": diagnostics.get("aggregation_method"),
    }
    # The official Multiple Objects backend reports only a frame count and how
    # many of those frames its conjunction accepted; keep both so the report can
    # show the same "how incomplete" number for both backends.
    if diagnostics.get("frame_count") is not None:
        evidence["frame_count"] = int(diagnostics["frame_count"])
    if diagnostics.get("success_frame_count") is not None:
        success = int(diagnostics["success_frame_count"])
        evidence["success_frame_count"] = success
        if evidence.get("frame_count"):
            evidence["joint_detection_rate"] = success / evidence["frame_count"]
    # Continuity backends (Motion Smoothness) record the mean/tail decomposition
    # of their own score rather than per-frame reasons; plan 13.3 asks for both
    # numbers next to the score, so keep them in the evidence block.
    for key in ("mean_discontinuity", "tail_discontinuity", "D_video"):
        if diagnostics.get(key) is not None:
            evidence[key] = float(diagnostics[key])
    if isinstance(frames, list) and frames:
        if isinstance(frames[0], dict):
            if any("frame_reason" in frame for frame in frames):
                counts: dict[str, int] = {}
                for frame in frames:
                    reason = str(frame.get("frame_reason"))
                    counts[reason] = counts.get(reason, 0) + 1
                evidence["frame_reason_counts"] = dict(sorted(counts.items()))
            else:
                # Multiple Objects: the audit backend reports per-target
                # confidences and a co-presence decision per frame, and those
                # are what separate "the target is absent" from "the detector
                # still answers with moderate confidence for a suppressed
                # region" (the failure this family exists to expose).
                evidence["frame_count"] = len(frames)
                decisions = [bool(frame.get("official_binary_frame_decision")) for frame in frames]
                scores = [float(frame.get("repaired_frame_score") or 0.0) for frame in frames]
                weakest = [
                    float(frame.get("weakest_official_threshold_score") or 0.0) for frame in frames
                ]
                # `joint_detection_rate` is the official-threshold pass on the
                # same frame; it is directly comparable to the official
                # backend's `joint_detection_rate`.
                evidence["joint_detection_rate"] = sum(decisions) / len(frames)
                evidence["mean_frame_score"] = sum(scores) / len(frames)
                evidence["mean_weakest_confidence"] = sum(weakest) / len(weakest)
                evidence["zero_frame_fraction"] = sum(1 for score in scores if score == 0.0) / len(scores)
        elif isinstance(frames[0], (int, float)):
            evidence["frame_count"] = len(frames)
            evidence["zero_frame_fraction"] = sum(1 for value in frames if float(value) == 0.0) / len(frames)
    return evidence


def _first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"score": None, "status": "failed", "error": "backend returned no rows"}
    row = rows[0]
    status = row.get("status")
    if status == "failed":
        return {"score": None, "status": "failed", "error": row.get("error") or row.get("failure_reason")}
    score = row.get("score")
    return {
        "score": float(score) if score is not None else None,
        "status": "succeeded" if score is not None else "failed",
        "error": None if score is not None else "backend returned no scalar score",
        "evidence": frame_evidence(row.get("diagnostics")),
    }


def existing_derived_ids(output: Path, backend: str, repair_config: dict[str, Any] | None) -> set[str]:
    """Already-scored ids in one shard file, refusing to mix repair variants.

    Resuming appends to the same file, so a shard that already holds rows from a
    different repair configuration is fatal rather than skipped: the CPA
    instrument would otherwise average two scoring contracts together.
    """
    done: set[str] = set()
    if not output.is_file():
        return done
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("backend") != backend:
            continue
        if repair_config is not None:
            recorded = {
                "repair_mode": entry.get("repair_mode", "ordered_role_identity_assignment"),
                "detection_conditioned": bool(entry.get("detection_conditioned", False)),
            }
            if recorded != repair_config:
                raise SystemExit(
                    f"{output} already holds {backend} rows scored with {recorded}, "
                    f"but this run requests {repair_config}; use a fresh --output"
                )
        done.add(entry["derived_id"])
    return done


def make_scorer(
    dimension: str,
    backend: str,
    device: Any,
    upstream: Path,
    *,
    repair_mode: str = "ordered_role_identity_assignment",
    detection_conditioned: bool = False,
) -> Callable[[Path, dict[str, Any]], dict[str, Any]]:
    """Return ev(video_path, meta_item) -> {'score','status','error'} for one backend.

    ``repair_mode``/``detection_conditioned`` only affect the Spatial Relationship
    repair, which is the one dimension whose repair backend exposes scoring
    variants (plan section 9.6).  Other dimensions ignore them.
    """
    _syspath()

    if dimension == "dynamics_degree":
        import numpy as np
        from dynamic_degree.backends.vbench import OfficialDynamicEvaluator
        from dynamic_degree.diagnostics import DiagnosticsLevel
        from dynamic_degree.metric import evaluate_audit_batch, weight_path
        from dynamic_degree.models import RaftFlowModel

        weight = Path(os.environ.get("VBENCH_AUDIT_RAFT_WEIGHT", weight_path())).expanduser()
        if backend == "official":
            evaluator = OfficialDynamicEvaluator(device, weight, upstream)
            def ev(video: Path, _: dict[str, Any]) -> dict[str, Any]:
                result = evaluator.evaluate_video(video)
                values = [float(value) for value in result.raw_flow_top5_mean]
                return {"score": float(np.mean(values)) if values else None, "status": "succeeded"}
        else:
            flow = RaftFlowModel(device, weight, upstream)
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(evaluate_audit_batch([video], {video.name: item}, device, weight, DiagnosticsLevel.FULL, flow_model=flow))
        return ev

    if dimension == "subject_consistency":
        from subject_consistency.metric import build_dino_config, evaluate_batch
        from subject_consistency.models import OfficialDinoFeatureExtractor
        config = build_dino_config()
        extractor = OfficialDinoFeatureExtractor(device, config, upstream)
        backend_name = "vbench" if backend == "official" else "audit"
        def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
            return _first(evaluate_batch(backend_name, [video], {video.name: item}, device, config, extractor=extractor))
        return ev

    if dimension == "human_action":
        from human_action.backends.audit import AuditHumanActionEvaluator
        from human_action.backends.vbench import OfficialHumanActionEvaluator, import_official_module
        from human_action.diagnostics import DiagnosticsLevel
        from human_action.metric import evaluate_audit_batch, load_categories, weight_path
        from human_action.models import LockedUmtClassifier

        weight = Path(os.environ.get("VBENCH_AUDIT_UMT_WEIGHT", weight_path())).expanduser()
        module, _ = import_official_module(upstream)
        if backend == "official":
            evaluator = OfficialHumanActionEvaluator(device, weight, upstream)
            def ev(video: Path, _: dict[str, Any]) -> dict[str, Any]:
                result = evaluator.evaluate_video(video)
                return {"score": float(bool(result.matched)), "status": "succeeded"}
        else:
            categories = load_categories(upstream)
            evaluator = AuditHumanActionEvaluator(LockedUmtClassifier(device, weight, module))
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(evaluate_audit_batch([video], {video.name: item}, device, weight, categories, DiagnosticsLevel.FULL, evaluator=evaluator))
        return ev

    if dimension == "spatial_relationship":
        from spatial_relationship.backends.vbench import (
            OfficialGritDetector,
            evaluate_official,
            normalize_official_results,
        )
        from spatial_relationship.diagnostics import DiagnosticsLevel
        from spatial_relationship.metric import evaluate_audit_batch, parse_query, weight_path
        from spatial_relationship.models import AblationMode

        weight = Path(os.environ.get("VBENCH_AUDIT_GRIT_WEIGHT", weight_path())).expanduser()
        if backend == "official":
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                raw, _ = evaluate_official([video], {video.name: item}, parse_query, device, weight, upstream_path=upstream)
                rows = normalize_official_results(raw, {video.name: item}, parse_query)
                return _first(rows)
        else:
            mode = AblationMode(repair_mode)
            detector = OfficialGritDetector(device, weight, upstream)
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(evaluate_audit_batch(
                    [video], {video.name: item}, device, weight, DiagnosticsLevel.FULL,
                    detector=detector, mode=mode, condition_on_detection=detection_conditioned,
                ))
        return ev

    if dimension == "scene":
        if backend == "official":
            from scene.metric import evaluate_official_batch
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(evaluate_official_batch([video], {video.name: item}, device))
        else:
            from scene.metric import evaluate_global_or_environment_batch
            from scene.models import build_scorer
            scorer = build_scorer(
                os.environ.get("VBENCH_AUDIT_SCENE_SCORER", "openclip"),
                device=device,
                model_name=os.environ.get("VBENCH_AUDIT_SCENE_MODEL", "ViT-B-32"),
                pretrained=os.environ.get("VBENCH_AUDIT_SCENE_PRETRAINED", "/root/.cache/clip/ViT-B-32.pt"),
            )
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(evaluate_global_or_environment_batch([video], {video.name: item}, scorer, "environment_grounded"))
        return ev

    if dimension == "multiplt_object":
        from multiple_objects.backends.vbench import DEFAULT_WEIGHT as GRIT_WEIGHT
        from multiple_objects.backends.vbench import evaluate_official_batch
        from multiple_objects.backends.audit import evaluate_audit_batch as mo_audit_batch
        from multiple_objects.schemas import MultipleObjectsConfig

        weight = Path(os.environ.get("VBENCH_AUDIT_GRIT_WEIGHT", str(GRIT_WEIGHT))).expanduser()
        if backend == "official":
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(evaluate_official_batch([video], {video.name: item}, device, weight))
        else:
            config = MultipleObjectsConfig()
            def ev(video: Path, item: dict[str, Any]) -> dict[str, Any]:
                return _first(mo_audit_batch([video], {video.name: item}, device, weight, config))
        return ev

    if dimension == "motion_smoothness":
        if backend == "official":
            from motion_smoothness.backends.vbench import DEFAULT_WEIGHT, OfficialMotionSmoothnessEvaluator
            weight = Path(os.environ.get("VBENCH_AUDIT_AMT_WEIGHT", str(DEFAULT_WEIGHT))).expanduser()
            evaluator = OfficialMotionSmoothnessEvaluator(device, checkpoint=weight, upstream_path=upstream)
            def ev(video: Path, _: dict[str, Any]) -> dict[str, Any]:
                return {"score": float(evaluator.evaluate_video(video)), "status": "succeeded"}
        else:
            from motion_smoothness.backends.audit import evaluate_timed_frames
            from motion_smoothness.models import RaftFlowEstimator, decode_timed_frames
            from motion_smoothness.schemas import MotionSmoothnessConfig
            weight = Path(os.environ.get("VBENCH_AUDIT_RAFT_WEIGHT", str(Path.home() / ".cache/vbench/raft_model/models/raft-things.pth"))).expanduser()
            estimator = RaftFlowEstimator(device, weight, upstream)
            config = MotionSmoothnessConfig()
            def ev(video: Path, _: dict[str, Any]) -> dict[str, Any]:
                frames, _ = decode_timed_frames(video)
                item = evaluate_timed_frames(video, frames, estimator, config)
                diagnostics = item.diagnostics
                return {
                    "score": float(item.score),
                    "status": "succeeded",
                    "evidence": {
                        "mean_discontinuity": diagnostics.get("mean_discontinuity"),
                        "tail_discontinuity": diagnostics.get("tail_discontinuity"),
                        "D_video": float(item.discontinuity),
                    },
                }
        return ev

    raise ValueError(f"no scorer for {dimension}/{backend}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True, choices=DIMENSIONS)
    parser.add_argument("--backend", required=True, choices=("official", "repair"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="derived counterfactual dataset holding the clips")
    parser.add_argument("--annotations-root", type=Path, default=None,
                        help="source VBench dataset holding annotations/ (defaults to --dataset-root)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--repair-mode",
        default=os.environ.get("VBENCH_AUDIT_SPATIAL_MODE", "ordered_role_identity_assignment"),
        choices=("official", "signed_only", "ordered_role", "ordered_role_identity_assignment"),
        help="Spatial Relationship repair variant (plan 9.6); ignored by other dimensions",
    )
    parser.add_argument(
        "--repair-detection-conditioned",
        action="store_true",
        default=os.environ.get("VBENCH_AUDIT_SPATIAL_DETECTION_CONDITIONED", "").lower() in {"1", "true", "yes"},
        help="Spatial Relationship repair: average only over frames where both roles were detected",
    )
    parser.add_argument("--check-only", action="store_true",
                        help="construct the scorer and exit; catches import/weight errors")
    args = parser.parse_args()

    os.environ["VBENCH_AUDIT_UPSTREAM"] = str(args.upstream)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["dimension"] == args.dimension]
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]

    annotations_root = args.annotations_root or args.dataset_root
    annotation = load_annotations(annotations_root, args.dimension)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip clips already scored by this backend.  A shard file must not
    # silently mix two repair configurations.  The knobs only exist for the
    # Spatial Relationship repair, so no other dimension records them.
    repair_config = {
        "repair_mode": args.repair_mode,
        "detection_conditioned": bool(args.repair_detection_conditioned),
    } if (args.dimension == "spatial_relationship" and args.backend == "repair") else None
    done = existing_derived_ids(args.output, args.backend, repair_config)
    scored = len(done)

    import torch
    scorer = make_scorer(
        args.dimension, args.backend, torch.device("cuda:0"), args.upstream,
        repair_mode=args.repair_mode,
        detection_conditioned=args.repair_detection_conditioned,
    )
    print(json.dumps({"event": "scorer_ready", "dimension": args.dimension, "backend": args.backend, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    if args.check_only:
        print(json.dumps({"event": "check_ok", "dimension": args.dimension, "backend": args.backend}), flush=True)
        return 0

    handle = args.output.open("a", encoding="utf-8")
    try:
        for index, row in enumerate(rows, start=1):
            derived_id = row["derived_id"]
            if derived_id in done:
                continue
            video = args.dataset_root / row["output_path"]
            try:
                item = metadata_item(args.dimension, row, annotation.get(row["prompt_en"]))
                result = scorer(video, item)
            except Exception as error:  # noqa: BLE001 - recorded, never dropped
                result = {"score": None, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            entry = {
                "derived_id": derived_id,
                "base_id": row["base_id"],
                "dimension": args.dimension,
                "family": row["family"],
                "level": row["level"],
                "expected_rank": row["expected_rank"],
                "split": row["split"],
                "backend": args.backend,
                "score": result.get("score"),
                "status": result.get("status"),
                "error": result.get("error"),
                "repair_mode": repair_config["repair_mode"] if repair_config else None,
                "detection_conditioned": repair_config["detection_conditioned"] if repair_config else None,
                "evidence": result.get("evidence"),
            }
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if result.get("score") is not None:
                scored += 1
            if index % 10 == 0 or index == len(rows):
                print(json.dumps({"event": "progress", "done": index, "total": len(rows), "scored": scored}), flush=True)
    finally:
        handle.close()
    print(json.dumps({"event": "complete", "dimension": args.dimension, "backend": args.backend, "scored": scored, "total": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
