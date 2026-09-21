"""Metric orchestration through shared I/O, models and evidence infrastructure."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from vbench_audit_core.artifacts import dimension_metadata, result_identity, write_json_artifact
from vbench_audit_core.schemas import RunSummary, VideoResult
from vbench_audit_models.labels import LabelVocabulary
from vbench_audit_models.prompt_compiler import PromptCompiler

from . import algorithms, models

DIMENSION = "object_class"
METRIC = "object-class"


def requires_cuda(model_config, backends):
    # Missing configuration must produce retained null rows even without CUDA.
    return bool(model_config.get("grit", {}).get("checkpoint"))


def failure(videos, metadata, backend, variant, exc):
    return [VideoResult(str(video), "failed", None,
            {**result_identity(video, metadata.get(video.name, {})),
             "backend": backend, "variant": variant,
             "failure_reason": type(exc).__name__}, f"{type(exc).__name__}: {exc}")
            for video in videos]


def evaluate_audit(videos, metadata, device, config, *, variant):
    model_config = config.get("model", {})
    evidence_dir = config.get("runtime", {}).get("evidence_dir")
    try:
        vocabulary = LabelVocabulary.from_file(model_config["labels"]["path"])
        if evidence_dir is None:
            raise ValueError("runtime.evidence_dir required for retained instance evidence")
        compiler_config = model_config.get("prompt_compiler", {})
        source = compiler_config.get("kind", "deterministic")
        compiler = None if source == "metadata" else PromptCompiler(DIMENSION, vocabulary, compiler_config)
        model = models.build_model(model_config, device=device)
    except Exception as exc:
        return failure(videos, metadata, "audit", variant, exc)
    rows = []
    for video in videos:
        entry = metadata.get(video.name, {})
        fields = {**result_identity(video, entry), "backend": "audit", "variant": variant,
                  "formula_version": "object_class-" + variant + "-v1",
                  "provenance": {**model.provenance, "vocabulary": vocabulary.provenance,
                                 "compiler": compiler.provenance if compiler else {"kind": "metadata",
                                 "scope": "explicit deterministic metadata experiment; not prompt-only LLM"}}}
        try:
            query = (dimension_metadata(entry, DIMENSION) if compiler is None else compiler(entry.get("prompt")))
            target = query.get("object")
            frames = model.detect_video(video)
            scored = algorithms.score_frames(frames, target, vocabulary, lexical=variant != "legacy")
            artifact = write_json_artifact(evidence_dir, DIMENSION,
                       {"video": str(video), "query": query, "raw_frames": frames,
                        "scoring": scored, "provenance": fields["provenance"]})
            fields.update({key: value for key, value in scored.items() if key not in {"frames", "score", "status"}})
            fields.update({"evidence": artifact, "query_source": source, "query": query})
            error = None if scored["status"] == "succeeded" else scored.get("reason", scored["status"])
            rows.append(VideoResult(str(video), scored["status"], scored["score"], fields, error))
        except Exception as exc:
            rows.append(VideoResult(str(video), "failed", None, fields, f"{type(exc).__name__}: {exc}"))
    return rows


def summarize(backend, rows):
    counts = Counter(row.status for row in rows)
    complete = bool(rows) and counts["succeeded"] == len(rows)
    observed = algorithms.aggregate(rows)
    status = "complete" if complete else ("partial" if counts["succeeded"] else "failed")
    versions = sorted({(row.metric or {}).get("formula_version", "unavailable") for row in rows})
    return RunSummary(METRIC, backend, status, len(rows), counts["succeeded"], counts["failed"],
                      counts["succeeded"], observed if complete else None, ",".join(versions),
                      [row.error for row in rows if row.error], dict(counts),
                      "all_sampled_frames",
                      {"input_video_count": len(rows), "retained_row_count": len(rows),
                       "scored_video_count": counts["succeeded"],
                       "coverage": counts["succeeded"]/len(rows) if rows else 0,
                       "unsupported_count": counts["unsupported"],
                       "official_dropped_count": counts["dropped_by_official"],
                       "conditional_dropped_count": counts["dropped_by_condition"],
                       "observed_subset_aggregate_diagnostic_only": observed,
                       "expected_frame_count": 16 * len(rows),
                       "observed_frame_count": sum((r.metric or {}).get("frame_count", 0) for r in rows)})
