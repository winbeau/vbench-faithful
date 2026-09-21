"""Unmodified locked compute_color call; original strings are preserved."""
from __future__ import annotations

from pathlib import Path

from vbench_audit_core.artifacts import dimension_metadata, result_identity, write_json_artifact
from vbench_audit_core.schemas import VideoResult
from vbench_audit_core.upstream import import_official_module
from vbench_audit_models.grit import preflight

from .runtime import failure


def evaluate(videos, metadata, device, config):
    provenance, request = None, None
    try:
        model_config = config.get("model", {})
        checkpoint = Path(model_config["grit"]["checkpoint"]).expanduser().resolve()
        upstream = model_config.get("runtime", {}).get("upstream_root")
        provenance = preflight(checkpoint, upstream)
        evidence_dir = config["runtime"]["evidence_dir"]
        inputs = []
        for video in videos:
            entry = metadata.get(video.name, {})
            aux = dimension_metadata(entry, "color")
            if not isinstance(aux.get("color"), str) or not aux["color"].strip():
                raise ValueError("raw official color metadata required")
            prompt = entry.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("raw official prompt required")
            inputs.append({"dimension": ["color"], "prompt_en": prompt,
                           "auxiliary_info": {"color": {"color": aux["color"]}},
                           "video_list": [str(video.absolute())]})
        request = write_json_artifact(evidence_dir, "official-input", inputs)
        module, _ = import_official_module("color", upstream)
        import torch
        aggregate, raw_rows = module.compute_color(
            request["path"], torch.device(device or "cpu"), {"model_weight": str(checkpoint)})
        returned = write_json_artifact(evidence_dir, "official-return",
                                      {"aggregate": aggregate, "rows": raw_rows})
        by_video = {}
        expected = {str(video.absolute()) for video in videos}
        for raw in raw_rows:
            key = str(Path(raw["video_path"]).absolute())
            if key in by_video or key not in expected:
                raise ValueError("duplicate or unexpected upstream result")
            by_video[key] = raw
        results = []
        for video in videos:
            raw = by_video.get(str(video.absolute()))
            fields = {**result_identity(video, metadata.get(video.name, {})),
                      "backend": "vbench", "variant": "official",
                      "formula_version": "vbench-fd18b3d-color", "provenance": provenance,
                      "official_input": request, "official_return": returned,
                      "official_raw": raw, "official_batch_aggregate": aggregate}
            if raw is None:
                results.append(VideoResult(str(video), "dropped_by_official", None, fields,
                                           "upstream returned no row; retained in input denominator"))
            else:
                fields["frame_count"] = 16
                results.append(VideoResult(str(video), "succeeded", float(raw["video_results"]), fields))
        return results
    except Exception as exc:
        # No silent per-video retries, no fabrication when upstream raises
        # (including its all-dropped ZeroDivisionError).
        rows = failure(videos, metadata, "vbench", "official", exc)
        for row in rows:
            row.metric.update({"provenance": provenance, "official_input": request})
        return rows
