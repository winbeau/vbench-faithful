"""Score the independent Dynamic Degree FPS validation set (P1.3).

Scores every clip of the holdout built by `build_dynamic_validation.py` with the
Official backend and with the repair at three fixed lag exponents, then reports
the per-base `fps2/fps8` ratio for each: median, IQR, and the fraction inside
+/-20%.

`alpha = 1` is the archived ballistic normalisation (`d/dt`), `alpha = 0.5` is the
shipped diffusive default, and `alpha = 0` is no time normalisation at all (raw
displacement). The contract is invariance, so the ratio should sit near 1.0; the
point of the holdout is that 0.5 was chosen on a different, prompt-disjoint set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, read_jsonl, write_json

ALPHAS = (0.0, 0.5, 1.0)


def _syspath() -> None:
    for rel in ("packages/audit-core/src", "packages/audit-models/src", "metrics/dynamic-degree/src"):
        path = ROOT / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    upstream = os.environ.get("VBENCH_AUDIT_UPSTREAM")
    if upstream and upstream not in sys.path:
        sys.path.insert(0, upstream)


def score_all(root: Path, manifest: Path, device: Any, weight: Path, upstream: Path, limit: int = 0) -> dict[str, dict[str, float]]:
    _syspath()
    from dynamic_degree.backends.audit import AuditConfig
    from dynamic_degree.backends.vbench import OfficialDynamicEvaluator
    from dynamic_degree.diagnostics import DiagnosticsLevel
    from dynamic_degree.metric import evaluate_audit_batch
    from dynamic_degree.models import RaftFlowModel

    rows = read_jsonl(manifest)
    if limit:
        rows = rows[: limit * 4]
    out: dict[str, dict[str, float]] = {}

    official = OfficialDynamicEvaluator(device, weight, upstream)
    flow = RaftFlowModel(device, weight, upstream)

    for index, row in enumerate(rows, 1):
        video = root / row["output_path"]
        entry = out.setdefault(row["derived_id"], {})
        try:
            result = official.evaluate_video(video)
            values = [float(v) for v in result.raw_flow_top5_mean]
            entry["official"] = float(np.mean(values)) if values else None
        except Exception as error:  # noqa: BLE001
            entry["official_error"] = f"{type(error).__name__}: {error}"
        for alpha in ALPHAS:
            key = f"alpha{alpha:g}"
            try:
                config = AuditConfig(
                    lag_exponent=alpha,
                    lag_exponent_source="p1_3_validation_fixed_alpha",
                )
                produced = evaluate_audit_batch(
                    [video], {video.name: {"prompt": row["prompt_en"]}}, device, weight,
                    DiagnosticsLevel.OFF, flow_model=flow, config=config,
                )
                entry[key] = produced[0].get("score")
            except Exception as error:  # noqa: BLE001
                entry[f"{key}_error"] = f"{type(error).__name__}: {error}"
        if index % 20 == 0 or index == len(rows):
            print(json.dumps({"event": "progress", "done": index, "total": len(rows)}), flush=True)
    return out


def ratio_report(scores: dict[str, dict[str, float]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_base: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        entry = scores.get(row["derived_id"], {})
        by_base.setdefault(row["base_id"], {})[row["level"]] = entry
    report: dict[str, Any] = {}
    for method in ("official", *[f"alpha{a:g}" for a in ALPHAS]):
        ratios = []
        for levels in by_base.values():
            high = levels.get("fps8", {}).get(method)
            low = levels.get("fps2", {}).get(method)
            if high and low and high > 0:
                ratios.append(low / high)
        if not ratios:
            report[method] = {"n_bases": 0}
            continue
        array = np.asarray(ratios, dtype=float)
        report[method] = {
            "n_bases": len(ratios),
            "median_ratio_fps2_over_fps8": round(float(np.median(array)), 4),
            "q25": round(float(np.percentile(array, 25)), 4),
            "q75": round(float(np.percentile(array, 75)), 4),
            "within_pm20pct": round(float(np.mean(np.abs(array - 1.0) <= 0.20)), 4),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="validation dataset root")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch

    scores = score_all(args.root, args.manifest, torch.device("cuda:0"), args.weight, args.upstream, args.limit)
    rows = read_jsonl(args.manifest)
    report = {"scores": scores, "ratio": ratio_report(scores, rows)}
    write_json(args.output, report)
    print(json.dumps({"status": "COMPLETE", "ratio": report["ratio"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
