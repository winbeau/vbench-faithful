from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import RunSummary, VideoResult


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"


def write_results(root: Path, results: Iterable[VideoResult], summary: RunSummary, run_info: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    result_list = []
    for item in results:
        serialized = asdict(item)
        metric_fields = serialized.pop("metric", None) or {}
        for key, value in metric_fields.items():
            if key in serialized:
                raise ValueError(f"metric result field collides with core field: {key}")
            serialized[key] = value
        result_list.append(serialized)
    (root / "results.json").write_text(json.dumps(result_list, ensure_ascii=False, indent=2), encoding="utf-8")
    preferred = ["video", "prompt", "subject", "relation", "object", "backend", "score", "status", "failure_reason", "error"]
    observed = {key for item in result_list for key, value in item.items() if not isinstance(value, (dict, list))}
    fields = [field for field in preferred if field in observed]
    fields.extend(sorted(observed - set(fields)))
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result_list:
            writer.writerow({field: item.get(field) for field in fields})
    (root / "summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log_lines = [
        f"metric={summary.metric}",
        f"backend={summary.backend}",
        f"status={summary.status}",
        f"total={summary.total}",
        f"succeeded={summary.succeeded}",
        f"failed={summary.failed}",
        f"output={root}",
    ]
    (root / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
