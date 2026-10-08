"""Concise evaluation progress and an append-only structured event stream."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Event field is not JSON serializable: {type(value).__name__}")


class EvalLogger:
    """Log orchestration events; raw model output belongs in worker log files."""

    def __init__(self, output: Path | None, total: int):
        if type(total) is not int or total < 0:
            raise ValueError("total must be a nonnegative integer")
        self.output = Path(output) if output is not None else None
        self.total = total
        self.console = Console(stderr=True, markup=False, highlight=False)
        self._stream = None
        self._progress = None
        self._task = None
        self._finished = 0
        self._lock = RLock()

    def __enter__(self):
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            self._stream = (self.output / "events.jsonl").open("a", encoding="utf-8")
        if self.console.is_terminal:
            self._progress = Progress(
                SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                TaskProgressColumn(), console=self.console, transient=True,
            )
            self._task = self._progress.add_task("Evaluation", total=self.total)
            self._progress.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self.event("error", message=str(exc), error_type=exc_type.__name__)
        finally:
            if self._progress is not None:
                self._progress.stop()
                self._progress = None
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        return False

    def _record(self, stage, dimension, backend, message, fields):
        if "timestamp" in fields:
            raise ValueError("timestamp is reserved for the event recorder")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "stage": stage, "dimension": dimension, "backend": backend,
            "message": message, **fields,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=_json_default)
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()

    def event(self, stage: str, dimension: str = "", backend: str = "", message: str = "", **fields):
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        with self._lock:
            self._record(stage, dimension, backend, message, fields)
            target = "/".join(value for value in (dimension, backend) if value)
            failed = stage in {"error", "failed"} or fields.get("status") in {"error", "failed"}
            style = "bold red" if failed else "cyan"
            label = Text(f"[{stage}]", style=style)
            if target:
                label.append(" " + target)
            if fields.get("status"):
                label.append(" · " + str(fields["status"]))
            if message:
                # The full message stays in JSONL; a model traceback must not
                # bury orchestration failures in the terminal.
                label.append(" · " + " ".join(message.splitlines())[:300])
            self.console.print(label)
            if self._progress is not None:
                self._progress.update(self._task, description=target or stage)

    def finish_task(self, dimension: str = "", backend: str = "", status: str = "completed", message: str = "", **fields):
        with self._lock:
            self._finished += 1
            self.event("finished", dimension, backend, message, status=status,
                       completed_tasks=self._finished, total_tasks=self.total, **fields)
            if self._progress is not None:
                self._progress.update(self._task, completed=self._finished)

    def summary(self, rows: list[dict]):
        """Render dimension/backend summaries without inventing missing scores."""
        table = Table(title="Evaluation summary", show_lines=False)
        for name in ("Dimension", "Backend", "Status", "Coverage", "Score"):
            table.add_column(name, justify="right" if name in {"Coverage", "Score"} else "left")
        for row in rows:
            status = row.get("status", "complete" if row.get("complete") else "partial")
            coverage = row.get("coverage")
            coverage_text = f"{coverage:.1%}" if isinstance(coverage, (int, float)) else "—"
            score = row.get("score")
            score_text = f"{score:.6g}" if isinstance(score, (int, float)) else "—"
            table.add_row(str(row.get("dimension", "")), str(row.get("backend", "")),
                          str(status), coverage_text, score_text)
        with self._lock:
            self._record("summary", "", "", "", {"rows": rows})
            self.console.print(table)
