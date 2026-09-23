#!/usr/bin/env python3
"""Plot training curves (loss, token accuracy, dev exact match) without new deps.

Reads the TRL ``trainer_state.json`` log history plus ``dev_metrics.jsonl`` from
one or more run directories and writes, per run:

* ``history.csv``   -- step, loss, learning_rate, grad_norm, token accuracy, eval loss
* ``loss.svg``      -- training/eval loss over steps
* ``accuracy.svg``  -- token accuracy and generation-based dev exact match
* ``curves.html``   -- both charts side by side for quick viewing

Pure standard library: the project lock stays unchanged and the same command
works on the training server.

Usage::

    uv run --no-sync python scripts/plot_training_curves.py --run runs/formal/8b/spatial
    uv run --no-sync python scripts/plot_training_curves.py --glob 'runs/formal/8b/*' --output-dir docs/curves
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 900, 420
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 70, 20, 40, 50
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def find_state(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1]))
    if candidates:
        return candidates[-1]
    fallback = run_dir / "trainer_state.json"
    return fallback if fallback.exists() else None


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    state_path = find_state(run_dir)
    if state_path is None:
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return list(state.get("log_history", []))


def load_dev_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "dev_metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def collect_series(history: Iterable[dict[str, Any]], dev_rows: Iterable[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {
        "train_loss": [],
        "eval_loss": [],
        "token_accuracy": [],
        "eval_token_accuracy": [],
        "grad_norm": [],
        "learning_rate": [],
        "dev_exact_match": [],
        "dev_parseable_rate": [],
    }
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            series["train_loss"].append((float(step), float(entry["loss"])))
        if "eval_loss" in entry:
            series["eval_loss"].append((float(step), float(entry["eval_loss"])))
        if "mean_token_accuracy" in entry:
            series["token_accuracy"].append((float(step), float(entry["mean_token_accuracy"])))
        if "eval_mean_token_accuracy" in entry:
            series["eval_token_accuracy"].append((float(step), float(entry["eval_mean_token_accuracy"])))
        if "grad_norm" in entry:
            series["grad_norm"].append((float(step), float(entry["grad_norm"])))
        if "learning_rate" in entry:
            series["learning_rate"].append((float(step), float(entry["learning_rate"])))
    for row in dev_rows:
        step = row.get("step")
        if step is None:
            continue
        if "dev_exact_match" in row:
            series["dev_exact_match"].append((float(step), float(row["dev_exact_match"])))
        if "dev_parseable_rate" in row:
            series["dev_parseable_rate"].append((float(step), float(row["dev_parseable_rate"])))
    return {key: value for key, value in series.items() if value}


def _scale(values: Sequence[float], lo: float, hi: float, start: float, end: float) -> list[float]:
    span = hi - lo or 1.0
    return [start + (value - lo) / span * (end - start) for value in values]


def render_chart(title: str, panels: list[tuple[str, list[tuple[float, float]]]], *, y_min: float | None = None, y_max: float | None = None) -> str:
    points = [p for _, series in panels for p in series]
    if not points:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"><text x="20" y="30">no data for {title}</text></svg>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo = y_min if y_min is not None else min(ys)
    y_hi = y_max if y_max is not None else max(ys)
    if y_lo == y_hi:
        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5
    pad = (y_hi - y_lo) * 0.08
    y_lo -= pad
    y_hi += pad
    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

    def sx(x: float) -> float:
        return PAD_LEFT + (x - x_lo) / (x_hi - x_lo or 1.0) * plot_w

    def sy(y: float) -> float:
        return PAD_TOP + plot_h - (y - y_lo) / (y_hi - y_lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="{PAD_LEFT}" y="24" font-size="16" font-weight="bold">{title}</text>',
    ]
    for i in range(5):
        value = y_lo + (y_hi - y_lo) * i / 4
        y = sy(value)
        parts.append(f'<line x1="{PAD_LEFT}" y1="{y:.1f}" x2="{WIDTH - PAD_RIGHT}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{PAD_LEFT - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#555">{value:.3g}</text>')
    for i in range(5):
        value = x_lo + (x_hi - x_lo) * i / 4
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" y1="{PAD_TOP}" x2="{x:.1f}" y2="{PAD_TOP + plot_h}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{x:.1f}" y="{PAD_TOP + plot_h + 18}" font-size="11" text-anchor="middle" fill="#555">{value:.0f}</text>')
    parts.append(f'<text x="{PAD_LEFT + plot_w / 2:.0f}" y="{HEIGHT - 8}" font-size="12" text-anchor="middle" fill="#333">training step</text>')
    for index, (label, series) in enumerate(panels):
        color = COLORS[index % len(COLORS)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in series)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        for x, y in series:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.2" fill="{color}"/>')
        legend_x = PAD_LEFT + 8 + index * 220
        parts.append(f'<rect x="{legend_x}" y="{HEIGHT - 34}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{HEIGHT - 24}" font-size="12" fill="#333">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def process_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    history = load_history(run_dir)
    dev_rows = load_dev_metrics(run_dir)
    series = collect_series(history, dev_rows)
    name = run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}-history.csv"
    steps = sorted({int(step) for values in series.values() for step, _ in values})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        keys = [key for key in series]
        writer.writerow(["step", *keys])
        for step in steps:
            row: list[Any] = [step]
            for key in keys:
                value = next((value for s, value in series[key] if int(s) == step), "")
                row.append(value)
            writer.writerow(row)

    loss_panels = [(label, series[key]) for label, key in (("train loss", "train_loss"), ("eval loss", "eval_loss")) if key in series]
    acc_panels = [
        (label, series[key])
        for label, key in (
            ("token accuracy", "token_accuracy"),
            ("eval token accuracy", "eval_token_accuracy"),
            ("dev exact match", "dev_exact_match"),
            ("dev parseable", "dev_parseable_rate"),
        )
        if key in series
    ]
    loss_svg = render_chart(f"{name}: loss", loss_panels)
    acc_svg = render_chart(f"{name}: accuracy", acc_panels, y_min=0.0, y_max=1.0)
    (output_dir / f"{name}-loss.svg").write_text(loss_svg, encoding="utf-8")
    (output_dir / f"{name}-accuracy.svg").write_text(acc_svg, encoding="utf-8")
    html = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{name} training curves</title>"
        "<body style='font-family:Helvetica,Arial,sans-serif;margin:24px'>"
        f"<h2>{name}</h2>"
        f"<div>{loss_svg}</div><div style='margin-top:24px'>{acc_svg}</div>"
        "</body>"
    )
    (output_dir / f"{name}-curves.html").write_text(html, encoding="utf-8")
    final = {key: values[-1][1] for key, values in series.items()}
    return {
        "run": name,
        "steps": len(steps),
        "final": final,
        "csv": str(csv_path),
        "loss_svg": str(output_dir / f"{name}-loss.svg"),
        "accuracy_svg": str(output_dir / f"{name}-accuracy.svg"),
        "html": str(output_dir / f"{name}-curves.html"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="run directory (repeatable)")
    parser.add_argument("--glob", default=None, help="glob of run directories")
    parser.add_argument("--output-dir", default=None, help="where to write curves (default: each run directory)")
    parser.add_argument("--summary", default=None, help="optional JSON file collecting all runs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [Path(item) for item in args.run]
    if args.glob:
        runs += [Path(item) for item in sorted(glob.glob(args.glob)) if Path(item).is_dir()]
    runs = [run for run in runs if run.exists()]
    if not runs:
        print("no run directories matched", file=sys.stderr)
        return 2
    results = []
    for run in runs:
        target = Path(args.output_dir) if args.output_dir else run
        results.append(process_run(run, target))
        print(json.dumps(results[-1], ensure_ascii=False))
    if args.summary:
        Path(args.summary).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
