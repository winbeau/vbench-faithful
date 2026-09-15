"""Attribute the natural-preference deficit of the shipped v2 repair to its exponent.

P1.1 measured the shipped v2 repair (`d/dt**0.5`, residual channel, continuous
score with a dev-calibrated tie margin) against Official on human pair labels and
found it significantly worse (-0.1155).  That compares two whole decision rules,
so it does not say whether the *sqrt* exponent is responsible or whether the rest
of the v2 bundle is.

The natural run stored its full per-transition evidence in
`<natural-root>/dynamic_degree/structured_evidence.jsonl`, which carries
`residual_displacement` and `dt_seconds` for every valid transition.  The shipped
score is exactly the duration-weighted mean of `residual_displacement / dt**alpha`
over those transitions, so every other exponent can be reconstructed offline and
scored with the repository's own pair statistics.

This script writes `<tmp>/<alpha>/predictions.csv` from the frozen evidence and
calls `scripts/evaluate_paired_backend_delta.py` for each alpha.  `alpha = 0.5`
must reproduce the shipped P1.1 numbers; that is the fidelity check on the whole
reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIM = "dynamic_degree"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transition_intensity(transitions: list[dict], alpha: float) -> float | None:
    """Duration-weighted mean of `displacement / dt**alpha` over valid transitions."""
    numerator = 0.0
    duration = 0.0
    for transition in transitions:
        displacement = transition.get("residual_displacement")
        dt = transition.get("dt_seconds")
        if not transition.get("valid") or displacement is None or dt is None or dt <= 0:
            continue
        numerator += (displacement / dt**alpha) * dt
        duration += dt
    return numerator / duration if duration > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-root", type=Path,
                        default=Path("/root/wenbiao_zhao/datasets/natural-preference-runs"))
    parser.add_argument("--official-scores-root", type=Path,
                        default=ROOT / "results/e0/raw_official_scores")
    parser.add_argument("--work", type=Path, default=Path("/tmp/natural_alpha_attribution"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    evidence_path = args.natural_root / DIM / "structured_evidence.jsonl"
    template_path = args.natural_root / DIM / "predictions.csv"
    with template_path.open(newline="", encoding="utf-8") as handle:
        template = list(csv.DictReader(handle))
        fieldnames = list(template[0].keys())
    evidence = {row["video_uid"]: row for row in read_jsonl(evidence_path)}

    results: dict[str, dict] = {"shipped": {"rows": len(template), "scored": sum(
        1 for row in template if row.get("repair_status") == "succeeded_scalar")}}
    for alpha in args.alphas:
        directory = args.work / f"alpha{alpha:g}" / DIM
        directory.mkdir(parents=True, exist_ok=True)
        out_rows, missing = [], 0
        for row in template:
            entry = evidence.get(row["video_uid"])
            value = None
            if entry is not None:
                value = transition_intensity(entry.get("diagnostics", {}).get("transitions") or [], alpha)
            updated = dict(row)
            if value is None:
                missing += 1
                updated["repair_status"] = "failed"
                updated["repair_score"] = ""
            else:
                updated["repair_status"] = "succeeded_scalar"
                updated["repair_score"] = repr(float(value))
            out_rows.append(updated)
        with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        command = [
            sys.executable, str(ROOT / "scripts/evaluate_paired_backend_delta.py"),
            "--dimension", DIM,
            "--predictions-root", str(args.work / f"alpha{alpha:g}"),
            "--official-scores-root", str(args.official_scores_root),
            "--bootstrap-iterations", str(args.iterations), "--seed", "2026",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, cwd=ROOT)
        record = None
        if completed.returncode == 0:
            try:
                record = json.loads(completed.stdout.split("\n\n")[0])
            except json.JSONDecodeError:
                record = None
        results[f"alpha{alpha:g}"] = {
            "missing_evidence": missing, "returncode": completed.returncode,
            "record": record,
            "stderr_tail": completed.stderr.strip().splitlines()[-2:] if completed.stderr else [],
        }
        if record:
            print(json.dumps({"event": "alpha", "alpha": alpha,
                              "repair_accuracy": record["repair_accuracy"],
                              "delta": record["delta"], "ci": record["delta_ci95"],
                              "margin": record["repair_tie_margin"]}), flush=True)

    output = args.work / "alpha_attribution.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
