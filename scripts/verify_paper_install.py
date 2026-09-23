#!/usr/bin/env python3
"""Run real video installation checks against frozen paper scores and official 16D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from paper_common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, help="default: install-fixture beside assets.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["repair", "origin", "both"], default="both")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture = args.fixture or args.assets.resolve().parent / "install-fixture"
    if args.backend != "origin" and not (fixture / "dynamic-cf/inputs.json").is_file():
        raise FileNotFoundError("Restore the complete install fixture, including dynamic-cf/inputs.json")
    args.output.mkdir(parents=True)
    report = {"scope": "representative fresh video inference; not all paper cohorts or retraining",
              "passed": True, "checks": {}}
    for backend in (["repair", "origin"] if args.backend == "both" else [args.backend]):
        output = args.output / backend
        input_path = fixture / (backend + "-inputs.json")
        extra = []
        if backend == "repair" and (fixture / "dynamic-cf/inputs.json").exists():
            extra = json.loads((fixture / "dynamic-cf/inputs.json").read_text())
            inputs = [{**row, "video": str((fixture / row["video"]).resolve())}
                      for row in json.loads(input_path.read_text())]
            inputs.extend({**row, "video": str((fixture / "dynamic-cf" / row["video"]).resolve())} for row in extra)
            input_path = args.output / "repair-inputs.json"
            write_json(input_path, inputs)
        run = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_vbench.py"),
                              "--input", str(input_path),
                              "--assets", str(args.assets), "--output", str(output),
                              "--backend", backend, "--dimensions", "paper" if backend == "repair" else "all"])
        summary_path = output / "summary.json"
        if not summary_path.exists():
            report["checks"][backend] = {"passed": False, "reason": "evaluation failed before producing scores"}
            report["passed"] = False
            continue
        summary = json.loads(summary_path.read_text())
        checked = {"passed": run.returncode == 0 and summary["complete"], "dimensions": summary["dimensions"]}
        if backend == "repair":
            expected = json.loads((fixture / "repair-expected.json").read_text())
            expected["records"].extend({"id": row["id"], "expected": row["expected"]} for row in extra)
            observed = {}
            for path in output.glob("*/repair.json"):
                for row in json.loads(path.read_text())["rows"]:
                    if row["id"] in observed:
                        raise ValueError("Duplicate verification input")
                    observed[row["id"]] = row
            differences = []
            for row in expected["records"]:
                actual = observed.get(row["id"], {}).get("score")
                error = None if actual is None else abs(actual - row["expected"])
                differences.append({"id": row["id"], "expected": row["expected"], "actual": actual,
                                    "absolute_error": error, "passed": error is not None and error <= 1e-6})
            checked.update(records=differences, tolerance=1e-6)
            checked["passed"] &= (set(observed) == {r["id"] for r in expected["records"]}
                                  and all(row["passed"] for row in differences))
        report["checks"][backend] = checked
        report["passed"] &= checked["passed"]
    write_json(args.output / "verification.json", report)
    print(json.dumps({"passed": report["passed"], "receipt": str(args.output / "verification.json")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
