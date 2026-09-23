#!/usr/bin/env python3
"""Evaluate videos with the nine fixed paper repairs and/or official VBench 1.0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from paper_common import ROOT, PAPER, OFFICIAL, TASKS, dimension, digest, load_assets, load_inputs, process, result, summarize, verify_assets, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="per-video JSON/JSONL or VBench full-info JSON with video_list")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--dimensions", nargs="+", default=["paper"], help="paper (nine), all (sixteen), or named dimensions")
    parser.add_argument("--backend", choices=["repair", "origin", "both"], default="repair")
    parser.add_argument("--plan", action="store_true", help="validate inputs and print routes without loading models")
    args = parser.parse_args()
    dims = list(PAPER if args.dimensions == ["paper"] else OFFICIAL if args.dimensions == ["all"] else map(dimension, args.dimensions))
    if len(set(dims)) != len(dims):
        raise ValueError("Duplicate requested dimension")
    if args.backend != "origin" and any(d not in PAPER for d in dims):
        raise ValueError("Repair is defined for nine paper dimensions; select --backend origin --dimensions all for official VBench 16D")
    assets = load_assets(args.assets)
    inputs = load_inputs(args.input, args.video_root)
    profile = json.loads((ROOT / "configs/reproduction/paper-methods.json").read_text())
    counts = {d: sum(d in row["dimensions"] for row in inputs) for d in dims}
    if any(count == 0 for count in counts.values()):
        raise ValueError("No input for requested dimension: " + str([d for d, n in counts.items() if not n]))
    official_method = {"implementation": "official VBench 1.0",
                       "commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"}
    methods = {}
    for dim in dims:
        methods[dim] = {}
        if args.backend in {"origin", "both"}:
            methods[dim]["origin"] = official_method
        if args.backend in {"repair", "both"}:
            methods[dim]["repair"] = profile["methods"][dim]
    plan = {"backend": args.backend, "dimensions": dims, "inputs": counts,
            "methods": methods,
            "input_sha256": digest(args.input), "assets_sha256": digest(args.assets),
            "profile_sha256": digest(ROOT / "configs/reproduction/paper-methods.json")}
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    if output.exists() or any(output == ROOT / p or ROOT / p in output.parents for p in ["data", "results", "splits", "runs"]):
        raise ValueError("Use a new output directory outside frozen research directories")
    plan["verified_assets"] = verify_assets(assets, dims, args.backend)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    normalized_assets = output / "assets.json"
    write_json(normalized_assets, assets)
    backends = ["origin", "repair"] if args.backend == "both" else [args.backend]
    report = {"profile": "official-vbench-1.0" if args.backend == "origin" else "paper-selected-nine-dimensions",
              "dimensions": {}, "complete": True}
    for dim in dims:
        chosen = [row for row in inputs if dim in row["dimensions"]]
        folder = output / dim
        folder.mkdir()
        input_path = folder / "inputs.json"
        write_json(input_path, chosen)
        for backend in backends:
            destination = folder / (backend + ".json")
            try:
                common = ["--input", input_path, "--assets", normalized_assets, "--dimension", dim]
                if backend == "repair" and dim in TASKS:
                    evidence = folder / "evidence.json"
                    process(assets["visual_python"], "paper_visual.py", [*common, "--mode", "evidence", "--output", evidence], assets, folder / "evidence.log")
                    process(assets["semantic_python"], "paper_semantic.py", [*common, "--evidence", evidence, "--output", destination], assets, folder / "repair.log")
                elif backend == "repair" and dim in {"object_class", "color"}:
                    compiled = folder / "compiled.json"
                    process(assets["semantic_python"], "paper_semantic.py", [*common, "--output", compiled], assets, folder / "compile.log")
                    process(assets["visual_python"], "paper_visual.py", [*common, "--mode", "repair", "--compiled", compiled, "--output", destination], assets, folder / "repair.log")
                else:
                    process(assets["visual_python"], "paper_visual.py", [*common, "--mode", backend, "--output", destination], assets, folder / (backend + ".log"))
                payload = json.loads(destination.read_text())
                if {row["id"] for row in payload["rows"]} != {row["id"] for row in chosen}:
                    raise ValueError("Worker changed input coverage")
            except Exception as exc:
                payload = {"rows": [result(row, None, "failed", error=str(exc)) for row in chosen]}
                write_json(destination, payload)
            summary = summarize(payload["rows"])
            if "official_aggregate" in payload:
                summary["official_aggregate"] = payload["official_aggregate"]
                # MUSIQ returns per-video scores in 0..100 but its official
                # aggregate in 0..1; other dimensions also have native reducers.
                summary["score"] = payload["official_aggregate"] if summary["complete"] else None
                summary["aggregation"] = "official_upstream"
            report["dimensions"][f"{dim}/{backend}"] = summary
            report["complete"] &= summary["complete"]
            write_json(output / "summary.json", report)
            print(json.dumps({"dimension": dim, "backend": backend, **summary}), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
