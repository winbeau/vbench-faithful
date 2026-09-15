"""Score counterfactual clips with the lag-calibrated Audit backend.

This is the driver behind the regenerated `dynamics_degree` report.  It differs
from `score.py` in what it persists: `score.py` keeps only score/status/error, so
the lag-calibrated evidence never reaches a report.  Here every clip also records
its measured `LagScalingEvidence`, the applied exponent and provenance, the
threshold domain and the per-channel coverage.

Per clip it stores, for each motion channel, the duration-weighted mean
displacement in image diagonals and the mean dt.  Within a clip dt is constant, so
the channel intensity under exponent p is exactly `mean_displacement / dt**p`: one
GPU pass therefore yields the whole exponent sweep without re-running RAFT, which
is how the shipped 0.5 exponent and its bootstrap interval were calibrated.

Usage (one shard per physical GPU, each process sees one card as logical cuda:0):

    for i in 0 1 2 3 4; do
      CUDA_VISIBLE_DEVICES=$((i+1)) python -m scripts.counterfactual.score_lagcal \
        --shard-index $i --num-shards 5 --mode fixed \
        --output <dir>/fixed05__shard$i.jsonl
    done

`--mode` selects the normalisation: `fixed` (the shipped default), `measured`
(each clip's own fitted exponent), `ballistic` (the archived `d/dt`), or `raw`
(no time normalisation).  Feed the resulting files to
`scripts/counterfactual/regen_dynamics_degree_report.py`.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in (
    "packages/audit-core/src", "packages/audit-models/src", "metrics/dynamic-degree/src")]
import torch  # noqa: E402
from dynamic_degree.backends import audit as audit_backend  # noqa: E402
from dynamic_degree.backends.audit import AuditConfig, LagExponentMode, analyze_timed_flow_sequence  # noqa: E402
from dynamic_degree.models import RaftFlowModel, decode_timed_frames  # noqa: E402

DEFAULT_DATASET_ROOT = Path("/root/wenbiao_zhao/datasets/counterfactual-vbench")
DEFAULT_WEIGHT = Path("/root/wenbiao_zhao/models/raft/raft-things.pth")
DEFAULT_UPSTREAM = Path("/root/wenbiao_zhao/VBench")


def channel_means(result):
    """Duration-weighted mean displacement per channel, and the mean dt."""
    valid = [t for t in result.transitions if t.valid and t.dt_seconds > 0]
    duration = sum(t.dt_seconds for t in valid)
    out = {"valid_transitions": len(valid), "duration_s": duration}
    if duration <= 0:
        return out
    for channel in ("apparent", "global", "residual"):
        values = [(getattr(t, f"{channel}_displacement"), t.dt_seconds) for t in valid
                  if getattr(t, f"{channel}_displacement") is not None]
        if not values:
            out[f"{channel}_mean_displacement"] = None
            continue
        total = sum(v * dt for v, dt in values)
        weight = sum(dt for _, dt in values)
        out[f"{channel}_mean_displacement"] = total / weight
        out[f"{channel}_mean_dt"] = weight / len(values)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--mode", default="fixed",
                        choices=("fixed", "measured", "ballistic", "raw"))
    parser.add_argument("--exponent", type=float, default=None,
                        help="explicit fixed exponent; requires --exponent-source")
    parser.add_argument("--exponent-source", default=None)
    parser.add_argument("--dimension", default="dynamics_degree")
    parser.add_argument("--limit", type=int, default=0, help="first N rows after sharding, for smoke tests")
    args = parser.parse_args()
    CF = args.dataset_root

    rows = [json.loads(line) for line in (CF / "manifest.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["dimension"] == args.dimension]
    rows.sort(key=lambda r: r["derived_id"])
    rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[: args.limit]

    if args.exponent is not None:
        config = AuditConfig(lag_exponent=args.exponent,
                             lag_exponent_source=args.exponent_source or "cli_fixed")
    elif args.mode == "ballistic":
        config = AuditConfig(lag_exponent_mode=LagExponentMode.BALLISTIC)
    elif args.mode == "measured":
        config = AuditConfig(lag_exponent_mode=LagExponentMode.MEASURED)
    elif args.mode == "raw":
        config = AuditConfig(ablation=audit_backend.AuditAblation.WITHOUT_TIME_NORMALIZATION)
    else:
        config = AuditConfig()

    flow = RaftFlowModel(torch.device("cuda:0"), args.weight, args.upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle = args.output.open("a", encoding="utf-8")
    try:
        for index, row in enumerate(rows, start=1):
            video = CF / row["output_path"]
            entry = {"derived_id": row["derived_id"], "base_id": row["base_id"], "level": row["level"],
                     "dimension": row["dimension"],
                     "split": row["split"], "expected_rank": row["expected_rank"], "family": row["family"],
                     "mode": args.mode, "status": None, "score": None, "exponent": None,
                     "exponent_source": None, "mean_dt": None}
            try:
                sequence = decode_timed_frames(video)
                result = analyze_timed_flow_sequence(str(video), row["prompt_en"], sequence, flow, config=config)
                entry.update({"status": result.status, "score": result.score,
                              "exponent": result.time_normalization_exponent,
                              "exponent_source": result.time_normalization_exponent_source,
                              "selected_channel": result.selected_evidence_channel,
                              "target": result.target_decision.target,
                              "n_frames": len(sequence.frames),
                              "frame_shape": list(sequence.frame_shape),
                              "lag_scaling": None if result.lag_scaling is None else result.lag_scaling.to_dict(),
                              "threshold": {"value": result.threshold.value, "units": result.threshold.units,
                                            "exponent": result.threshold.exponent},
                              "coverage": {"apparent": result.apparent.temporal_coverage,
                                           "camera": result.camera.temporal_coverage,
                                           "residual": result.residual.temporal_coverage},
                              "channel_means": channel_means(result),
                              "escalar_units_note": "intensity = mean_displacement / mean_dt**exponent"})
            except Exception as error:  # noqa: BLE001
                entry.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 5 == 0 or index == len(rows):
                print(json.dumps({"event": "progress", "done": index, "total": len(rows),
                                  "shard": args.shard_index}), flush=True)
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
