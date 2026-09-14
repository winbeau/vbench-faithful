#!/usr/bin/env bash
# Resumable frozen-E0 repair evaluation.  This script never changes inputs.
set -uo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="$ROOT/runs/official_repair"
PY="$ROOT/.venv/bin/python"
RUNNER="$ROOT/scripts/run_official_dataset_compare_dimension.py"
STATS="$ROOT/scripts/evaluate_pairwise_statistics.py"
STATUS="$OUT/run_status.tsv"
OFFICIAL_METRICS="$OUT/reconstructed_baseline_metrics.csv"
DIMS=(dynamic_degree subject_consistency human_action spatial_relationship)

export VBENCH_AUDIT_UPSTREAM="${VBENCH_AUDIT_UPSTREAM:-$ROOT/../VBench}"
# DINO/GRiT resources are machine-specific. Set these variables explicitly
# when running the repair; no private workstation path is baked into this
# resumable research script.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$ROOT/packages/audit-core/src:$ROOT/metrics/dynamic-degree/src:$ROOT/metrics/subject-consistency/src:$ROOT/metrics/human-action/src:$ROOT/metrics/spatial-relationship/src:$VBENCH_AUDIT_UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT"
if [[ ! -f "$STATUS" ]]; then
  printf 'dimension\tstatus\texit_code\tstart_time\tend_time\tpredictions_path\tmetrics_path\n' > "$STATUS"
fi

update_status() {
  "$PY" -c '
import csv, sys
from pathlib import Path
path, dim, status, code, started, ended, predictions, metrics = sys.argv[1:]
fields=["dimension","status","exit_code","start_time","end_time","predictions_path","metrics_path"]
records={}
p=Path(path)
if p.exists():
    with p.open(newline="",encoding="utf8") as h:
        for row in csv.DictReader(h, delimiter="\t"):
            records[row["dimension"]]=row
records[dim]=dict(zip(fields,[dim,status,code,started,ended,predictions,metrics]))
tmp=p.with_suffix(".tmp")
with tmp.open("w",newline="",encoding="utf8") as h:
    w=csv.DictWriter(h,fieldnames=fields,delimiter="\t"); w.writeheader()
    for key in ("dynamic_degree","subject_consistency","human_action","spatial_relationship"):
        if key in records: w.writerow(records[key])
tmp.replace(p)
' "$STATUS" "$@"
}

is_complete() {
  "$PY" -c '
import csv, math, sys
from pathlib import Path
root, dim, repo = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
alias={"dynamic_degree":"dynamics_degree","subject_consistency":"subject_consistency","human_action":"human_action","spatial_relationship":"spatial_relationship"}[str(dim)]
manifest=repo / "data/processed/e0_scoring_manifest.csv"
with manifest.open(newline="",encoding="utf8") as h:
    expected=[r["video_uid"] for r in csv.DictReader(h) if r["dimension"]==alias]
pred=root/str(dim)/"predictions.csv"; metrics=root/str(dim)/"metrics.csv"
if not pred.is_file() or not metrics.is_file(): raise SystemExit(1)
with pred.open(newline="",encoding="utf8") as h: rows=list(csv.DictReader(h))
if len(rows)!=len(expected) or {r["video_uid"] for r in rows}!=set(expected): raise SystemExit(1)
for r in rows:
    status=r.get("repair_status")
    if status == "unsupported": continue
    if status != "succeeded_scalar": raise SystemExit(1)
    try: score=float(r["repair_score"])
    except (TypeError,ValueError): raise SystemExit(1)
    if not math.isfinite(score): raise SystemExit(1)
with metrics.open(newline="",encoding="utf8") as h: result=list(csv.DictReader(h))
if len(result)!=1 or result[0].get("dimension")!=str(dim): raise SystemExit(1)
' "$OUT" "$1" "$ROOT"
}

run_dimension() {
  local dim="$1" dir="$OUT/$1" log="$OUT/$1/run.log" start end rc metric_rc state
  mkdir -p "$dir"
  if is_complete "$dim"; then
    end=$(date -Iseconds)
    update_status "$dim" SKIPPED 0 "$end" "$end" "$dir/predictions.csv" "$dir/metrics.csv"
    echo "{\"event\":\"dimension_skipped_complete\",\"dimension\":\"$dim\"}"
    return 0
  fi
  start=$(date -Iseconds)
  update_status "$dim" RUNNING '' "$start" '' "$dir/predictions.csv" "$dir/metrics.csv"
  echo "{\"event\":\"dimension_start\",\"dimension\":\"$dim\",\"time\":\"$start\"}"
  "$PY" "$RUNNER" --dimension "$dim" --output-root "$OUT" --retry-failed 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  end=$(date -Iseconds)
  if [[ "$rc" -ne 0 ]]; then
    update_status "$dim" FAILED "$rc" "$start" "$end" "$dir/predictions.csv" "$dir/metrics.csv"
    echo "{\"event\":\"dimension_failed\",\"dimension\":\"$dim\",\"exit_code\":$rc}"
    return 0
  fi
  "$PY" "$STATS" --predictions-root "$OUT" --dimension "$dim" --output "$dir/metrics.csv" --bootstrap-iterations 2000 --seed 2026 2>&1 | tee -a "$log"
  metric_rc=${PIPESTATUS[0]}
  end=$(date -Iseconds)
  if [[ "$metric_rc" -ne 0 ]]; then
    update_status "$dim" FAILED "$metric_rc" "$start" "$end" "$dir/predictions.csv" "$dir/metrics.csv"
    echo "{\"event\":\"statistics_failed\",\"dimension\":\"$dim\",\"exit_code\":$metric_rc}"
  elif is_complete "$dim"; then
    update_status "$dim" SUCCESS 0 "$start" "$end" "$dir/predictions.csv" "$dir/metrics.csv"
    echo "{\"event\":\"dimension_success\",\"dimension\":\"$dim\"}"
  else
    update_status "$dim" FAILED 2 "$start" "$end" "$dir/predictions.csv" "$dir/metrics.csv"
    echo "{\"event\":\"dimension_incomplete_predictions\",\"dimension\":\"$dim\"}"
  fi
}

for dimension in "${DIMS[@]}"; do
  run_dimension "$dimension"
done

all_complete=1
for dimension in "${DIMS[@]}"; do
  is_complete "$dimension" || all_complete=0
done
if [[ "$all_complete" -eq 1 ]]; then
  "$PY" "$STATS" --predictions-root "$OUT" --output "$OUT/repair_metrics.csv" --bootstrap-iterations 2000 --seed 2026 \
    --official-metrics "$OFFICIAL_METRICS" --comparison-csv "$OUT/official_vs_repair.csv" --comparison-md "$OUT/official_vs_repair.md"
  echo '{"event":"official_vs_repair_complete"}'
else
  echo '{"event":"official_vs_repair_deferred","reason":"one_or_more_dimensions_incomplete"}'
fi
