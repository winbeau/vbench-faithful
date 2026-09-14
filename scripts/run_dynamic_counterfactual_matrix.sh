#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/vbench-audit
DATA=/root/autodl-tmp/vbench-audit-storage/datasets/counterfactual/dynamic_degree
RUN=/root/autodl-tmp/vbench-audit-storage/runs/dynamic_degree
LOG=/root/autodl-tmp/vbench-audit-storage/logs/dynamic_degree
PY=/root/vbench-audit/.venv/bin/python
RAFT=/root/autodl-tmp/vbench-audit-storage/models/raft/raft-things.pth
EVAL="$RUN/evaluations.jsonl"
EVIDENCE="$RUN/structured_evidence.jsonl"
STATUS="$RUN/stage_status"
export VBENCH1_ROOT=/root/vbench1
export PYTHONPATH="$ROOT/指标/dynamic-degree/src:$ROOT/公共/audit-core/src"

mkdir -p "$RUN" "$LOG" "$STATUS" "$RUN/statistics"
exec 9>"$RUN/.matrix.lock"
flock -n 9 || { echo 'another Dynamic Degree matrix controller holds the lock'; exit 3; }

preflight() {
  test -x "$PY"
  test -f "$RAFT"
  test -f "$DATA/metadata/DATASET_FROZEN"
  test -f "$DATA/metadata/FROZEN_DATASET_HASHES.txt"
  grep -q 'READY_FOR_BATCH = YES' "$RUN/DYNAMIC_READINESS_REPORT_V3.md"
  (cd "$DATA" && sha256sum -c metadata/FROZEN_DATASET_HASHES.txt)
  "$PY" -c 'from dynamic_degree.evaluation_records import EvaluationRecordWriter; from dynamic_degree.backends.audit import AuditVariant; from dynamic_degree.backends.vbench import verify_upstream; verify_upstream()'
  echo "preflight PASS $(date --iso-8601=seconds)"
}

run_stage() {
  local split=$1
  local variant=$2
  local stage_log="$LOG/${variant}_${split}.log"
  echo "stage_start split=$split variant=$variant time=$(date --iso-8601=seconds)"
  "$PY" "$ROOT/scripts/run_dynamic_counterfactual_stage.py" \
    --split "$split" --variant "$variant" --dataset "$DATA" \
    --evaluations "$EVAL" --evidence "$EVIDENCE" \
    --stage-status "$STATUS/${variant}_${split}.json" --weight "$RAFT" \
    2>&1 | tee -a "$stage_log"
  echo "stage_complete split=$split variant=$variant time=$(date --iso-8601=seconds)"
}

run_statistics() {
  local split=$1 output=$2 report_root=$3 log_path=$4
  "$PY" "$ROOT/scripts/run_dynamic_counterfactual_statistics.py" \
    --split "$split" --evaluations "$EVAL" --evidence "$EVIDENCE" \
    --output "$output" --report-root "$report_root" \
    --bootstrap-iterations 5000 --seed 20260912 2>&1 | tee -a "$log_path"
}

preflight
if [[ ${1:-} == --dry-run ]]; then
  echo 'dry validation PASS; no evaluator executed'
  exit 0
fi

for variant in official time_only source_only duration_only source_time full; do
  run_stage dev "$variant"
done
run_statistics dev "$RUN/statistics/dev" "$RUN/dev_statistics" "$LOG/statistics.log"

"$PY" - "$STATUS" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for variant in ('official','time_only','source_only','duration_only','source_time','full'):
    x=json.loads((root/f'{variant}_dev.json').read_text())
    if x['expected_records'] != x['actual_records'] or x['duplicates'] or x['missing'] or x['unexpected'] or x['sidecar_records'] != x['expected_records']:
        raise SystemExit(f'DEV integrity failed: {variant}: {x}')
print('DEV integrity gate PASS')
PY

for variant in official time_only source_only duration_only source_time full; do
  run_stage test "$variant"
done
run_statistics test "$RUN/statistics" "$RUN" "$LOG/statistics.log"
"$PY" - "$RUN/statistics/bootstrap.json" 2>&1 <<'PY' | tee -a "$LOG/bootstrap.log"
import json, pathlib, sys
x=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert x['iterations']==5000 and x['unit']=='base_id'
print(f"bootstrap validation PASS iterations={x['iterations']} seed={x['seed']} unit={x['unit']}")
PY
echo "all_experiments_complete time=$(date --iso-8601=seconds)"
