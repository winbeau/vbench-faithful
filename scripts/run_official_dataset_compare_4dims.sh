#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
OUT=/root/autodl-tmp/vbench-audit-storage/runs/official_dataset_compare_4dims
PY="$ROOT/.venv/bin/python"
DIMS=(dynamic_degree spatial_relationship human_action subject_consistency)
mkdir -p "$OUT/logs"
export VBENCH_AUDIT_DINO_REPO=/root/autodl-tmp/vbench-audit-storage/models/dino/facebookresearch_dino_main
export VBENCH_AUDIT_UPSTREAM="${VBENCH_AUDIT_UPSTREAM:-$ROOT/../VBench}"
export PYTHONPATH="$ROOT/packages/audit-core/src:$ROOT/metrics/dynamic-degree/src:$ROOT/metrics/spatial-relationship/src:$ROOT/metrics/human-action/src:$ROOT/metrics/subject-consistency/src:$VBENCH_AUDIT_UPSTREAM"
: >"$OUT/logs/all.log"
for d in "${DIMS[@]}"; do
 "$PY" "$ROOT/scripts/run_official_dataset_compare_dimension.py" --dimension "$d" --dry-run 2>&1 | tee "$OUT/logs/$d.log" | tee -a "$OUT/logs/all.log"
done
echo "DRY-RUN PASS" | tee -a "$OUT/logs/all.log"
