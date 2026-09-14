#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/vbench-audit
OUT=/root/autodl-tmp/vbench-audit-storage/runs/official_dataset_compare_4dims
PY="$ROOT/.venv/bin/python"
DIMS=(dynamic_degree spatial_relationship human_action subject_consistency)
mkdir -p "$OUT/logs"
export VBENCH_AUDIT_DINO_REPO=/root/autodl-tmp/vbench-audit-storage/models/dino/facebookresearch_dino_main
export PYTHONPATH="$ROOT/公共/audit-core/src:$ROOT/指标/dynamic-degree/src:$ROOT/指标/spatial_relationship/src:$ROOT/指标/human_action/src:$ROOT/指标/subject_consistency/src:/root/vbench1"
: >"$OUT/logs/all.log"
for d in "${DIMS[@]}"; do
 "$PY" "$ROOT/scripts/run_official_dataset_compare_dimension.py" --dimension "$d" --dry-run 2>&1 | tee "$OUT/logs/$d.log" | tee -a "$OUT/logs/all.log"
done
echo "DRY-RUN PASS" | tee -a "$OUT/logs/all.log"
