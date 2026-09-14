#!/usr/bin/env bash
set -uo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUNNER="$ROOT/scripts/run_e0_official_dimension.py"
OUT=/root/autodl-tmp/vbench-audit-storage/scores/official/e0
LOG=/root/autodl-tmp/vbench-audit-storage/logs/e0_official
mkdir -p "$OUT" "$LOG"
rm -f "$OUT/status.txt"
export VBENCH_AUDIT_UPSTREAM="${VBENCH_AUDIT_UPSTREAM:-$ROOT/../VBench}"
export VBENCH_AUDIT_UMT_WEIGHT=/root/autodl-tmp/vbench-audit-storage/models/umt/l16_ptk710_ftk710_ftk400_f16_res224.pth
export VBENCH_AUDIT_DINO_REPO=/root/autodl-tmp/vbench-audit-storage/models/dino/facebookresearch_dino_main
export VBENCH_AUDIT_DINO_WEIGHT=/root/autodl-tmp/vbench-audit-storage/models/dino/dino_vitbase16_pretrain.pth
export VBENCH_AUDIT_RAFT_WEIGHT=/root/autodl-tmp/vbench-audit-storage/models/raft/raft-things.pth
export VBENCH_AUDIT_GRIT_WEIGHT=/root/autodl-tmp/vbench-audit-storage/models/grit/grit_b_densecap_objectdet.pth
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" "$RUNNER" metadata >>"$LOG/e0_all.log" 2>&1
dims=(human_action subject_consistency dynamics_degree spatial_relationship)
run_one () {
  local stage="$1" dim="$2"; shift 2
  if [[ "$dim" == spatial_relationship ]]; then
    CUDA_HOME=/root/autodl-tmp/vbench-audit-storage/toolchains/cuda-12.8 PATH=/root/autodl-tmp/vbench-audit-storage/toolchains/cuda-12.8/bin:$PATH LD_LIBRARY_PATH=/root/autodl-tmp/vbench-audit-storage/toolchains/cuda-12.8/lib64:${LD_LIBRARY_PATH:-} TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=12 PYTHONPATH=/root/autodl-tmp/vbench-audit-storage/envs/spatial-grit/site-packages:$VBENCH_AUDIT_UPSTREAM:$VBENCH_AUDIT_UPSTREAM/vbench/third_party/grit_src:$VBENCH_AUDIT_UPSTREAM/vbench/third_party/grit_src/centernet2 "$PY" "$RUNNER" "$stage" --dimension "$dim" 2>&1 | tee -a "$LOG/$dim.log" | tee -a "$LOG/e0_all.log"
  else
    PYTHONPATH="$ROOT/metrics/human-action/src:$ROOT/metrics/subject-consistency/src:$ROOT/metrics/dynamic-degree/src:$ROOT/metrics/spatial-relationship/src:$VBENCH_AUDIT_UPSTREAM" "$PY" "$RUNNER" "$stage" --dimension "$dim" 2>&1 | tee -a "$LOG/$dim.log" | tee -a "$LOG/e0_all.log"
  fi
  return ${PIPESTATUS[0]}
}
declare -A ready
all_pass=1
for dim in "${dims[@]}"; do
  if run_one preflight "$dim"; then ready[$dim]=1; else ready[$dim]=0; all_pass=0; fi
done
if (( all_pass )); then echo 'E0 PREFLIGHT = PASS' | tee -a "$LOG/e0_all.log"; else echo 'E0 PREFLIGHT = PARTIAL' | tee -a "$LOG/e0_all.log"; fi
for dim in "${dims[@]}"; do
  if [[ ${ready[$dim]} == 1 ]]; then
    if ! run_one run "$dim"; then "$PY" -c "from pathlib import Path; p=Path('$OUT/status.txt'); s=p.read_text(); p.write_text(s.replace('$dim RUNNING','$dim DIMENSION FAILED'),encoding='utf-8')"; echo "$dim DIMENSION FAILED" | tee -a "$LOG/e0_all.log"; fi
  fi
done
"$PY" "$RUNNER" summary >>"$LOG/e0_all.log" 2>&1
