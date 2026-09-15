#!/usr/bin/env bash
# Score counterfactual-vbench with Official and Repair backends.
#
# Six GPUs in parallel per dimension; dimensions run strictly one at a time so a
# dimension is merged and verified before the next starts (AGENTS.md multi-card
# rule).  Each worker sees one physical GPU through its own CUDA_VISIBLE_DEVICES
# mask and uses logical cuda:0 inside the process.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Everything this run needs lives under /root/wenbiao_zhao (the winbeau branch);
# nothing is read from or written into another user's tree.
PY=/root/wenbiao_zhao/venvs/vbench/bin/python
DATA=/root/wenbiao_zhao/datasets/vbench-1.0-human-preference
CF=/root/wenbiao_zhao/datasets/counterfactual-vbench
MANIFEST="$CF/manifest.jsonl"
UPSTREAM=/root/wenbiao_zhao/VBench
SCORES="$CF/scores"
mkdir -p "$SCORES"

# Weight locations on the H100 box (none are downloaded here).
export VBENCH_AUDIT_RAFT_WEIGHT=/root/wenbiao_zhao/models/raft/raft-things.pth
export VBENCH_AUDIT_AMT_WEIGHT=/root/.cache/vbench/amt_model/amt-s.pth
export VBENCH_AUDIT_GRIT_WEIGHT=/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth
export VBENCH_AUDIT_UMT_WEIGHT=/root/.cache/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth
export VBENCH_AUDIT_TAG2TEXT_WEIGHT=/root/.cache/vbench/caption_model/tag2text_swin_14m.pth
export VBENCH_AUDIT_UPSTREAM="$UPSTREAM"

# GPU 6 is visible to nvidia-smi but not CUDA-available inside this container,
# so the usable physical range is 1-5.
GPUS=(1 2 3 4 5)
DIMS=(dynamics_degree subject_consistency human_action spatial_relationship scene multiple_objects motion_smoothness)
BACKENDS=(official repair)

run_dimension() {
  local dim=$1 backend=$2
  local pids=()
  for i in "${!GPUS[@]}"; do
    local shard_file="$SCORES/${dim}__${backend}__shard${i}.jsonl"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -B -m scripts.counterfactual.score \
      --dimension "$dim" --backend "$backend" \
      --manifest "$MANIFEST" --dataset-root "$CF" \
      --output "$shard_file" --upstream "$UPSTREAM" \
      --shard-index "$i" --num-shards "${#GPUS[@]}" \
      > "$SCORES/${dim}__${backend}__shard${i}.log" 2>&1 &
    pids+=("$!")
  done
  local ok=1
  for pid in "${pids[@]}"; do
    wait "$pid" || ok=0
  done
  [[ "$ok" == "1" ]]
}

for dim in "${DIMS[@]}"; do
  for backend in "${BACKENDS[@]}"; do
    echo "[$(date -Is)] start $dim/$backend"
    if run_dimension "$dim" "$backend"; then
      echo "[$(date -Is)] done $dim/$backend"
    else
      echo "[$(date -Is)] FAILED $dim/$backend" >&2
    fi
  done
done

echo "[$(date -Is)] all dimensions scored"
