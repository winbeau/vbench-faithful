#!/usr/bin/env bash
set -euo pipefail

# Launch one dimension as independent, resume-safe processes.  Each process is
# restricted to one physical GPU and therefore uses logical cuda:0 internally.
# The script never kills or signals existing processes.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 DIMENSION [MANIFEST] [OUTPUT_ROOT]" >&2
  exit 2
fi

dimension=$1
manifest=${2:-data/processed/e0_scoring_manifest.csv}
output_root=${3:-output/supplementary_20260914/repair_shards/${dimension}}
gpus_csv=${GPUS:-0,1,2,3}
backend=${BACKEND:-repair}
data_root=${DATA_ROOT:-/data/chenjiayu/wenbiao_zhao/vbench-official-v1}
official_root=${OFFICIAL_ROOT:-output/supplementary_20260914/official_scores}
upstream=${UPSTREAM:-/data/chenjiayu/wenbiao_zhao/VBench-fd18b3d-clean}

IFS=',' read -r -a gpus <<< "$gpus_csv"
if [[ "${#gpus[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi

mkdir -p "$output_root"
num_shards=${#gpus[@]}
for shard in "${!gpus[@]}"; do
  gpu=${gpus[$shard]}
  shard_root="$output_root/shard$shard"
  mkdir -p "$shard_root"
  CUDA_VISIBLE_DEVICES="$gpu" nohup uv run --no-sync python scripts/run_official_dataset_compare_dimension.py \
    --dimension "$dimension" --backend "$backend" --manifest "$manifest" \
    --num-shards "$num_shards" --shard-index "$shard" \
    --data-root "$data_root" --official-root "$official_root" \
    --output-root "$shard_root" --upstream "$upstream" \
    > "$shard_root/console.log" 2>&1 &
  echo $! > "$shard_root/pid"
  printf 'launched dimension=%s shard=%s physical_gpu=%s pid=%s output=%s\n' \
    "$dimension" "$shard" "$gpu" "$(cat "$shard_root/pid")" "$shard_root"
done

printf 'started %s shard(s); poll with: %s\n' "$num_shards" \
  "for p in $output_root/shard*/pid; do ps -p \$(cat \"\$p\"); done"
