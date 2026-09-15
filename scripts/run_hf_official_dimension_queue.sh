#!/usr/bin/env bash
set -u

# Download physical VBench dimension directories serially. Dynamic Degree and
# Motion Smoothness officially share Subject Consistency videos; create their
# local links with link_shared_official_dimensions.sh after that download.
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DEST="${VBENCH_OFFICIAL_DEST:-/data/chenjiayu/wenbiao_zhao/vbench-official-v1}"
LOG="$DEST/logs"
mkdir -p "$LOG"

dimensions=(subject_consistency human_action spatial_relationship scene multiple_objects overall_consistency)
status_file="$LOG/dimension_queue.tsv"
if [[ ! -f "$status_file" ]]; then
  printf 'dimension\texit_code\tstarted\tended\tmanifest\n' > "$status_file"
fi

for dimension in "${dimensions[@]}"; do
  if awk -F '\t' -v d="$dimension" '$1 == d && $2 == 0 {found=1} END {exit found ? 0 : 1}' "$status_file"; then
    continue
  fi
  existing_pid_file="$LOG/${dimension}.pid"
  if [[ -f "$existing_pid_file" ]]; then
    existing_pid=$(cat "$existing_pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      while kill -0 "$existing_pid" 2>/dev/null; do sleep 20; done
      if grep -q '"event": "complete"' "$LOG/${dimension}.log" && ! grep -q '"failures": [1-9]' "$LOG/${dimension}.log"; then
        printf '%s\t0\t%s\t%s\t%s\n' "$dimension" "existing" "$(date -Iseconds)" "$DEST/_manifests/${dimension}.jsonl" >> "$status_file"
        continue
      fi
      printf '%s\t1\t%s\t%s\t%s\n' "$dimension" "existing" "$(date -Iseconds)" "$DEST/_manifests/${dimension}.jsonl" >> "$status_file"
      exit 1
    fi
  fi
  started=$(date -Iseconds)
  log="$LOG/${dimension}.log"
  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 "$ROOT/scripts/download_vbench_official_by_dimension.py" \
    --dimension "$dimension" --dest "$DEST" --workers 4 > "$log" 2>&1
  rc=$?
  ended=$(date -Iseconds)
  printf '%s\t%s\t%s\t%s\t%s\n' "$dimension" "$rc" "$started" "$ended" "$DEST/_manifests/${dimension}.jsonl" >> "$status_file"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
done

printf '{"status":"COMPLETE","dimensions":%s,"status_file":"%s"}\n' \
  "$(printf '%s\n' "${dimensions[@]}" | python3 -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))')" \
  "$status_file"
