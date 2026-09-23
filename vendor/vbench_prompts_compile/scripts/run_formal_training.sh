#!/usr/bin/env bash
# Launch the four formal trainings, one task per physical GPU.
#
# Usage: scripts/run_formal_training.sh <0.6b|8b> <spatial_card> <action_card> <objects_card> <scene_card> [model_root]
#
# Rules kept from the plan: one process per card (the process only ever sees
# logical cuda:0), no killing of other users' jobs, offline mode so nothing is
# downloaded implicitly, and per-task logs + exit codes in runs/formal/<size>/logs.
set -u

SIZE="${1:?usage: run_formal_training.sh <0.6b|8b> <spatial_card> <action_card> <objects_card> <scene_card> [model_root]}"
shift
if [ "$#" -lt 4 ]; then
  echo "need four card ids" >&2
  exit 2
fi
CARDS=("$1" "$2" "$3" "$4")
MODEL_ROOT="${5:-/data1/wenbiao_zhao/models}"
CONFIG_DIR="${CONFIG_DIR:-configs/formal}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/runs/formal/${SIZE}/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR"

case "$SIZE" in
  0.6b) MODEL="${MODEL_ROOT}/Qwen3-0.6B"; REV="c1899de289a04d12100db370d81485cdf75e47ca" ;;
  8b)   MODEL="${MODEL_ROOT}/Qwen3-8B";   REV="b968826d9c46dd6066d109eabc6255188de91218" ;;
  *) echo "unknown size $SIZE" >&2; exit 2 ;;
esac

TASKS=(spatial action objects scene)
PIDS=()
for index in 0 1 2 3; do
  task="${TASKS[$index]}"
  card="${CARDS[$index]}"
  entry="scripts/train_adapter.py"
  [ "$task" = "scene" ] && entry="scripts/train_scene.py"
  echo "launch $task on physical card $card -> $LOG_DIR/${task}.log"
  (
    cd "$PROJECT_ROOT" || exit 1
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$card" \
      "$PY" "$entry" \
        --config "${CONFIG_DIR}/${task}-qwen3-${SIZE}.json" \
        --model "$MODEL" \
        --model-revision "$REV" \
        > "$LOG_DIR/${task}.log" 2>&1
    echo "$?" > "$LOG_DIR/${task}.exit"
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
echo "all tasks finished; per-task exit codes:"
for task in "${TASKS[@]}"; do
  printf '  %-8s %s\n' "$task" "$(cat "$LOG_DIR/${task}.exit" 2>/dev/null || echo missing)"
done
exit "$status"
