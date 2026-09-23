#!/usr/bin/env bash
# Control the four-task training suite inside a tmux session on the GPU host.
#
# The suite is *staged*, not started: after `stage` it waits for a go-file, so a
# network drop, an SSH timeout or a closed laptop cannot kill it, and no GPU is
# occupied until someone explicitly says go.
#
# Usage (run on the GPU host, inside the project directory):
#   scripts/tmux_train_control.sh stage 8b 4 6 3 1
#   scripts/tmux_train_control.sh go          # start training
#   scripts/tmux_train_control.sh pause       # SIGSTOP the workers (keeps GPU memory)
#   scripts/tmux_train_control.sh resume      # SIGCONT
#   scripts/tmux_train_control.sh status
#   scripts/tmux_train_control.sh stop        # terminate workers, keep the session
set -u

ACTION="${1:-status}"
SIZE="${2:-8b}"
shift 2 2>/dev/null || true

SESSION="${SESSION:-vpc-train}"
GO_FILE="${GO_FILE:-/data1/wenbiao_zhao/GO_TRAIN}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATCH="scripts/train_adapter.py|scripts/train_scene.py"

workers() { pgrep -f "$MATCH" || true; }

case "$ACTION" in
  stage)
    if [ "$#" -gt 0 ]; then CARDS=("$@"); else CARDS=(4 6 3 1); fi
    rm -f "$GO_FILE"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    CONFIG_DIR_RESOLVED="${CONFIG_DIR:-configs/formal}"
    tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" \
      "export CONFIG_DIR='$CONFIG_DIR_RESOLVED'; export GO_FILE='$GO_FILE'; \
       echo \"staged: size=$SIZE cards=${CARDS[*]} config_dir=\$CONFIG_DIR\"; \
       echo 'waiting for $GO_FILE ...'; while [ ! -f '$GO_FILE' ]; do sleep 5; done; \
       echo started \$(date -Is); bash scripts/run_formal_training.sh $SIZE ${CARDS[*]} 2>&1 | tee runs/formal/$SIZE/logs/suite.log; \
       echo finished \$(date -Is); exec bash"
    echo "staged session '$SESSION' (size=$SIZE cards=${CARDS[*]}) waiting for $GO_FILE"
    ;;
  go)
    touch "$GO_FILE"
    echo "go: training starts inside tmux session '$SESSION'"
    ;;
  pause)
    for pid in $(workers); do kill -STOP "$pid" && echo "paused $pid"; done
    ;;
  resume)
    for pid in $(workers); do kill -CONT "$pid" && echo "resumed $pid"; done
    ;;
  stop)
    for pid in $(workers); do kill -TERM "$pid" && echo "terminated $pid"; done
    rm -f "$GO_FILE"
    ;;
  status)
    tmux ls 2>/dev/null | sed -n "1,20p"
    echo "--- go file: $([ -f "$GO_FILE" ] && echo present || echo absent) ---"
    echo "--- workers ---"
    ps -o pid,stat,etime,pcpu,args -p "$(workers | paste -sd, -)" 2>/dev/null | sed -n '1,10p'
    echo "--- gpu ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ;;
  *)
    echo "usage: $0 {stage|go|pause|resume|status|stop} [size] [cards...]" >&2
    exit 2
    ;;
esac
