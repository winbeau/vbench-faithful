#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DATA=/root/autodl-tmp/vbench-audit-storage/datasets/counterfactual/human_action/action_execution
RUN=/root/autodl-tmp/vbench-audit-storage/runs/human_action/action_execution
PY="$PROJECT/.venv/bin/python"
export PYTHONPATH="$PROJECT/metrics/human-action/src:$PROJECT/packages/audit-core/src"
mkdir -p "$RUN/logs" "$DATA/inputs" "$DATA/metadata"
validation="$($PY "$PROJECT/scripts/prepare_human_action_execution.py")"
printf '%s\n' "$validation" | tee "$RUN/logs/validation.json"
present=$(printf '%s' "$validation" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["present"])')
if [ "$present" != "15" ]; then
  echo "ACTION_EXECUTION_DATA_MISSING" >&2
  exit 3
fi
$PY - <<'PY'
import json
from pathlib import Path
root=Path('/root/autodl-tmp/vbench-audit-storage/datasets/counterfactual/human_action/action_execution')
items=[json.loads(x) for x in (root/'metadata'/'expected_manifest.jsonl').read_text().splitlines() if x]
inputs=root/'inputs'; entries=[]
for i,item in enumerate(items,1):
    source=Path(item['video_path']); alias=inputs/f'video_{i:03d}.mp4'
    if alias.exists() or alias.is_symlink():
        if not alias.is_symlink() or alias.resolve()!=source.resolve(): raise SystemExit(f'unsafe alias {alias}')
    else: alias.symlink_to(source)
    entries.append({'video':alias.name,'prompt':item['prompt'],'target_action':item['target_action'],'dimension_metadata':item['dimension_metadata'],'action_group':item['action_group'],'video_state':item['video_state'],'source_video':str(source)})
(root/'metadata'/'input_metadata.json').write_text(json.dumps({'videos':entries},indent=2)+'\n')
PY
if [ -f "$RUN/COMPLETE" ]; then
  echo "ACTION_EXECUTION_ALREADY_COMPLETE"
  exit 0
fi
started=$(date -Iseconds)
if $PY -m human_action.cli --both --video-dir "$DATA/inputs" --metadata "$DATA/metadata/input_metadata.json" --output "$RUN/batch" --gpu 0 --seed 20260914 >"$RUN/logs/run.log" 2>&1; then
  printf 'completed_at=%s\nstarted_at=%s\n' "$(date -Iseconds)" "$started" > "$RUN/COMPLETE"
else
  echo "ACTION_EXECUTION_BATCH_FAILED_OR_INTERRUPTED" >&2
  exit 1
fi
