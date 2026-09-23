#!/usr/bin/env bash
# Run the new checkout with the existing, separately managed H100 model runtime.
set -euo pipefail
repair_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_python="${VBENCH_REPAIR_PYTHON:-/root/wenbiao_zhao/venvs/vbench/bin/python}"
if [[ ! -x "$runtime_python" ]]; then
  echo "Missing model runtime: $runtime_python (set VBENCH_REPAIR_PYTHON)" >&2
  exit 2
fi
repair_pythonpath="$repair_root"
for source_dir in "$repair_root"/packages/*/src "$repair_root"/metrics/*/src; do
  repair_pythonpath="$repair_pythonpath:$source_dir"
done
export PYTHONPATH="$repair_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VBENCH_AUDIT_UPSTREAM="${VBENCH_AUDIT_UPSTREAM:-/root/wenbiao_zhao/VBench}"
export VBENCH_AUDIT_RAFT_WEIGHT="${VBENCH_AUDIT_RAFT_WEIGHT:-/root/wenbiao_zhao/models/raft/raft-things.pth}"
cd "$repair_root"
exec "$runtime_python" "$@"
