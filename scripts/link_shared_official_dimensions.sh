#!/usr/bin/env bash
# Link VBench 1.0 dimensions that officially reuse subject_consistency videos.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_ROOT" >&2
  exit 2
fi

root="$(cd "$1" && pwd)"
generators=(lavie modelscope cogvideo videocrafter)
aliases=(dynamics_degree dynamic_degree motion_smoothness)

for generator in "${generators[@]}"; do
  directory="$root/$generator"
  source="$directory/subject_consistency"
  [[ -d "$source" ]] || { echo "missing source directory: $source" >&2; exit 1; }
  for alias in "${aliases[@]}"; do
    target="$directory/$alias"
    if [[ -e "$target" && ! -L "$target" ]]; then
      echo "refusing to replace non-symlink: $target" >&2
      exit 1
    fi
    ln -sfn subject_consistency "$target"
    [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || exit 1
    printf '%s -> %s\n' "$target" "$(readlink "$target")"
  done
done
