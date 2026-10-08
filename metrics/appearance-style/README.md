# Appearance Style

This package runs the pinned VBench 1.0 implementation. It has no selected paper repair.
The dimension adapter preserves upstream inputs, per-video results and the native aggregate.

From the workspace root:

```bash
uv run appearance-style --config configs/eval.yaml --plan
uv run appearance-style --config configs/eval.yaml --backend origin
```

This entry fixes the dimension to `appearance_style` even when the YAML selects `all`.
It uses the same YAML options, isolated visual environment, shared runtime dependencies,
asset verification, cache and logs as `scripts/eval.py`. Heavy model dependencies and
the pinned upstream checkout are restored through `scripts/restore_paper_runtime.py`,
not installed into this lightweight package. See [the evaluation guide](../../docs/evaluation.md).
