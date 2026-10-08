# Imaging Quality

This package offers the pinned original VBench 1.0 implementation (`--backend official`)
and the default accelerated implementation (`--backend ours`). Both retain the native
aggregation contract. This dimension has no selected paper repair; numerical acceptance
of its accelerator is documented in the [optimization record](../../docs/plans/2026-10-08-speed32-optimization.md).

From the workspace root:

```bash
uv run imaging-quality --config configs/eval.yaml --plan
uv run imaging-quality --config configs/eval.yaml --backend origin
```

This entry fixes the dimension to `imaging_quality` even when the YAML selects `all`.
It uses the same YAML options, isolated visual environment, shared runtime dependencies,
asset verification, cache and logs as `scripts/eval.py`. Heavy model dependencies and
the pinned upstream checkout are restored through `scripts/restore_paper_runtime.py`,
not installed into this lightweight package. See [the evaluation guide](../../docs/evaluation.md).
