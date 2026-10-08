# Repository cleanup — 2026-10-08

The public working tree now centers on the 16-dimension evaluator and the
paper-selected nine repairs. The reference before cleanup is
[`f57822e`](https://github.com/winbeau/vbench-faithful/tree/f57822e58bc2fdc18e590ffd06ead2c6a3d57ed6).
This removes working-tree material; it does not rewrite Git history.

| Previous location | Decision |
|---|---|
| `data/`, `splits/`, `results/e0/` | Remove the early E0 preference baseline, inventories, split and score tables |
| `runs/official_repair/` | Remove its sole file, an obsolete E0 reconstruction report; it contained no runnable evaluator |
| `figures/` | Remove the superseded four-dimension supplementary tables/plots; current paper figures stay in `docs/assets/` |
| E0-only preprocessing, launchers and table generator | Remove with their frozen-baseline regression test |
| `packages/audit-core/`, `packages/audit-models/` | Keep the shared runtime required by the evaluator |
| `vendor/vbench_prompts_compile/` | Integrate the required adapters, training helpers and scorer into `packages/prompt-compiler/`; remove the nested project's obsolete configs, pilot scripts and lockfile |
| `.githooks/commit-msg` | Move to `scripts/hooks/commit-msg`; retain English-message enforcement |
| `.aoci/`, `aoci.txt` | Stop tracking; preserve the local pre-cleanup index, ignored by Git; check and refresh its coverage before reuse |

The three tests that read the retired E0 selection pool are removed. Pure
algorithm, construction, isolation, cache, failure-state and CLI contracts remain.
Historical documents may name old paths; use the reference commit above for
those experiments rather than treating them as current quick-start inputs.

## Paper reproduction

`configs/reproduction/release.json` still pins the exact published data,
139-file training snapshot, models and replay archives. Restore them with
`scripts/prepare_reproduction.py` under `output/`. Numerical table replay uses
the restored `model-code/` explicitly, as documented in the reproduction guide.
The selected checkpoints, nine method definitions, paper result tables and
published archive hashes are unchanged.

The active semantic source manifest now pins the 16 integrated runtime files:
14 moved byte for byte, and `scoring.py` contains six AST-identical functions
from the published scorer with package-relative imports. The unchanged UMT
initializer lives in `audit-models/umt.py`. Each manifest entry
also records its published source path and hash. Current inference no longer
depends on the old snapshot's unrelated configurations or command scripts.

## Validation

Root lock resolution and locked sync pass. CPU tests: **1,566 passed, 3 skipped**;
all **16** dimension CLI help checks pass. This includes 78 golden
scorer cases generated with the pre-integration implementation. They cover
Scene, Action, Spatial and Objects across original/rule/model routes, partial
evidence, direction changes, temporal confirmation, abstention and missingness.
Golden comparisons require the entire structured result to match, not just means.

H100 fresh inference also passes on the same 32 videos for the four affected
semantic dimensions: **128/128 complete result rows, scores, states and semantic
predictions match the pre-cleanup implementation exactly**. Scene, Action and
Objects each score 32/32; Spatial keeps its original 11/32 scores and 21 unsupported
states. Exit status 1 correctly reports that existing partial coverage.

The tested source tree is `53bb478bcdb25a7488c1897e1b064642e5a74a7f`, recorded
by the temporary H100 validation commit `7be47cada975b5139d4bb511953d7dbd57eaa17d`.
Publication adds documentation and Git attributes; inference/scoring source bytes
are identical to this tested tree. The run took 238.87 seconds on physical H100
GPU 3 with fresh isolated environments and `--no-reuse`. This is a four-dimension
integration check on a shared host, not a replacement for the 16-dimension speed
comparison or a rerun of all paper experiments.

The [machine-readable validation receipt](../validation/layout-cleanup-20261008.json)
records the GPU, original upstream SHA, source identity, input hashes, per-video
media durations, coverage and per-dimension comparison. Selected paper release
manifests and existing numerical tables were not regenerated.
