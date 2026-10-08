# VBench Faithful development

Read `README.md` and `docs/architecture.md` before changing the project.
The workspace uses Python 3.11.14 (`.python-version`) and the root `uv.lock`.
The September workspace plans are historical; the current layout is documented
in `docs/reproduction/REPOSITORY_CLEANUP.md`.

## Scope and boundaries

- Support all 16 VBench dimensions. Default `ours` runs the nine paper-selected
  repairs and seven accelerated dimensions; `official` runs original VBench.
  Method selection is fixed in `configs/reproduction/paper-methods.json`.
- Keep independent dimension packages in `metrics/<metric>/src/<import_name>/`.
  Metrics must not import one another.
- `packages/audit-core/` owns input, metadata, devices, scheduling, output,
  caching and provenance. `packages/audit-models/` owns shared model adapters.
  Do not put scoring formulas in either shared infrastructure package.
- `packages/prompt-compiler/` is first-party semantic adapter and paper scoring
  code, managed by the root workspace. Do not recreate a nested vendor project
  or add another lockfile. Keep the published scoring and parsing contracts.
- Preserve paper methods, protocols, source identities and selected model hashes
  in `configs/reproduction/`, and numerical evidence and figure provenance in
  `docs/`. Do not rewrite published results as a side effect of evaluation.
- Write downloads, environments, caches and new results under ignored `output/`.
  Model weights, external checkouts and machine-specific tool state stay out of Git.
  Retired E0 data/results/splits and obsolete root figures were removed with user
  authorization. Restore historical experiments from their commit, not by silently
  rebuilding old inputs with new selection rules.
- AOCI (`.aoci/`, `aoci.txt`, `.codex/config.toml`) is optional local tooling and
  is not published. A local index can become stale after source changes; verify
  its current coverage before claiming that it represents the working tree.
- Official Dynamic Degree and Motion Smoothness sampled videos reuse each
  generator's Subject Consistency media. Use symlinks for input aliases, not
  divergent copies; this does not imply shared formulas or human annotations.

## Validation and reporting

Run affected algorithm and contract tests after source changes. For workspace
or interface changes run `uv lock --check`, `uv sync --locked --group test`,
the CPU torch overlay documented in `CONTRIBUTING.md`,
`uv run --no-sync --group test pytest tests metrics`, and CLI help checks.
Do not label numerical replay or CPU tests as a CUDA rerun. GPU reports must
record the code and upstream SHA, devices, cohort and media duration, wall time,
coverage, cache status and output location.

Optimize one dimension at a time after measuring all 16. The four GRiT dimensions
may be optimized together. Seven accelerated dimensions must pass the per-video
error gate `abs(ours - official) <= max(1e-6, 0.01 * abs(official))`.
Keep the nine selected repairs' formulas and missing/unsupported states intact.
Do not terminate other users' GPU processes. Isolated workers see one physical
GPU as logical `cuda:0`; deduplicate resumed shards by video identity.

Dynamic aligned-v1 uses the frozen encoder/readout and the published 450-source
cohort; its external LASIESTA/BMC evidence is reported separately. Preserve
construction warnings, paired denominators, label provenance and uncertainty.
Do not describe aggregate stability as per-video invariance or agent review as
human ground truth. Historical negative results do not establish uniform gains.

## Commits

Commit and push when authorized by the user; authorization persists across the
active task. The current repository cleanup includes commit/push authorization.

**Commit messages must be written entirely in English.** This applies to the
subject, body, and trailers, including copied descriptions and logs. Chinese
text is forbidden. This rule supersedes Chinese commit examples in historical
plans and reports; it does not require translating source comments or research
documents.

Use Conventional Commits: `type(scope): description`, or
`type(scope)!: description` for breaking changes. Allowed types are
`feat/fix/refactor/test/docs/build/ci/chore`; keep the subject within 100
characters. Validate the complete proposed message with
`python3 scripts/check_commit_message.py /path/to/commit-message.txt` before
committing. Install `scripts/hooks/commit-msg` in the repository's local hooks
directory to enforce the same check for `git commit`. Never use `--no-verify`
to bypass this language rule. Rewriting published history requires explicit
user authorization.

Historical research receipts retain their original commit IDs. Use the
[commit identity map](docs/reproduction/git-history-map.tsv) to locate the
equivalent commits after the English-message history rewrite. Every mapped
commit preserves its file tree, author, committer, timestamps, and parent
relationships; the map is not evidence of a new evaluation run.
