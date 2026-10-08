# Development

Use Python 3.11.14 and the root `uv.lock`. Each of the 16 metrics is an independent
package. Shared infrastructure belongs in `packages/audit-core/`, model adapters
in `packages/audit-models/`, and semantic training/scoring in
`packages/prompt-compiler/`. Metrics must not import one another.

```bash
uv lock --check
uv sync --locked --group test
# Explicit CPU overlay for algorithm tests, separate from model runtimes.
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.14.0+cpu'
uv run --no-sync --group test pytest tests metrics
uv run --no-sync python scripts/eval.py --help
```

Check all 16 package CLI help entries after workspace changes. Help and argument
validation must not download weights or initialize CUDA. CPU tests do not
establish model parity. GPU reports must record code, upstream, device, cohort,
coverage, cache state and wall time. Seven accelerators use the per-video error
gate `abs(ours - official) <= max(1e-6, 0.01 * abs(official))`; preserve the nine
repairs' missing and unsupported states.

The default entry is `scripts/eval.py` with `configs/eval.yaml`.
[Evaluation](evaluation.md) documents isolation and caching;
[training and construction](training.md) lists selected versions and necessary
parent stages. Older metric CLIs remain compatibility tools; their defaults do
not select the nine paper methods.

Downloads, environments, generated media, models, new results and local tool
state stay out of Git. See the [cleanup record](reproduction/REPOSITORY_CLEANUP.md)
for archived experiments. Never silently regenerate frozen inputs with a new
selection rule.

Commit messages must be entirely in English, including subject, body and
trailers. Use Conventional Commits (`feat/fix/refactor/test/docs/build/ci/chore`)
with a subject of at most 100 characters. Validate before committing:

```bash
python3 scripts/check_commit_message.py /path/to/commit-message.txt
mkdir -p .git/hooks
cp scripts/hooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
git diff --check
```

Never bypass the language check with `--no-verify`. Commit and push when the
user has authorized them; authorization persists during the task.
