# Paper semantic compiler

First-party workspace package for the paper's Scene, Human Action, Spatial
Relationship and Multiple Objects adapters and scoring. It retains the
`vbench_prompts_compile` import name for checkpoint and reproduction compatibility.
Object Class and Color use the shared prompt compiler in `audit-models`.

The root `uv.lock` manages this package. It has no model dependencies in the
root environment; the evaluator uses the isolated semantic runtime described
in [the evaluation guide](../../docs/evaluation.md).

Fourteen modules were moved byte for byte from the published semantic snapshot.
`scoring.py` contains the six unchanged scoring functions from `score_matrix.py`,
with package-relative imports. Source hashes and the original publication
identities are recorded in
[`semantic-source.json`](../../configs/reproduction/semantic-source.json).
The visual worker's UMT initializer is integrated separately in
`audit-models/src/vbench_audit_models/umt.py`, preserving its original model setup.

The former nested project's pilot scripts, old configurations and second lockfile
are removed. Full paper training and numerical replay still restore the exact
published snapshot through `scripts/prepare_reproduction.py` and
[`release.json`](../../configs/reproduction/release.json); no published archive
hashes are changed. The training wrapper calls the pinned TRL `SFTTrainer` and
PEFT APIs; it does not copy upstream trainer code. See the
[reproduction guide](../../docs/reproduction/README.md) for the selected jobs.
