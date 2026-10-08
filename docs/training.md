# Selected paper training and data construction

These are part of the deliverable alongside the 16-dimension evaluator. Earlier
version numbers remain only where they provide required initialization, source
selection or shared operators. Inference selection stays fixed in
[`paper-methods.json`](../configs/reproduction/paper-methods.json).

| Dimension | Selected version | Construction / training entry |
|---|---|---|
| Scene | v8, step 300; 455 train / 108 dev | `scripts/semantic/prepare_repair_training.py --task scene`; `train_scene.py` |
| Human Action | v9, step 300; 605 / 67 | `scripts/semantic/prepare_action_isolation.py`; `train_adapter.py` |
| Spatial Relationship | v8, step 600; 2,679 / 312 | `scripts/semantic/prepare_repair_training.py --task spatial`; `train_adapter.py` |
| Multiple Objects | v6, step 900; 7,688 / 811 | `scripts/semantic/build_formal_sets.py`; `train_adapter.py` |
| Object Class / Color | final step 300; 424 / 465 training records | `scripts/object_color.py`, `object_color_construction.py`, `object_color_semantics.py` |
| Dynamic Degree | aligned-v1, step 300 | `scripts/counterfactual/vjepa_motion_probe.py` → `train_vjepa_anchored.py` / `train_vjepa_anchored_chunked.py` → `train_vjepa_aligned.py` |
| Subject Consistency | frozen hybrid-v5 / evaluated-v9 official720 | `scripts/counterfactual/generate_subject_masks_parallel.py`, `build_region_discrimination.py`, `build_subject_frame_probe.py` |
| Background Consistency | `patch_frame_calibrated`, gain 1.75 | `scripts/counterfactual/build_background_interventions.py`, `verify_background_interventions.py`, `run_background_holdout.py` |

Subject and Background use frozen pretrained models and deterministic scoring.
There is no selected Subject LoRA or Background model training stage. The seven
accelerated dimensions do not introduce training.

## Semantic training

Restore the published training inputs and dependency snapshot; this reuses
frozen teacher outputs without an annotation API call:

```bash
uv venv output/bootstrap --python 3.11.14
uv pip install --python output/bootstrap/bin/python 'huggingface_hub==1.32.0' requests
output/bootstrap/bin/python scripts/prepare_reproduction.py \
  --output output/reproduction --base-model /absolute/path/to/Qwen3-8B \
  --allow-official-fallback

# The pinned published lock builds a separate training environment in output/.
repro_work="$PWD/output/reproduction"
uv sync --project "$repro_work/model-code" --locked --extra train
train_python="$repro_work/model-code/.venv/bin/python"

# Training code is integrated in this repository.
"$train_python" scripts/semantic/train_scene.py \
  --config "$repro_work/training-configs/scene.json"
"$train_python" scripts/semantic/train_adapter.py \
  --config "$repro_work/training-configs/spatial_relationship.json"
```

Use `human_action.json` and `multiple_objects.json` for the other two adapters.
The four selected source configurations are in
[`configs/training/`](../configs/training/). Preparation relocates input, model
and output paths without changing hyperparameters. Python 3.11.14, torch
2.7.1+cu126, Transformers 4.52.4, TRL 0.19.1 and PEFT 0.15.2 stay separate from
the visual inference ABI. Qwen3-8B is pinned at
`b968826d9c46dd6066d109eabc6255188de91218`.

Object Class and Color use the complete restored `records.json`:

```bash
PYTHONPATH="$PWD/packages/audit-core/src:$PWD/packages/audit-models/src:$PWD/packages/prompt-compiler/src:$PWD" \
  "$train_python" -m scripts.object_color_semantics train \
  --dimension object_class --base /absolute/path/to/Qwen3-8B \
  --output output/reproduction/object-color
```

Repeat with `--dimension color`. Each has its own adapter. Do not substitute a
filtered JSONL export for the shared training records or re-run `silver` to
reproduce frozen labels. Use fresh output directories for repeated training.

## Data construction

The integrated [`scripts/semantic/`](../scripts/semantic/) builders retain source
cleaning, group splits, weak supervision, caption evidence, Action test-family
exclusion and Spatial four-direction conversion. `build_matrix_plan.py` freezes
counterfactual prompts and eligibility; `cache_matrix_transforms.py` constructs
mirrors and occlusions with all 16 frame slots and failures retained. Shared
helpers live in `packages/prompt-compiler/`. The
[source manifest](../configs/reproduction/training-source.json) records original
hashes and path/import adaptations. Smoke models and superseded configurations
are excluded. Restored raw annotations and source partitions must be supplied;
new online teacher output is not byte-exact reproduction.

Construction defaults to `output/reproduction/model-code/data`. Set
`VBENCH_TRAINING_DATA` for another data tree, `VBENCH_TRAINING_RAW` for raw source
snapshots, and `VBENCH_TRAINING_FIXTURES` for the original engineering fixtures.
The training receipt hashes the restored training lock (`VBENCH_TRAINING_LOCK`
can override its location), separately from inference dependency management.
Raw corpus and fixture regeneration needs those original inputs; the release
already includes the frozen final train/dev records for retraining.

Subject uses the
[official720 v9 protocol](../configs/subject-repair/stability_official720_protocol_v9.json),
its frozen manifest, official class map and
[v9 corruption settings](../configs/subject-repair/stability_official720_corruption_v9.json).
Build SegFormer construction masks, then quarter/full interventions with
`build_region_discrimination --gaussian-reference-short-side 128`; the single-frame
probe composes frames from that verified parent. Keep every candidate and
rejection, including the predeclared review exclusion. Independent scoring never
receives construction masks. [Final-cohort report](counterfactual-reports/subject_official_extension_20260920.md).

Background uses
[`construction_test_v1.json`](../configs/background-repair/construction_test_v1.json)
and [`holdout_protocol_v1.json`](../configs/background-repair/holdout_protocol_v1.json).
Test construction is the frozen SegFormer semantic union; development GrabCut
refinement is separate and must not replace it. Verify pixels and candidate
coverage before scoring. [Heldout report](counterfactual-reports/background_holdout_20260920.md).

Dynamic retains the complete [probe → anchored → aligned recipe](../configs/dynamic-static-jitter/README.md).
`validate_vjepa_motion`, `expand_vjepa_validation` and `official_video_jitter`
preserve source selection, native 16-frame/8-FPS inputs, 8px local-coordinate
jitter and encoding controls. Parent training stages are required, while the
aligned head is the sole selected inference method. Keep failed development
gates and previously exposed prompts explicit. See the
[model guide](reproduction/DYNAMIC_ALIGNED.md) and separate
[external validation](reproduction/DYNAMIC_GENERALIZATION.md).

Object/Color protocols and vocabulary are in the
[Object/Color guide](object-color-repair.md) and `configs/four_dimension/`.
Full construction needs original media and source annotations. This cleanup
does not generate replacement research inputs or establish a fresh retraining
result. Inspect arguments with `uv run python -m scripts.semantic.<entry> --help`
or `uv run python -m scripts.counterfactual.<entry> --help`; put new artifacts in
`output/`.
