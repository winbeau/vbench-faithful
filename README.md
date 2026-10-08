<div align="center">

# Aligning VBench Scores with Their Intended Targets:<br>Toward Faithful Video Evaluation

**Wenbiao Zhao<sup>1,*</sup>, Zichao Nie<sup>2,*</sup>, Shuyuan Meng<sup>1</sup>,<br>Meng Wang<sup>2</sup>, Haiwei Xue<sup>3</sup>, Panpan Zheng<sup>1,†</sup>**

<sup>1</sup> Xinjiang University &nbsp; <sup>2</sup> Tsinghua University<br>
<sup>3</sup> The Hong Kong University of Science and Technology

<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding author

[Data](https://huggingface.co/datasets/xju-arlab/vbench-repair) · [Models](https://huggingface.co/xju-arlab/vbench-model) · [Quick Start](#quick-start) · [Reproduction](docs/reproduction/README.md) · [Experiment Index](docs/EXPERIMENT_INDEX.md)

</div>

**VBench Repair** accompanies our manuscript on the measurement validity of video evaluation. We audit nine VBench dimensions, trace unintended score responses to their implementation, and develop targeted repairs. The evaluator supports all **16 official VBench 1.0 dimensions**, with the **nine paper-selected repairs** available through one configuration.

[![Overview of target–evidence misalignment and repair strategies](docs/assets/fig2.png)](docs/assets/fig2.pdf)

*Overview of target–evidence misalignment and repair strategies. [PDF](docs/assets/fig2.pdf) · [Figure provenance](docs/assets/README.md)*

## Highlights

- **Three failure mechanisms.** Target Substitution replaces the intended property with an inequivalent proxy; Nuisance Entanglement mixes relevant and irrelevant evidence; Evidence Collapse discards information needed to distinguish target states.
- **Explicit behavioral tests.** Controlled counterfactuals separately test sensitivity to target changes and stability under irrelevant variation, with construction eligibility, missing scores, and paired denominators retained.
- **Repairs at the responsible stage.** Semantic target parsing, subject/background isolation, a frozen V-JEPA 2.1 motion encoder with a learned readout, and signed spatial geometry address different parts of the scoring pipeline. Text parsing alone cannot recover direction discarded by an unsigned geometric backend.
- **One evaluation command.** YAML configuration, automatic per-dimension environments, isolated model processes, shared model assets, and validated artifact reuse support repeatable evaluation without manually installing each dimension.

## Installation

The controller uses **Python 3.11.14** and the checked-in `uv.lock`. Model inference uses the separately restored, pinned **Python 3.10.20 visual** and **Python 3.11.14 semantic** runtimes.

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, Linux x86_64 with Ubuntu 24.04-compatible system libraries, FFmpeg, and a working NVIDIA driver. Reserve approximately **100 GB** for runtime archives, extracted dependencies, model weights, and verification outputs. Qwen3-8B inference needs roughly **20–24 GB of free GPU memory** on the selected device.

```bash
# Repository access is required.
git clone -b winbeau git@github.com:winbeau/vbench-repair.git
cd vbench-repair
uv sync --locked

# Ubuntu system dependencies; skip packages already installed.
sudo apt-get update
sudo apt-get install -y python3-venv ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libgomp1
```

Restore the inference runtimes and selected model assets once:

```bash
python3 -m venv output/bootstrap
output/bootstrap/bin/pip install 'huggingface_hub==1.32.0' requests
output/bootstrap/bin/python scripts/restore_paper_runtime.py \
  --output output/runtime \
  --downloads output/runtime-downloads \
  --allow-official-fallback
```

This creates `output/runtime/assets.json`, verifies the pinned archive and model hashes, and restores the selected LoRA adapters, Qwen3-8B base model, Dynamic scoring head, and visual assets. Downloads first use the configured HF mirror; the flag permits fallback to the official endpoint. If model downloads are interrupted after extraction, repeat the same command with `--resume`. Keep the runtime on an executable filesystem.

The evaluator then creates a small `uv` environment for each required dimension and runtime role automatically. These environments reuse the matching restored dependencies and shared weights; visual and semantic Python ABIs remain separate. Root `uv sync` prepares orchestration and development dependencies; the restoration step supplies the inference stack.

See the [runtime recovery guide](docs/reproduction/CONTAINER_RESET.md) for archive revisions, recovery details, and representative GPU installation checks.

## Quick Start

### 1. Prepare an input manifest

For videos generated from the standard VBench prompts, use filenames `prompt-0.mp4` through `prompt-4.mp4` (original GIF inputs are also supported where the selected method permits them):

```bash
uv run python scripts/prepare_vbench_inputs.py \
  --video-dir /absolute/path/generated-videos \
  --output output/inputs.json
```

The command attaches official dimension metadata and frozen subject annotations. An incomplete standard suite is rejected by default. For an intentional partial run, add `--allow-missing` and select only the dimensions present in your inputs when evaluating.

Custom inputs can provide `id`, `video`, `prompt`, `dimensions`, and the dimension's required `auxiliary_info`. A one-line JSONL record looks like this:

```json
{"id":"sample-1","video":"clip.mp4","prompt":"a white car","dimensions":["color"],"auxiliary_info":{"color":{"color":"white","object":"car"}}}
```

Media paths are relative to the input manifest unless `video_root` is supplied. Subject Repair additionally requires an explicit `subject_en`; it does not infer the subject from an arbitrary prompt. Preserve standard filenames for official Human Action, whose original target lookup depends on the filename.

### 2. Configure the evaluation

Edit [`configs/eval.yaml`](configs/eval.yaml). A complete example is:

```yaml
version: 1
input: ../output/inputs.json
assets: ../output/runtime/assets.json
output: ../output/eval/my-run
backend: ours
dimensions: all
gpus: [0]
cache_dir: ~/.cache/vbench-repair
env_dir: ../.venvs/metrics
reuse: true
```

Paths in YAML are relative to the YAML file's directory; CLI path overrides are relative to your current directory. `~` and `${ENV_NAME}` are supported. Omit `output` to create a fresh timestamped directory under `output/eval/` when using the supplied configuration. Explicit output directories must be new.

`gpus` accepts visible GPU indices or complete GPU/MIG UUIDs. For example, with `CUDA_VISIBLE_DEVICES=4,5`, `gpus: [0, 1]` selects those two visible devices. Each worker sees one device as `cuda:0`.

### 3. Inspect and run

```bash
# Validate input media and inspect the selected methods; no model inference.
uv run python scripts/eval.py --config configs/eval.yaml --plan

# Run the configured evaluation.
uv run python scripts/eval.py --config configs/eval.yaml

# Equivalent shell entry point.
./scripts/eval.sh --config configs/eval.yaml
```

Useful overrides:

```bash
# All 16 official dimensions.
./scripts/eval.sh --config configs/eval.yaml \
  --backend official --dimensions all --output output/eval/origin

# Compare Origin and Repair on the nine paper dimensions.
./scripts/eval.sh --config configs/eval.yaml \
  --backend both --dimensions paper --gpus 0 1 --output output/eval/paired

# A selected dimension, forcing fresh computation instead of artifact reuse.
./scripts/eval.sh --config configs/eval.yaml \
  --dimensions color --no-reuse --output output/eval/color-fresh
```

By default, `backend: ours` runs the **nine selected repairs plus seven accelerated dimensions**. `backend: official` runs pinned original VBench in one process per selected GPU. `backend: both` compares all 16 dimensions (32 tasks). Legacy names `repair` and `origin` remain aliases for `ours` and `official`. Each result records its actual `repair`, `accelerated` or `origin` implementation.

The seven accelerators passed same-32 per-video checks against official scores: absolute error must be at most `max(1e-6, 0.01 × abs(official_score))`. This measures numerical agreement on the tested cohort, not perceptual quality or a guarantee for arbitrary inputs. [Optimization records](docs/plans/2026-10-08-speed32-optimization.md).

The output contains `plan.json`, `summary.json`, per-dimension `origin.json`, `repair.json` or `accelerated.json`, and worker logs. Cache receipts identify reused stages. Failed, unsupported, or omitted scores remain `null`; incomplete coverage does not produce a complete-population mean. Official dataset aggregation is preserved, including dimension-specific scales and weighting.

## Supported Dimensions

`--backend official` selects the original VBench implementation for every row. The table describes the default route under `--backend ours`; `--dimensions paper` selects the nine repaired rows.

All **16 dimensions have independent packages under `metrics/`**. The four packages added to complete the workspace are [Aesthetic Quality](metrics/aesthetic-quality/), [Imaging Quality](metrics/imaging-quality/), [Temporal Flickering](metrics/temporal-flickering/), and [Appearance Style](metrics/appearance-style/). Their package CLIs use the same YAML, cache and isolated workers as the unified evaluator:

```bash
uv run aesthetic-quality --config configs/eval.yaml --backend origin
uv run imaging-quality --config configs/eval.yaml --backend origin
uv run temporal-flickering --config configs/eval.yaml --backend origin
uv run appearance-style --config configs/eval.yaml --backend origin
```

Each command fixes its own dimension; YAML `dimensions: all` does not expand a dimension-specific command. Use `scripts/eval.py` for a multi-dimension run. The other twelve package CLIs retain their documented research interfaces.

| Dimension | CLI key | Default route (`ours`) |
|---|---|---|
| Scene | `scene` | **Repair:** Qwen3-8B verifier over fixed Tag2Text caption evidence |
| Human Action | `human_action` | **Repair:** action parsing, canonical K400 targets, and declared synonym interface over UMT evidence |
| Object Class | `object_class` | **Repair:** prompt target compilation with case, word-form, and declared-alias normalization over GRiT evidence |
| Subject Consistency | `subject_consistency` | **Repair:** independent Mask R-CNN/MobileSAM localization, subject isolation before encoding, and DINO patch comparisons |
| Background Consistency | `background_consistency` | **Repair:** independent foreground localization, CLIP background patch pooling, and the selected calibrated frame-pair score |
| Dynamic Degree | `dynamic_degree` | **Repair:** frozen V-JEPA 2.1 ViT-B encoder and the 51,393-parameter aligned-v1 continuous scoring head |
| Spatial Relationship | `spatial_relationship` | **Repair:** ordered relation targets and direction-aware signed geometry |
| Multiple Objects | `multiple_objects` | **Repair:** variable-length entity parsing and adjacent-frame detection confirmation |
| Color | `color` | **Repair:** target-instance color binding with all sampled frames in the denominator |
| Motion Smoothness | `motion_smoothness` | **Accelerated:** Batched AMT interpolation |
| Temporal Flickering | `temporal_flickering` | **Accelerated:** Streaming CPU frame differences |
| Aesthetic Quality | `aesthetic_quality` | **Accelerated:** Batched CLIP visual tower and native aesthetic head |
| Imaging Quality | `imaging_quality` | **Accelerated:** Batched FP32 MUSIQ |
| Temporal Style | `temporal_style` | **Accelerated:** Batched ViCLIP |
| Overall Consistency | `overall_consistency` | **Accelerated:** Batched ViCLIP with current-run video feature reuse |
| Appearance Style | `appearance_style` | **Accelerated:** Batched CLIP and cached text embeddings |

The selected methods and asset identities are fixed in [`paper-methods.json`](configs/reproduction/paper-methods.json). Subject uses `hybrid / exclude / preencode_crop`; Background uses `patch_frame_calibrated` with the frozen gain of 1.75. Legacy per-dimension research CLIs and the historical eleven-dimension development scope remain documented separately; their defaults are not the paper method selection.

**Dynamic input contract:** aligned-v1 accepts the paper protocol of **16 native square RGB frames at 8 FPS**. It does not silently crop, duplicate frames, resample an incompatible clip, or recalibrate the score. The continuous output measures relative motion on the learned scale, not calibrated physical motion intensity. Rectangular full-frame adaptation belongs to the separately documented [external validation](docs/reproduction/DYNAMIC_GENERALIZATION.md).

## Reproduction and Interpretation

The [reproduction guide](docs/reproduction/README.md) distinguishes frozen evidence replay, representative model inference, full-dataset GPU reruns, and training. It links the selected weights, training inputs, initialization chain, exact configurations, and verification receipts.

- **Frozen main-table replay:** the checked-in [nine-dimension table](docs/reproduction/verified-nine-dimension-replay/main-table.csv) uses complete pairs shared by all four Origin/Repair × original/counterfactual cells. Missing pairs are retained in coverage reporting. The original Object Class and Color table uses metadata-based repairs; the CLI uses the selected LoRA repairs, whose validation scope is described in the [Object/Color report](docs/object-color-repair.md).
- **Representative inference checks:** installation verification runs real models on representative inputs. These checks do not establish a full-dataset rerun or retraining result. Cached score replay likewise does not count as fresh GPU inference.
- **Unified evaluator acceptance:** [four-H100 validation](docs/validation/h100-eval-20261008.md) completed 16 official tasks and nine repaired tasks. All 19 repaired fixture records matched the frozen expected scores exactly; the default 16-dimension combination reused all tasks successfully.
- **Performance and package acceptance:** [single-H100 follow-up](docs/validation/h100-performance-20261008.md) validates all 16 independent packages. Parallel full-file verification reduced the controlled nine-repair fixture from 220.38 s to 181.63 s with all 19 scores unchanged; the final-version warm run took 13.22 s using cached artifacts.
- **Scope of the evidence:** score stability is different from accuracy, and a stable mean does not imply every video is stable. Dynamic's 450-source experiment uses 30 previously exposed prompts, averages two perturbation seeds within each source, and retains individual failures and earlier development-gate failures.
- **Construction and model limits:** predicted masks are not segmentation ground truth; declared synonym contracts do not establish open-ended semantic generalization. Detection confirmation can remove genuine positives as well as false positives. The [Multiple Objects review](docs/counterfactual-reports/multiplt_object.review.md) records additional interpretation limits.

Historical negative results, rejected constructions, and paused experiments remain available in the [experiment index](docs/EXPERIMENT_INDEX.md) and [rejected/paused experiment guide](docs/reproduction/REJECTED_EXPERIMENTS.md). The repairs should be interpreted within those measured contracts rather than as a claim of uniform improvement across every video or benchmark.

## Development

See the [evaluation configuration and cache guide](docs/evaluation.md), the [evaluation skill](skills/vbench-eval/SKILL.md), [CONTRIBUTING.md](CONTRIBUTING.md), the [workspace plan](docs/plans/2026-09-14-workspace-refactor.md), and the [architecture notes](docs/architecture.md). Each metric owns its scoring implementation; shared input, scheduling, output, and provenance infrastructure lives in `packages/audit-core/`, and shared model adapters live in `packages/audit-models/`.

Use a new `output/` directory for experiments. `data/`, `results/`, `splits/`, and `runs/` preserve frozen research inputs and results. Model weights and external checkouts stay outside version control. Dynamic Degree and Motion Smoothness reuse official Subject Consistency video files, which does not imply shared formulas or human annotations.

## Citation

Citation information for the current manuscript:

```bibtex
@misc{zhao2026aligningvbench,
  title  = {Aligning {VBench} Scores with Their Intended Targets:
            Toward Faithful Video Evaluation},
  author = {Zhao, Wenbiao and Nie, Zichao and Meng, Shuyuan and
            Wang, Meng and Xue, Haiwei and Zheng, Panpan},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/winbeau/vbench-repair}
}
```
