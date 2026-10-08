# Unified VBench evaluation

The fast controller runs the existing paper workers. It supports all 16 VBench 1.0 dimensions, with the nine selected repairs in `configs/reproduction/paper-methods.json` and seven separately identified accelerated implementations. It does not alter the nine repair formulas.

```bash
uv sync --locked
uv run python scripts/eval.py --config configs/eval.yaml --plan
uv run python scripts/eval.py --config configs/eval.yaml
./scripts/eval.sh --config configs/eval.yaml --backend origin --dimensions all
./scripts/eval.sh --config configs/eval.yaml --backend both --dimensions paper --gpus 0 1
```

First restore the pinned model runtimes and assets using [the recovery guide](reproduction/CONTAINER_RESET.md). Change `input` and `assets` in the example YAML to your manifest and recovered `assets.json`. `--plan` validates input media identity and routing without loading models or creating environments; it does not certify runtime assets or CUDA.

All sixteen dimensions are registered uv workspace packages under `metrics/`. The four packages completing the workspace expose YAML-based commands:

```bash
uv run aesthetic-quality --config configs/eval.yaml --backend origin
uv run imaging-quality --config configs/eval.yaml --backend origin
uv run temporal-flickering --config configs/eval.yaml --backend origin
uv run appearance-style --config configs/eval.yaml --backend origin
```

These commands fix their own dimension regardless of the YAML dimension selection, and accept the controller's other options, including `--plan`, `--gpus`, and `--no-reuse`. With `backend: ours` (alias `repair`) they use their accelerated implementations. With `backend: official` (alias `origin`) they run the pinned original functions. Both routes preserve the native aggregate, and these dimensions remain outside the nine paper repairs. Existing twelve package CLIs keep their legacy research interfaces; use the unified controller for the selected nine paper repairs.

## Configuration

One flat YAML mapping is sufficient. No inheritance, executable resolvers, anchors, aliases, or nested configuration objects are supported. Unknown and duplicate keys are errors.

| Key | Default / meaning |
| --- | --- |
| `version` | `1` |
| `input` | Required JSON/JSONL manifest or VBench full-info list with `video_list` |
| `assets` | Required fixed-runtime `assets.json` |
| `output` | New `../output/eval/<UTC timestamp>` directory |
| `video_root` | Optional override for relative media paths |
| `backend` | `ours`; also `official` or `both`; legacy aliases `repair` / `origin` |
| `dimensions` | `all`; also `paper` or a list of canonical dimension names |
| `gpus` | `[0]`; nonnegative visible indices or complete GPU/MIG UUIDs |
| `cache_dir` | `~/.cache/vbench-repair` |
| `env_dir` | Repository `.venvs/metrics` |
| `reuse` | `true`; `--no-reuse` forces inference |

YAML paths resolve relative to the configuration file. CLI path overrides resolve relative to the current working directory. `~` and `${ENV_NAME}` expand in paths; an undefined environment variable is an error. Media paths inside the JSON/JSONL resolve relative to that manifest unless `video_root` is set. A minimal custom record is:

```json
[{"id":"clip-1","video":"videos/clip.mp4","prompt":"a beach","dimensions":["scene"],"auxiliary_info":{"scene":{"scene":{"scene":"beach"}}}}]
```

Use the original metadata schema for official prompt-conditioned dimensions. For standard suites, `scripts/prepare_vbench_inputs.py` preserves the bundled official metadata. Input dimensions default to the paper nine when omitted; to evaluate all 16, provide dimension membership explicitly. Requested dimensions with zero matching inputs are errors. Subject Repair requires `subject_en`. Dynamic aligned-v1 accepts native square videos with 16 frames at 8 FPS; it returns a failure for unsupported videos instead of silently changing the protocol.

`official + all` runs 16 original tasks. `ours + all` runs nine paper repairs and seven accelerated tasks. `both + all` runs 32 tasks, comparing every dimension. The seven accelerated dimensions are Motion Smoothness, Temporal Flickering, Aesthetic Quality, Imaging Quality, Temporal Style, Overall Consistency, and Appearance Style. Output keys name the actual `origin`, `repair` or `accelerated` implementation; there are no official fallbacks.

Each accelerator passed per-video checks on the fixed same-32 cohort with `abs(candidate - official) <= max(1e-6, 0.01 * abs(official))`, complete input identity and native-aggregate checks. This is numerical agreement on the tested cohort, not perceptual quality or universal equivalence. See the [sequential optimization record](plans/2026-10-08-speed32-optimization.md).

## Environments and reuse

Root uv manages the lightweight controller, shared infrastructure, configuration and logging. Each project dimension gets its own uv-created environment at `.venvs/metrics/<dimension>/<visual|semantic>/<fingerprint>`. Model dependencies are shared read-only from the existing pinned visual Python 3.10 and semantic Python 3.11 runtimes. Mixing these ABIs in the root environment would break compiled dependencies. The thin environments have distinct prefixes and do not inherit other metric source paths; this is dependency/process isolation, not a filesystem security sandbox.

Environment fingerprints bind the source interpreter, ABI, package metadata and startup paths. Preparation is locked and atomic. Do not mutate the shared runtimes while evaluation is running. Weight files are shared through the asset paths; uv's own package cache remains available for development installs.

Stage caching covers official scoring, accelerated scoring, visual evidence, prompt compilation and repaired scoring. Keys bind the ordered inputs (including filenames, prompts, metadata and media SHA-256), selected asset hashes, code, environment and GPU runtime. Every cached artifact is hashed on reuse. Failed or incomplete stages are not published as hits. Identical tasks can reuse work when the selected dimensions or backend mode change, as long as that task's inputs and dependencies stay the same. This version does not combine partially overlapping batches or share detector evidence across different preprocessing protocols.

Outputs link to immutable cache generations so absolute diagnostic references remain valid. Keep the cache while these runs are needed. `--no-reuse` writes independent stage directories directly into the new output. Cache reuse is recorded separately from original artifact provenance and never reported as fresh inference.

Within each new evaluation, GRiT also shares identical model-input frames through `model-inference/`. Object Class, Spatial Relationship and Multiple Objects use the same pinned ObjectDet protocol; Color uses a separate DenseCap namespace. Keys bind the actual frame bytes, shape, dtype, weights, source and GPU runtime. Failed outputs are never shared and stored outputs are checked for corruption. Each dimension still runs in its own environment and applies its own scoring rules. `--no-reuse` permits this sharing inside the current run, while preventing reuse of any previous run's inference. Worker logs and evidence receipts report the frame hits/misses. Unused GRiT display rendering is skipped. Inside one ObjectDet prediction, equivalent ROI/text heads reuse an independently copied result, guarded by shared model, decoder settings and identical feature/proposal objects. Color keeps both distinct heads; raw evidence, preprocessing and tuple conversion retain their original semantics. Logs count same-frame head reuse separately.

Temporal Style and Overall Consistency can also share native eight-middle-frame ViCLIP video feature batches within this run. Keys include transformed input bytes, batch size, checkpoint digest and runtime; text queries and scoring stay separate. Receipts distinguish fresh visual inference from current-run feature hits. Overall's measured speed benefit depends on this reuse; a standalone run is not proven faster.

Official mode uses a separate `official/visual/<fingerprint>` uv environment with no metric package sources. One persistent process per selected GPU calls the original VBench compute APIs with the supplied metadata and native reducers, avoiding repeated interpreter/CUDA startup. It does not apply project model optimizations. Each request still has independently validated per-video coverage and its own stage-cache identity. This is the original compute path, not the upstream filename-based input builder.

Multiple selected GPUs run independent dimensions concurrently. Each available GPU takes the next dimension from the configured order when its current dimension finishes. Each worker sees a single device as logical `cuda:0`; backends and stages within a dimension run serially. Integer IDs are interpreted under the inherited `CUDA_VISIBLE_DEVICES` mask. The controller rejects multi-rank launches, probes CUDA allocation, and stops only its own worker process groups on interruption.

Asset verification reads and hashes every selected file on every invocation, including cache hits. Four CPU threads hash independent files concurrently; Qwen prompt compilation also hashes independent base-model files concurrently. GPU probes run in parallel in separate processes. These changes retain full content verification, model precision, frame sampling, generation parameters and scoring formulas. Project dimensions load models in isolated processes. Temporal/Overall feature reuse uses the controller's freshly verified asset digest; standalone calls without that receipt rehash the checkpoint. Temporal Flickering is CPU-only and loads neither Torch nor a model.

## Results and logs

The terminal displays stage events, progress and a final dimension/backend table. Non-interactive logs contain plain events without terminal animation. Each run contains:

- `plan.json`: resolved config, routes, media/config/source/model identities and GPU runtime.
- `summary.json`: actual backend, full input denominator, coverage, score, timing and cache status.
- `events.jsonl`: timestamped controller events, preflight breakdown and stage durations.
- `<dimension>/<backend>.json`: per-video results and this run's execution provenance.
- `<dimension>/<stage>/worker.log`: model output or a link to the shared official process log, with stage artifacts alongside.
- `official-<device-id>.log`: original VBench output and tracebacks across that GPU's requests.

Successful records require finite scores; failures retain null and the original input identity. An incomplete task has `score: null`; `observed_subset_mean` is only a diagnostic. Official and accelerated scores preserve upstream aggregation, including MUSIQ's aggregate scale. Exit codes are 0 for complete success, 1 for incomplete scoring, 2 for setup/configuration errors, and 130 for interruption.

`plan.json` and `summary.json` expose `timing_seconds` for inputs, asset verification, source verification, GPU probes and total time before dispatch. Each completed stage event and per-run execution receipt records `elapsed_seconds`, `environment_seconds` and `worker_seconds`. Worker time includes process startup, model loading and inference, rather than GPU kernels alone. Cached stages have zero worker time; overlapping GPU lanes must not be summed to claim wall time.

See the [H100 performance report](validation/h100-performance-20261008.md) for the controlled nine-repair comparison, remaining startup costs and final sixteen-package GPU acceptance.

The existing `scripts/evaluate_vbench.py` remains the strict paper interface (`repair` restricted to nine). Legacy per-metric CLIs retain their research meanings. The seven accelerated implementations do not promote historical repair candidates into the paper.
