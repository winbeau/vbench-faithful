# Nine-repair performance and sixteen-package acceptance

On `h100-server`, the controlled nine-repair fixture went from **220.38 s to 181.63 s**: **38.76 s saved, 17.59% less wall time, 1.21× speedup**. All **19 per-video scores are exactly equal**. This is one paired measurement on a small installation fixture, with the same single physical GPU 3, inputs, weights, precision and generation settings. It does not establish large-dataset throughput.

The [machine-readable receipt](h100-performance-20261008.json) contains the exact code and upstream SHAs, input hashes and media durations, GPU identity, per-dimension and stage times, profile hashes, coverage and comparisons. All outputs are under `/root/wenbiao_zhao/vbench-repair-eval-20261008/`.

## Where the time went

The baseline controller profile attributed **22.05 s** to serial full-file asset verification, including **16.93 s** in SHA-256 updates and **5.25 s** in reads across controller hashing. Object Class and Color independently hash the full Qwen3-8B base again while assembling prompt-compiler provenance. Those hashes are CPU/file work before model inference. Six repaired dimensions also start independent Qwen processes and load the base model; that startup cost remains.

| Nine-repair dimension | Before | After |
| --- | ---: | ---: |
| Scene | 27.91 s | 25.61 s |
| Human Action | 21.11 s | 21.38 s |
| Object Class | 37.11 s | 24.95 s |
| Subject Consistency | 9.23 s | 9.37 s |
| Background Consistency | 10.82 s | 10.06 s |
| Dynamic Degree | 8.69 s | 9.00 s |
| Spatial Relationship | 23.03 s | 23.39 s |
| Multiple Objects | 23.22 s | 23.19 s |
| Color | 34.89 s | 26.27 s |
| Full-file asset preflight | 22.05 s | 6.07 s |
| Total wall time, including all setup | **220.38 s** | **181.63 s** |

In the optimized run, Object Class/Color compilation takes 13.65/14.00 s and their visual scoring takes 11.30/12.27 s. Individual environment preparations take roughly 0.27–0.40 s in these stages. uv environment creation is a small part of the measured cost. Stage worker time includes interpreter startup, model loading, hashing and inference; it is not a GPU-kernel timer.

Earlier four-GPU acceptance also showed unbalanced fixed lanes: approximately 74.6, 33.1, 62.3 and 33.7 seconds of dimension work. The scheduler now gives the next configured dimension to the next available GPU. Multi-GPU scheduling and concurrent CUDA probes have CPU concurrency/regression tests; this follow-up used only GPU 3 because GPUs 0–2 were occupied by unrelated jobs. A new four-GPU speedup was not measured.

## Changes and equivalence

- Hash independent files using four CPU threads, both in controller preflight and Qwen compiler provenance. Every selected file is still read in full on each invocation; unchanged size/mtime cannot bypass content verification. The verifier preserves pin order and the first relevant error.
- Probe selected GPUs concurrently and dispatch dimensions from one queue, retaining one active dimension per GPU and serial stages/backends within a dimension.
- Record preflight, environment and worker durations in JSON and terminal stage events. Cache hits explicitly have zero worker time.

No scoring formula, sampling, model weight, attention implementation, precision, generation limit or aggregation was changed. The two profiled repair runs match all 19 scores, prompt compilation records and diagnostic content. Diagnostic comparisons verify referenced artifact hashes and normalize output/stage paths. The final sixteen-package version adds four pinned upstream source identities to provenance; those additions were independently checked before comparing the remaining diagnostics. It also matches all 19 repair scores exactly.

## Final-version acceptance

Code `13ce34ca53890a7b74b330ea3f433747408fad6e` has **16 independent uv packages** under `metrics/`, each with a CLI and tests. Aesthetic Quality, Imaging Quality, Temporal Flickering and Appearance Style now own official adapters used by the real visual worker. Their YAML CLIs fix the dimension and reuse the shared evaluator's environments, caches, coverage checks and native official aggregate.

| Run | Effective tasks | Records / unique media hashes | Cache hits | Single-H100 wall time |
| --- | --- | --- | --- | ---: |
| `perf-origin16` | 16 Origin | 16 / 11 | 0 / 16 | 130.79 s |
| `perf-repair-cache-cold` | 9 Repair | 19 / 16 | 0 / 9 | 176.52 s |
| `perf-repair-cache-warm` | 9 Repair | 19 / 16 | 9 / 9 | 13.22 s |

These three runs use the ordinary controller without cProfile. The cold run creates immutable stage artifacts; the warm run reuses them and performs no model inference. Four additional package-entry runs each selected exactly one official dimension and reused its verified artifact: Aesthetic 3.35 s, Imaging 2.66 s, Flickering 2.54 s, Appearance 2.81 s. All 13 warm task comparisons retain identical per-video results and native aggregates, with every stage explicitly cached.

The 16 official results were compared with the earlier installation acceptance: **15 dimensions are exact; Motion Smoothness differs by 3.56×10⁻⁹** in its score and aggregate, within the recorded `1e-6` tolerance. That reference used four GPUs. This comparison does not establish parity against the historical E0 research cohort.

Repair records comprise eight original/counterfactual pairs and one original plus two counterfactual Dynamic clips. Some records share media; they are 19 input identities, not 19 independent videos. The JSON retains every media duration, including `null` for GIFs without positive explicit frame delays, joined to the earlier FFprobe/GIF measurements by input ID and verified media hash.

## Runtime and commands

Both controlled runs used NVIDIA H100 80 GB HBM3, physical GPU 3, UUID `GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30`, driver `590.48.01`. Root Python is 3.11.14; visual workers share Python 3.10.20 / torch 2.5.1+cu121, semantic workers Python 3.11.14 / torch 2.7.1+cu126. The upstream is `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`. Dependencies, weights, thin environments and OS file caching were already available. Download/restoration time is excluded. No unrelated GPU process was stopped.

From the remote `code/` directory, baseline `01c7da5` and optimized `d4651fb` respectively used:

```bash
uv run --locked python -m cProfile -o ../perf-baseline-controller.prof \
  scripts/eval.py --config ../repair.yaml --gpus 3 --output ../perf-baseline-cold --no-reuse
uv run --locked python -m cProfile -o ../perf-optimized-controller.prof \
  scripts/eval.py --config ../repair.yaml --gpus 3 --output ../perf-optimized-cold --no-reuse
```

The same cProfile wrapper was applied to both controllers; model subprocesses were not profiled. Stage artifact reuse was disabled in both. The final-version runs used `uv run --locked python scripts/eval.py` with `--config ../origin.yaml` or `../repair.yaml`, `--gpus 3`, and the fresh output names in the table. The four package CLIs used `uv run --locked <package> --config ../origin.yaml --gpus 3 --output ../perf-cli-<dimension>`.

`uv lock --check`, `uv sync --locked --group test`, all 16 CLI help checks and the CPU torch-overlay suite passed: **1,427 passed, 3 skipped**. Tests cover full content verification with preserved timestamps, error precedence, concurrent probes, busy-GPU scheduling, failure denominators, zero worker time on cache hits, package isolation and native upstream reducers. All 37 frozen files and six pre-existing untracked research files checked against the original snapshot remain byte-identical.

## Commit record

| Commit | Change |
| --- | --- |
| `d4651fb` | Parallel full-byte verification, GPU queue/probes, stage timing; controlled repair benchmark. |
| `13ce34c` | Four missing metric packages, official adapters, YAML CLI entries, uv lock and sixteen-package tests; final GPU acceptance. |

Both implementation commits were pushed to `origin/winbeau`. The following documentation/index commit records the measurements without changing the tested scoring code.
