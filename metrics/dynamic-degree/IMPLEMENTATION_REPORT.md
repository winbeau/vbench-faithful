# Dynamic Degree implementation report

## Upstream and scope

- Reference: `/home/msy625/vbench1`
- Remote: `https://github.com/msy625/VBench.git`
- Branch: `master`
- Commit: `13dee903cc97e2633ed6e8f50dea61bc90717935`
- Worktree: clean at implementation time
- Submodules: none reported
- Scope: root VBench 1.0 `vbench/dynamic_degree.py`; VBench 2.0 and beta implementations are excluded

The reference directory name is not treated as a Git tag. The local `VBench1.0_paper.pdf` is invalid 404 HTML, so the definition was checked through the official arXiv HTML rendering. The paper describes RAFT, largest-5% flow strength, static/non-static classification, and the final non-static-video proportion. Code-level sampling and thresholds were confirmed from the locked source.

## Official execution path

```text
OpenCV decode and CAP_PROP_FPS
-> interval=max(1, round(fps/8))
-> adjacent sampled frames
-> official InputPadder
-> RAFT Things, 20 iterations
-> sqrt(u^2+v^2)
-> mean of largest int(H*W*0.05) pixels
-> threshold=6*min(H,W)/256
-> transition moving boolean
-> count_num=round(4*sampled_frame_count/16)
-> video boolean
-> dataset mean of booleans
```

`--vbench` keeps this behavior, including the zero-`count_num` short-sequence artifact. It loads the exact upstream RAFT class and existing Things checkpoint; no replacement optical-flow model is present.

## Repaired execution path

```text
Video + explicit prompt/override
-> OpenCV decoded TimedFrame(frame, timestamp, source index)
-> VBench-like ~8 FPS sampling
-> unchanged RAFT dense flow
-> robust partial-affine initial fit
-> geometric-inlier/residual-consistent border-band-prior candidate refit
-> apparent/global/residual flow
-> Top-5% displacement / image diagonal / dt
-> apparent/camera/residual speed
-> duration-weighted intensity and duration coverage
-> deterministic SUBJECT/CAMERA/GENERIC/BOTH/UNKNOWN routing
-> selected continuous evidence or structured abstention
```

## Three core repairs

1. Background-aware source decomposition fits translation, rotation, and uniform scale/zoom with `estimateAffinePartial2D`, produces dense global flow, and retains residual flow. After the initial robust fit, refit candidates must be geometric inliers, residual-consistent, and preferably inside an image-border band. The border-band prior is a deterministic spatial heuristic, not a semantic background mask. It uses the full border band rather than only four corners, and no model is selected by evaluating its error on the same points used to fit it. Fit parameters, candidate strategy, inlier ratio, candidate counts, residual statistics, and fallback reason are recorded.
2. Timestamp-aware normalization computes `d_norm=d/sqrt(H^2+W^2)` and `v=d_norm/dt`. Strictly increasing OpenCV timestamps are preferred; reliable FPS supplies an explicit nominal fallback. Non-finite or non-positive `dt` transitions are invalid and excluded.
3. Continuous temporal evidence uses `I=sum(v_t*dt_t)/sum(dt_t)` and `P=sum(1[v_t>tau]*dt_t)/sum(dt_t)`. The formal task-relevant result is structured intensity plus persistence; there is no default scalarization. Motion Integral is not implemented. `tau` only controls persistence.

The default `tau` converts the official pixel threshold to image-diagonals/second at nominal 8 FPS. Its provenance explicitly says it is not independently calibrated. Explicit internal thresholds require a source string.

## Boundary robustness fix

Audit records `official_count_num`, `audit_effective_count_num`, and `boundary_fix_applied` only as compatibility diagnostics. When a sampled sequence has at least one transition, the diagnostic compatibility threshold is at least one. Neither count participates in task evidence or scalarization. Zero valid transitions returns `insufficient_transitions`; this is an implementation robustness boundary, not a fourth core research repair.

## Prompt routing

- SUBJECT: residual/content evidence
- CAMERA: global/camera evidence
- GENERIC: apparent evidence
- BOTH: structured camera and residual evidence; no blind sum and no scalar score
- UNKNOWN: unresolved/abstain; never falls back to SUBJECT

Explicit `motion_target` metadata overrides the deterministic parser. The parser was checked against all 72 locked VBench Dynamic Degree prompts; all resolve to SUBJECT under the auditable action vocabulary. Missing prompts remain UNKNOWN. Prompt parsing and motion measurement are separate modules.

For SUBJECT, CAMERA, and GENERIC, `task_relevant_motion_evidence` formally stores the selected channel's `motion_intensity` and `temporal_coverage`. BOTH keeps both channels structured and unsummed. The default public `score` is `null` because no independently validated mapping from these heterogeneous components to a benchmark scalar exists. An internal optional scalar hook is available for future calibration or ablation work, but it must declare its source and whether it was independently calibrated.

The `WITHOUT_CONTINUOUS_PERSISTENCE_AGGREGATION` ablation now means intensity-only continuous evidence: intensity remains available, coverage is omitted, and no scalar is invented. It can therefore be compared directly with Full structured intensity-plus-persistence evidence.

## Preserved, modified, and added

Preserved: RAFT architecture/checkpoint loading, 20 iterations, approximately 8 FPS sampling, dense flow, Top-5% spatial aggregation, and all Official formulas.

Modified only in Audit: decoded frames become timed frames; apparent flow becomes apparent/global/residual channels; pixel displacement becomes image-diagonal/second speed; hard temporal booleans become intensity plus persistence; prompt routing selects the task-relevant channel.

Added: affine fit diagnostics, timestamp provenance, boundary diagnostics, internal ablation enum, structured task evidence, per-video failure isolation, and shared-GPU-worker integration.

## Tests

Synthetic deterministic tests cover zero motion, translation, local motion, subject plus camera, large foreground, parallax diagnostics, rotation, zoom, low-resolution Top-5%, time/FPS normalization, real-timestamp and nominal fallback paths, invalid `dt`, intensity, 0/25/50/75/100% coverage, speed monotonicity, prompt targets and override, short-video boundary, camera-shake/source-routing contracts, structured BOTH evidence, CLI single/batch/both/run-id/GPU forwarding/interruption, locked SHA, source-extracted Official formulas, and metadata plumbing.

The Dynamic Degree suite runs 50 tests: 49 pass and the opt-in real CUDA parity test is skipped. The existing Spatial Relationship suite still runs 23 tests with one real-model skip, and the shared-core suite passes all 3 tests. The current wheels build and install in a clean temporary Python 3.10 environment; the installed console script and root-level `uv run --active --no-sync dynamic-degree --help` both work without `--package`.

The existing 21,108,000-byte checkpoint successfully constructs the locked RAFT model and loads its state dict on CPU under PyTorch `2.13.0+cu130`; no CPU video inference was performed. A real 565,297-byte MP4 decoder smoke test produced 39 samples from 77 frames at 16 FPS using interval 2 and monotonic OpenCV timestamps. This validates decoding/timestamp plumbing, not RAFT scores.

The opt-in real parity test requires `VBENCH_AUDIT_REAL_DYNAMIC_PARITY=1`, `VBENCH_AUDIT_REAL_DYNAMIC_PARITY_VIDEO`, `VBENCH_AUDIT_RAFT_WEIGHT`, and CUDA. It compares the extracted Official result with direct upstream `DynamicDegree.infer()` on the same model/video/device.

## Known limitations and unrun work

- CUDA and NVML are unavailable in the current WSL session, so real RAFT parity, single-video inference, batch inference, and multi-GPU execution are not claimed.
- Partial affine cannot fully model parallax, rolling shutter, independently moving background, or arbitrary perspective motion. Diagnostics expose fit degradation and fallback.
- The border-band prior is deterministic but not semantic segmentation. A foreground that dominates the initial robust fit can still contaminate the affine estimate; degraded inlier/residual diagnostics expose this limitation. No SAM, VLM, tracking, or replacement motion model was added.
- The default persistence threshold is traceable but uncalibrated. No current counterfactual set was used to tune it, and no final benchmark calibration is claimed.
- No default scalar mapping from intensity plus persistence is calibrated. Consequently Audit emits structured evidence and `score=null` unless an explicitly sourced optional scalar hook is supplied.
- Static plus camera shake, Subject x Camera 2x2, FPS resampling, subject speed, motion coverage, real T2V, and human preference experiments from the old repository were not rerun. Their historical numbers are not embedded in code and are not new evidence.
- Human pairwise agreement, ranking accuracy, Kendall tau, and Spearman remain future evaluation work.
- Root `uv.lock` was absent before implementation and remains intentionally ungenerated until the workspace-wide model dependency combination is validated.

## Fourth repair: lag-calibrated time normalisation

Added after the counterfactual FPS-invariance run (`dynamics_degree`, family
`fps_resampling`, 40 bases / 160 clips) showed the repair was not actually
frame-rate invariant. The full critique, including why the composite CPA could
not have reported it and what the released clips do and do not explain, is in
`docs/counterfactual-reports/dynamics_degree.review.md`; the re-measured report
is `docs/counterfactual-reports/dynamics_degree.md`.

### What the archived data said

| backend | 8 fps | 6 fps | 4 fps | 2 fps | ratio profile | fitted `s ~ dt**p` |
|---|---:|---:|---:|---:|---|---:|
| Official raw top-5% flow (px) | 16.5108 | 19.7201 | 24.2540 | 32.8744 | 1 / 1.19 / 1.47 / 1.99 | **p = +0.491** |
| Archived Repair (`d/dt`, diag/s) | 0.2484 | 0.2230 | 0.1864 | 0.1282 | 1 / 0.90 / 0.75 / 0.52 | **p = -0.481** |

The two rows are mirror images, and `official / (dt * repair)` is constant at
0.015-0.016 across all four rungs: the archived "repair" was the Official flow
magnitude divided by `dt`, nothing more. Dividing by `dt**1` is exact only when
displacement grows linearly with the lag (constant-velocity motion). An
invariance family has every pair expected to tie, so the counterfactual's
tie-margin CPA is blind to the sign of the level effect: the reported `+0.0111`
was noise.

The aggregate `+0.491` is also a heavy-tailed mean-of-means statistic. On the 10
dev bases the per-clip across-rung exponent averages **+0.706** and correlates
**+0.925** with the within-clip RAFT exponent fitted from the clip's own lag-1/2/4
displacements (and `+0.935` with `log(s_fps2 / s_fps8)`).

### What was measured on the real clips

All 160 clips were scored with the patched backend on the scoring host (5
physical GPUs, one logical `cuda:0` each). The displacement law is sublinear in
`dt` and the fitted exponent is itself lag-window dependent, which is why a
per-clip calibration under-corrects:

| rung | frames | mean measured within-clip exponent |
|---|---:|---:|
| 8 fps | 16, 27 | +0.618 |
| 6 fps | 12, 20 | +0.557 |
| 4 fps | 8, 14 | +0.512 |
| 2 fps | 4, 7 | +0.480 |

Applying an exponent `alpha` to the recorded per-clip displacements gives:

| alpha | test level profile (8/6/4/2 fps) | test slope | test `fps2/fps8` |
|---:|---|---:|---:|
| 0 (Official) | 1 / 1.196 / 1.437 / 1.911 | +0.458 | 1.911 |
| **0.5 (shipped default)** | **1 / 1.042 / 1.040 / 0.996** | **-0.011** | **0.996** |
| 1 (archived repair) | 1 / 0.902 / 0.736 / 0.498 | -0.511 | 0.498 |

The exponent that flattens the ladder is `+0.518`, base-cluster bootstrap 95% CI
`[+0.417, +0.651]`; both `0` and `1` fall outside it. Calibrated on the 10 dev
bases alone it is `+0.624`, which leaves a residual test slope of `-0.135`: the
dev split is too small to calibrate this constant, so `alpha = 0.5` ships as an
a-priori diffusive constant (`displacement ~ sqrt(dt)`) that the CI supports.

### What changed

- `AuditConfig.lag_exponent_mode` defaults to `LagExponentMode.FIXED` with
  `default_lag_exponent = 0.5`, and the resolution order is: ablation, explicit
  `lag_exponent` (requires `lag_exponent_source`), `BALLISTIC` mode (`dt**1`,
  the archived behaviour), `MEASURED` mode (each clip's own fit), else the fixed
  default. `time_normalization_exponent_source` records which applied.
- `LagScalingEvidence` is still measured and emitted for every clip (`lag_set`
  default 1/2/4 sampled frames, capped by `max_lag_frames`): the fitted exponent,
  the per-lag chord and path displacements, `straightness = chord/path`, pair
  counts and the fit residual. It is a diagnostic, not the normaliser, because
  the fitted value moves with the lag window the rung exposes.
- `derive_motion_threshold` now takes the applied exponent and re-expresses the
  Official pixel threshold as `official_px / diagonal / reference_lag**exponent`
  (`AuditConfig.reference_lag_seconds`, default 0.125). `aggregate_channel`'s
  coverage fraction and the per-transition significance flags use the same
  `d / dt**alpha` domain, so the static/moving decision no longer mixes a pixel
  threshold with a per-frame displacement. At the reference lag the two domains
  coincide by construction, and `MotionThreshold.exponent` records which was used.
- `time_normalization.intensity_units` states the resulting unit, which is
  `image_diagonals_per_second_pow_alpha` when `alpha != 1`.

### Verification

Unit level: `tests/test_lag_scaling.py` builds trajectories whose lag law is
known and samples each at two rates, and `tests/test_temporal_aggregation.py`
pins the threshold/coverage domain. With the law measured from the real clips:

| trajectory | archived `d/dt`, 2 fps / 8 fps | `d/dt**0.5`, 2 fps / 8 fps |
|---|---:|---:|
| ballistic (`p = 1`) | 1.0000 | 1.0000 (under `BALLISTIC`) |
| measured real law (`p = 0.5`) | **0.4931** | **1.0000** |

End to end: 160/160 counterfactual clips scored on the H100 host, and the
recorded scores reproduce the offline `mean_displacement / dt**0.5` calibration
to `1.7e-16`. Coverage on the test split moves `0.501 -> 0.333` under
`alpha = 1` but `0.501 -> 0.533` under `alpha = 0.5`.

### What this does not deliver

- **Per-clip invariance is not achieved.** After the fix the per-base
  `fps2/fps8` ratio still has median `1.284`, IQR `[0.807, 1.537]`, and only 20%
  of bases sit within ±20% of 1. The fix makes the reported *dataset aggregate*
  frame-rate invariant — which is how VBench defines dynamic degree — but
  individual clips remain incomparable across frame rates.
- The two construction defects in the counterfactual ladder remain in the
  released dataset: the coarse rungs sample a shorter span of the source
  trajectory (fps2 covers 80% of a 16-frame 8 fps source), and the 10 fps GIF
  rungs drift in container duration by up to 6.1%.
- The `~sqrt(dt)` law is measured, not explained. Two independent estimators are
  both sublinear (RAFT `+0.819`, Farneback `+0.608` on the same frame pairs), and
  the median `straightness` is `0.793` at lag 2 and `0.574` at lag 4 across all
  160 clips, so the trajectories genuinely turn at these lags and the estimator
  is not the whole story; separating the two completely would need a
  per-pixel chord-vs-path field comparison against a non-saturating estimator.
