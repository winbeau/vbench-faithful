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
