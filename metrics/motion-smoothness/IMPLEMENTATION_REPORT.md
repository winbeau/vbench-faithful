# Motion Smoothness Implementation Report

## Official audit

The locked VBench 1.0 checkout is `/home/msy625/vbench1` at commit
`13dee903cc97e2633ed6e8f50dea61bc90717935`. `vbench/motion_smoothness.py`
uses OpenCV sequential decoding, takes every `max(1, round(fps / 8))` frame,
keeps even-indexed frames as interpolation inputs, and runs AMT-S from
`third_party/amt/cfgs/AMT-S.yaml` with the `amt-s.pth` checkpoint. The midpoint
is fixed at `t=1/2`; real odd frames are compared with interpolated odd frames
using RGB `cv2.absdiff` mean error. The per-video score is
`(255 - mean_error) / 255`. Dataset aggregation is an arithmetic mean of
per-video scores. The official distributed path gathers video results and
retains the same equal-weight mean. No optical-flow, FPS normalization,
temporal derivative, or tail statistic is present in this baseline.

The checkout's bundled `VBench1.0_paper.pdf` is a 404 HTML stub, so source
verification is based on the locked official Python implementation.

## Evaluation contract and failure hypothesis

Given that motion exists, Motion Smoothness should measure temporal continuity,
not motion magnitude or reconstruction difficulty. The mechanism hypothesis is
that interpolation reconstructability is not identical to temporal motion
continuity; AMT error may also reflect motion magnitude, frame interval,
occlusion, and appearance reconstruction difficulty. This is not an empirical
claim about benchmark failure.

## Repair implemented

The audit path decodes timed frames, estimates consecutive flow fields, forms
velocity `v_t = flow_t / dt_t`, aligns fields with deterministic forward
nearest-neighbour splatting, and computes acceleration and jerk-like changes.
Magnitude change is normalized by a local median velocity scale plus `eps`;
direction change is `1 - cosine(v_prev, v_curr)` only above the configured
minimum motion magnitude. Per-time discontinuity is a weighted sum of these
components, spatially summarized by robust mean/median/percentile statistics.
Temporal aggregation is:

```text
D_mean = mean(D_t)
D_tail = quantile(D_t, tail_quantile)
D_video = (1 - tail_weight) * D_mean + tail_weight * D_tail
score = exp(-D_video)
```

Defaults are `eps=1e-3`, `min_motion_magnitude=0.05`,
`magnitude_weight=0.7`, `direction_weight=0.3`, `tail_quantile=0.90`, and
`tail_weight=0.25`. They are configurable engineering defaults, not tuned
values. A sequence with fewer than three motion fields returns a finite score
of `1.0` with an explicit diagnostic status.

## Difference from Dynamic Degree

Dynamic Degree measures how much motion exists using flow magnitude and a
thresholded Boolean decision. Motion Smoothness measures how the flow state
evolves over time using normalized derivatives and direction continuity; it
does not threshold motion amount.

## Compatibility and files

The repository already contained the `指标/motion_smooth` project (package
`motion_smoothness`), so the implementation extends that established path
instead of introducing a parallel `指标/motion_smoothness` project.

`official` remains a wrapper around `vbench.compute_motion_smoothness` with the
original AMT config/checkpoint contract. `audit` reuses the existing
Dynamic Degree RAFT adapter and timed OpenCV decoding. `audit-core` changed: no.

Diagnostics include FPS, frame and flow counts, dt statistics, motion and valid
pixel ratios, mean/tail/direction discontinuity, `D_video`, final score, and
per-time `D_t`, magnitude, and direction components.

## Tests and limitations

Mechanism-level tests cover constant velocity, magnitude disentanglement,
gradual acceleration, abrupt/duplicate-frame changes, direction reversal,
time normalization, alignment, tail aggregation, static/short sequences,
official source contracts, and CLI parsing. Real AMT/RAFT checkpoint parity is
not run in unit tests because it requires heavyweight weights and CUDA.

Known limitations are optical-flow noise, occlusion and articulated motion,
possible spatial bias from nearest-neighbour alignment, provisional
hyperparameters, and unvalidated human alignment. Temporal persistence is not
a separate repair. No empirical superiority over VBench is claimed.
