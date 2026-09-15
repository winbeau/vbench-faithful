# Counterfactual Pair Accuracy: dynamics_degree

- family: `fps_resampling`
- transformation: resample one trajectory to 8/6/4/2 fps with the duration held fixed.
- expected relation: invariance — every rung should score the same.
- bases: 40 (dev 10, test 30)
- derived clips: 160 (dev 40, test 120)
- levels: `fps8`, `fps6`, `fps4`, `fps2`
- code SHA: `a044ac9 + lagcal patch (uncommitted)`
- backends: `official` means raw top-5% flow in pixels (the Official intermediate quantity, exponent 0); `repair_v1` divides by dt (exponent 1, the archived repair); `repair_v2` divides by dt**0.5, the shipped default (exponent 0.5)

This family has a single expected rank, so **every** level pair expects a tie and the
composite CPA is degenerate (see the appendix). The primary evidence is the
sampling-interval response below.

## Score coverage

| backend | scored clips | expected |
|---|---:|---:|
| official | 160 | 160 |
| repair_v1 | 160 | 160 |
| repair_v2 | 160 | 160 |

## Sampling-interval response (primary)

Per-level mean score, the ratio profile relative to the 8 fps rung, and the fitted
exponent `p` in `score ~ dt**p`. Target `p = 0`. Dev and test are reported separately.

### dev

| backend | 8 fps | 6 fps | 4 fps | 2 fps | ratio profile | fitted p | fps2/fps8 |
|---|---:|---:|---:|---:|---|---:|---:|
| official | 10.7698 | 12.7531 | 17.6140 | 25.8855 | 1.000 / 1.184 / 1.635 / 2.404 | +0.641 | 2.404 |
| repair_v1 | 0.2039 | 0.1796 | 0.1648 | 0.1195 | 1.000 / 0.881 / 0.808 / 0.586 | -0.376 | 0.586 |
| repair_v2 | 0.0721 | 0.0733 | 0.0824 | 0.0845 | 1.000 / 1.017 / 1.143 / 1.172 | +0.124 | 1.172 |

### test

| backend | 8 fps | 6 fps | 4 fps | 2 fps | ratio profile | fitted p | fps2/fps8 |
|---|---:|---:|---:|---:|---|---:|---:|
| official | 18.4244 | 22.0424 | 26.4673 | 35.2040 | 1.000 / 1.196 / 1.437 / 1.911 | +0.458 | 1.911 |
| repair_v1 | 0.2632 | 0.2375 | 0.1936 | 0.1311 | 1.000 / 0.902 / 0.736 / 0.498 | -0.511 | 0.498 |
| repair_v2 | 0.0930 | 0.0970 | 0.0968 | 0.0927 | 1.000 / 1.042 / 1.040 / 0.996 | -0.011 | 0.996 |

### all

| backend | 8 fps | 6 fps | 4 fps | 2 fps | ratio profile | fitted p | fps2/fps8 |
|---|---:|---:|---:|---:|---|---:|---:|
| official | 16.5108 | 19.7201 | 24.2540 | 32.8744 | 1.000 / 1.194 / 1.469 / 1.991 | +0.491 | 1.991 |
| repair_v1 | 0.2484 | 0.2230 | 0.1864 | 0.1282 | 1.000 / 0.898 / 0.750 / 0.516 | -0.481 | 0.516 |
| repair_v2 | 0.0878 | 0.0910 | 0.0932 | 0.0906 | 1.000 / 1.037 / 1.061 / 1.032 | +0.019 | 1.032 |

### Calibrated exponent

- exponent that flattens the ladder, all 40 bases: **+0.518**, base-cluster bootstrap 95% CI [+0.417, +0.651]
- calibrated on the 10 dev bases only: +0.624; applied to test it leaves a residual slope of -0.135 (the dev split is too small to calibrate this constant)
- both `0` (the Official intermediate quantity) and `1` (the archived repair) fall outside the CI

## Per-clip dispersion and measured exponent

A flat level *mean* does not imply per-clip invariance. `ratio` is the per-base
fps2/fps8 score ratio; `CV` and relative range are within-base dispersion.

| backend | split | bases | mean CV | mean relative range | median fps2/fps8 | IQR | within ±20% |
|---|---|---:|---:|---:|---:|---|---:|
| official | dev | 10 | 0.3838 | 0.9862 | 2.748 | [2.264, 3.309] | 0% |
| official | test | 30 | 0.3177 | 0.8196 | 2.189 | [1.487, 3.043] | 7% |
| official | all | 40 | 0.3342 | 0.8612 | 2.548 | [1.604, 3.055] | 5% |
| repair_v1 | dev | 10 | 0.1800 | 0.4662 | 0.669 | [0.585, 0.779] | 10% |
| repair_v1 | test | 30 | 0.2360 | 0.6107 | 0.582 | [0.365, 0.760] | 20% |
| repair_v1 | all | 40 | 0.2220 | 0.5746 | 0.642 | [0.404, 0.769] | 18% |
| repair_v2 | dev | 10 | 0.1749 | 0.4375 | 1.338 | [1.169, 1.558] | 20% |
| repair_v2 | test | 30 | 0.1822 | 0.4760 | 1.163 | [0.731, 1.520] | 20% |
| repair_v2 | all | 40 | 0.1804 | 0.4664 | 1.284 | [0.807, 1.537] | 20% |

Within-clip exponent fitted from each clip's own multi-lag displacements (lag 1/2/4
sampled frames). It is a diagnostic: the fitted value is itself lag-window dependent,
which is why applying it per clip under-corrects relative to the fixed benchmark exponent.

| rung | clips | frames | mean measured exponent | median | sd |
|---|---:|---|---:|---:|---:|
| `fps8` | 40 | [16, 27] | +0.618 | +0.656 | 0.298 |
| `fps6` | 40 | [12, 20] | +0.557 | +0.593 | 0.310 |
| `fps4` | 40 | [8, 14] | +0.512 | +0.540 | 0.312 |
| `fps2` | 40 | [4, 7] | +0.480 | +0.521 | 0.496 |

## Threshold and coverage domain

The static/moving decision uses the same normalisation as the intensity: the Official
pixel threshold is re-expressed as `official_px / diagonal / reference_lag**alpha` and
compared against `d / dt**alpha`, which makes the fraction frame-rate invariant.

| backend (exponent) | split | mean coverage 8 fps | 6 fps | 4 fps | 2 fps |
|---|---|---:|---:|---:|---:|
| repair_v1 (1.0, ballistic) | dev | 0.3867 | 0.3727 | 0.3429 | 0.3000 |
| repair_v1 (1.0, ballistic) | test | 0.5010 | 0.4944 | 0.4410 | 0.3333 |
| repair_v2 (0.5, default) | dev | 0.3867 | 0.4091 | 0.4714 | 0.5667 |
| repair_v2 (0.5, default) | test | 0.5010 | 0.5219 | 0.5198 | 0.5333 |

## Composite CPA (non-diagnostic for this family)

Every pair in a same-rank family expects a tie, so `zero-margin` requires exact
floating-point equality (it is `0.0000` for any continuous score, including a perfectly
invariant one), and the dev-calibrated tie margin is the 0.9 quantile of dev deltas by
construction, which makes the dev tie-aware CPA `0.90` regardless of the score. The
statistic is also blind to the sign of the level effect, so the mirrored exponents of
`official` (+0.491) and `repair_v1` (-0.481) score the same. The archived values are:

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |
| official | test (tie-aware) | 20.14 | 180 | 0.8333 | [0.7444, 0.9111] |
| repair_v1 | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |
| repair_v1 | test (tie-aware) | 0.1466 | 180 | 0.8444 | [0.7388, 0.9333] |

The `+0.0111` difference is ~1/5 of the sampling standard deviation of that statistic on
this base budget and is expected to be ~0 under the null; it must not be quoted as a win.

## Status and limitations

- The clips are re-encoded per rung; on the identical 0.5 s trajectory span the 2 fps
  rung's lag-1 displacement and the 8 fps rung's lag-4 displacement agree to within 3.5%,
  so the level effect is a sampling-interval effect, not a codec effect.
- Fixing the level mean does not fix individual clips: the per-clip fps2/fps8 ratio stays
  widely dispersed, so per-video comparisons across frame rates remain unreliable.
- Two independent estimators are both sublinear (RAFT and Farneback), so the effect is not
  specific to RAFT; the chord/path ratio (`straightness`, emitted per clip with the
  per-lag chord and path displacements) is what distinguishes trajectory curvature from
  estimator saturation and is recorded for every clip.
- The coarse rungs sample a shorter span of the source trajectory (fps2 covers 80% of a
  16-frame 8 fps source and 93.8% of a 33-frame 10 fps source), so the rungs do not
  average exactly the same content.
- Model/CUDA parity against the frozen E0 numbers has not been verified.

