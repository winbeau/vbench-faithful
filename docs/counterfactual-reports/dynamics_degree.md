# Counterfactual Pair Accuracy: dynamics_degree

- family: `fps_resampling`
- transformation: Resample one trajectory to 8/6/4/2 fps with the duration held fixed.
- expected relation: Invariance: every rung should score the same.
- bases: 40 (dev 10, test 30)
- derived clips: 160 (dev 40, test 120)
- levels: `fps2`, `fps8`, `fps6`, `fps4`
- code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 160 | 160 | none |
| repair | 160 | 160 | none |
| official (dev) | 40 | 40 | — |
| official (test) | 120 | 120 | — |
| repair (dev) | 40 | 40 | — |
| repair (test) | 120 | 120 | — |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 60 | 0.0000 | [0.0000, 0.0000] |
| official | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |
| official | test (tie-aware) | 20.14 | 180 | 0.8333 | [0.7444, 0.9111] |
| repair | dev (zero-margin) | 0 | 60 | 0.0000 | [0.0000, 0.0000] |
| repair | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |
| repair | test (tie-aware) | 0.1466 | 180 | 0.8444 | [0.7388, 0.9333] |

## Invariance statistics

Every level of this family carries the same expected rank, so the
tie-aware CPA above is degenerate — widening the dev margin until all
differences count as ties yields CPA 1.0 regardless of how unstable the
metric is. The dispersion below is the meaningful invariance measure
(plan sections 6.4 and 8.3).

| backend | split | bases | mean within-base CV | mean relative range |
|---|---|---:|---:|---:|
| official | dev | 10 | 0.3838 | 0.9862 |
| official | test | 30 | 0.3177 | 0.8196 |
| repair | dev | 10 | 0.1800 | 0.4662 |
| repair | test | 30 | 0.2360 | 0.6107 |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `fps2` | 40 | 32.8744 | 30.2231 | 0.5438 | 153.1937 | 40 |
| official | `fps4` | 40 | 24.2540 | 26.6125 | 0.3743 | 153.0603 | 40 |
| official | `fps6` | 40 | 19.7201 | 23.8540 | 0.3515 | 141.0087 | 40 |
| official | `fps8` | 40 | 16.5108 | 19.8936 | 0.3038 | 105.4677 | 40 |
| repair | `fps2` | 40 | 0.1282 | 0.1141 | 0.0014 | 0.4035 | 40 |
| repair | `fps4` | 40 | 0.1864 | 0.1737 | 0.0020 | 0.7579 | 40 |
| repair | `fps6` | 40 | 0.2230 | 0.2131 | 0.0028 | 1.0322 | 40 |
| repair | `fps8` | 40 | 0.2484 | 0.2534 | 0.0032 | 1.1838 | 40 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| fps_resampling | 0.8333 | 0.8444 | +0.0111 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
