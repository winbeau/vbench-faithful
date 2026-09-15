# Counterfactual Pair Accuracy: motion_smoothness

- family: `temporal_jerk`
- transformation: Duplicate, skip and locally reverse short segments; frame count and rate fixed.
- expected relation: Monotone decreasing in perturbation severity.
- bases: 25 (dev 5, test 20)
- derived clips: 125 (dev 25, test 100)
- levels: `jerk_4_multiple`, `jerk_3_local_reverse`, `jerk_2_duplicate_skip`, `jerk_1_duplicate`, `jerk_0_original`
- code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 125 | 125 | none |
| repair | 125 | 125 | none |
| official (dev) | 25 | 25 | — |
| official (test) | 100 | 100 | — |
| repair (dev) | 25 | 25 | — |
| repair (test) | 100 | 100 | — |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 50 | 0.7600 | [0.5200, 0.9600] |
| official | test (zero-margin) | 0 | 200 | 0.8300 | [0.7700, 0.8900] |
| official | test (tie-aware) | 0 | 200 | 0.8300 | [0.7700, 0.8900] |
| repair | dev (zero-margin) | 0 | 50 | 0.7000 | [0.5200, 0.8600] |
| repair | test (zero-margin) | 0 | 200 | 0.7250 | [0.6500, 0.7900] |
| repair | test (tie-aware) | 0 | 200 | 0.7250 | [0.6500, 0.7900] |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `jerk_0_original` | 25 | 0.9565 | 0.0321 | 0.8475 | 0.9851 | 25 |
| official | `jerk_1_duplicate` | 25 | 0.9559 | 0.0324 | 0.8460 | 0.9856 | 25 |
| official | `jerk_2_duplicate_skip` | 25 | 0.9545 | 0.0338 | 0.8446 | 0.9866 | 25 |
| official | `jerk_3_local_reverse` | 25 | 0.9482 | 0.0383 | 0.8418 | 0.9845 | 25 |
| official | `jerk_4_multiple` | 25 | 0.9378 | 0.0474 | 0.8035 | 0.9835 | 25 |
| repair | `jerk_0_original` | 25 | 0.7562 | 0.0481 | 0.6833 | 0.9241 | 25 |
| repair | `jerk_1_duplicate` | 25 | 0.7292 | 0.0310 | 0.6531 | 0.7827 | 25 |
| repair | `jerk_2_duplicate_skip` | 25 | 0.6586 | 0.0378 | 0.6071 | 0.7292 | 25 |
| repair | `jerk_3_local_reverse` | 25 | 0.7094 | 0.0458 | 0.6101 | 0.7804 | 25 |
| repair | `jerk_4_multiple` | 25 | 0.6732 | 0.0653 | 0.5031 | 0.7655 | 25 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| temporal_jerk | 0.8300 | 0.7250 | -0.1050 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
