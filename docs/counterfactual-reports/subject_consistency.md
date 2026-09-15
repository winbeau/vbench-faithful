# Counterfactual Pair Accuracy: subject_consistency

- family: `temporal_relocation`
- transformation: One fixed subject-region corruption placed at the start, middle and end.
- expected relation: clean > corrupted, and the three positions should tie.
- bases: 25 (dev 5, test 20)
- derived clips: 100 (dev 20, test 80)
- levels: `corrupt_start`, `corrupt_end`, `corrupt_middle`, `clean`
- code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 100 | 100 | none |
| repair | 100 | 100 | none |
| official (dev) | 20 | 20 | — |
| official (test) | 80 | 80 | — |
| repair (dev) | 20 | 20 | — |
| repair (test) | 80 | 80 | — |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 30 | 0.5000 | [0.5000, 0.5000] |
| official | test (zero-margin) | 0 | 120 | 0.5000 | [0.5000, 0.5000] |
| official | test (tie-aware) | 0.01389 | 120 | 0.5917 | [0.5583, 0.6250] |
| repair | dev (zero-margin) | 0 | 30 | 0.5000 | [0.5000, 0.5000] |
| repair | test (zero-margin) | 0 | 120 | 0.5000 | [0.5000, 0.5000] |
| repair | test (tie-aware) | 0.04457 | 120 | 0.8500 | [0.7583, 0.9333] |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `clean` | 25 | 0.9022 | 0.0824 | 0.6957 | 0.9923 | 25 |
| official | `corrupt_end` | 25 | 0.8533 | 0.0780 | 0.6506 | 0.9542 | 25 |
| official | `corrupt_middle` | 25 | 0.8423 | 0.0791 | 0.6700 | 0.9550 | 25 |
| official | `corrupt_start` | 25 | 0.7738 | 0.0847 | 0.6303 | 0.9459 | 25 |
| repair | `clean` | 25 | 0.9135 | 0.0704 | 0.7743 | 0.9933 | 25 |
| repair | `corrupt_end` | 25 | 0.8387 | 0.0713 | 0.6892 | 0.9507 | 25 |
| repair | `corrupt_middle` | 25 | 0.8297 | 0.0739 | 0.6854 | 0.9532 | 25 |
| repair | `corrupt_start` | 25 | 0.8461 | 0.0686 | 0.6882 | 0.9545 | 25 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| temporal_relocation | 0.5917 | 0.8500 | +0.2583 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
