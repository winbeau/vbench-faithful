# Counterfactual Pair Accuracy: scene

- family: `environment_coverage`
- transformation: 2x2 grid mixing a wrong-scene donor with target-scene quadrants at 0-100%.
- expected relation: Monotone increasing in target coverage.
- bases: 25 (dev 5, test 20)
- derived clips: 125 (dev 25, test 100)
- levels: `coverage_000`, `coverage_025`, `coverage_050`, `coverage_075`, `coverage_100`
- code SHA: `bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 125 | 125 | none |
| repair | 125 | 125 | none |
| official (dev) | 25 | 25 | — |
| official (test) | 100 | 100 | — |
| repair (dev) | 25 | 25 | — |
| repair (test) | 100 | 100 | — |

## CPA by contract half

This family mixes an inequality half (the counterfactual must move the
score) with an invariance half (relocated variants must tie). A pooled CPA
is dominated by whichever half is easier, so each is scored separately.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) |
|---|---|---:|---:|---:|
| official | sensitivity | 250 | 0.3760 | 0.3760 |
| repair | sensitivity | 250 | 0.9240 | 0.9240 |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 50 | 0.3400 | [0.0600, 0.6200] |
| official | test (zero-margin) | 0 | 200 | 0.3850 | [0.2550, 0.5050] |
| official | test (tie-aware) | 0 | 200 | 0.3850 | [0.2550, 0.5050] |
| repair | dev (zero-margin) | 0 | 50 | 0.9000 | [0.7800, 1.0000] |
| repair | test (zero-margin) | 0 | 200 | 0.9300 | [0.8400, 0.9850] |
| repair | test (tie-aware) | 0 | 200 | 0.9300 | [0.8400, 0.9850] |

## Contract decomposition

This family declares more than one expected rank, so its CPA is a
mixture of two contracts and is dominated by whichever is easier. Rank
gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of
the levels declared equal, and there the only correct prediction is a
tie, so a widening dev margin raises this half without measuring
anything. Read the two halves separately, never the composite alone.

| backend | split | rank gap | pairs | match rate | tie rate |
|---|---|---:|---:|---:|---:|
| official | test (zero-margin) | 1 | 80 | 0.2750 | 0.6875 |
| official | test (zero-margin) | 2 | 60 | 0.3500 | 0.6167 |
| official | test (zero-margin) | 3 | 40 | 0.5000 | 0.5000 |
| official | test (zero-margin) | 4 | 20 | 0.7000 | 0.3000 |
| official | test (tie-aware) | 1 | 80 | 0.2750 | 0.6875 |
| official | test (tie-aware) | 2 | 60 | 0.3500 | 0.6167 |
| official | test (tie-aware) | 3 | 40 | 0.5000 | 0.5000 |
| official | test (tie-aware) | 4 | 20 | 0.7000 | 0.3000 |
| repair | test (zero-margin) | 1 | 80 | 0.9000 | 0.0000 |
| repair | test (zero-margin) | 2 | 60 | 0.9500 | 0.0000 |
| repair | test (zero-margin) | 3 | 40 | 0.9500 | 0.0000 |
| repair | test (zero-margin) | 4 | 20 | 0.9500 | 0.0000 |
| repair | test (tie-aware) | 1 | 80 | 0.9000 | 0.0000 |
| repair | test (tie-aware) | 2 | 60 | 0.9500 | 0.0000 |
| repair | test (tie-aware) | 3 | 40 | 0.9500 | 0.0000 |
| repair | test (tie-aware) | 4 | 20 | 0.9500 | 0.0000 |

Rank-gap-0 pairs are the family's actual target. Splitting them out
shows whether a Repair gain in the composite comes from sensitivity
(which both backends usually already have) or from the invariant half.

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `coverage_000` | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | `coverage_025` | 25 | 0.1150 | 0.3116 | 0.0000 | 1.0000 | 3 |
| official | `coverage_050` | 25 | 0.1200 | 0.2663 | 0.0000 | 1.0000 | 6 |
| official | `coverage_075` | 25 | 0.2500 | 0.3584 | 0.0000 | 1.0000 | 8 |
| official | `coverage_100` | 25 | 0.3950 | 0.4049 | 0.0000 | 1.0000 | 6 |
| repair | `coverage_000` | 25 | 0.3592 | 0.0114 | 0.3363 | 0.3805 | 25 |
| repair | `coverage_025` | 25 | 0.3709 | 0.0105 | 0.3540 | 0.3952 | 25 |
| repair | `coverage_050` | 25 | 0.3825 | 0.0131 | 0.3610 | 0.4158 | 25 |
| repair | `coverage_075` | 25 | 0.3921 | 0.0132 | 0.3636 | 0.4228 | 25 |
| repair | `coverage_100` | 25 | 0.4021 | 0.0154 | 0.3661 | 0.4317 | 25 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| environment_coverage | 0.3850 | 0.9300 | +0.5450 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
