# Counterfactual Pair Accuracy: human_action

- family: `filename_invariance`
- transformation: Byte-identical copies named with a correct, wrong and neutral action.
- expected relation: Invariance: only the filename changes.
- bases: 25 (dev 5, test 20)
- derived clips: 75 (dev 15, test 60)
- levels: `filename_neutral`, `filename_correct`, `filename_wrong`
- code SHA: `77ca915ac5e41f7d6d1a122181c0051791404748`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 60 | 75 | none |
| repair | 60 | 75 | none |
| official (dev) | 12 | 15 | — |
| official (test) | 48 | 60 | — |
| repair (dev) | 12 | 15 | — |
| repair (test) | 48 | 60 | — |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 12 | 0.5000 | [0.3333, 0.8333] |
| official | test (zero-margin) | 0 | 48 | 0.4167 | [0.3333, 0.5385] |
| official | test (tie-aware) | 1 | 48 | 1.0000 | [1.0000, 1.0000] |
| repair | dev (zero-margin) | 0 | 12 | 1.0000 | [1.0000, 1.0000] |
| repair | test (zero-margin) | 0 | 48 | 1.0000 | [1.0000, 1.0000] |
| repair | test (tie-aware) | 0 | 48 | 1.0000 | [1.0000, 1.0000] |

## Invariance statistics

Every level of this family carries the same expected rank, so the
tie-aware CPA above is degenerate — widening the dev margin until all
differences count as ties yields CPA 1.0 regardless of how unstable the
metric is. The dispersion below is the meaningful invariance measure
(plan sections 6.4 and 8.3).

| backend | split | bases | mean within-base CV | mean relative range |
|---|---|---:|---:|---:|
| official | dev | 3 | 1.4142 | 3.0000 |
| official | test | 14 | 1.4142 | 3.0000 |
| repair | dev | 4 | 0.0000 | 0.0000 |
| repair | test | 16 | 0.0000 | 0.0000 |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `filename_correct` | 20 | 0.8500 | 0.3571 | 0.0000 | 1.0000 | 2 |
| official | `filename_neutral` | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | `filename_wrong` | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| repair | `filename_correct` | 20 | 0.9738 | 0.0567 | 0.7583 | 0.9999 | 20 |
| repair | `filename_neutral` | 20 | 0.9738 | 0.0567 | 0.7583 | 0.9999 | 20 |
| repair | `filename_wrong` | 20 | 0.9738 | 0.0567 | 0.7583 | 0.9999 | 20 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| filename_invariance | 1.0000 | 1.0000 | +0.0000 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
