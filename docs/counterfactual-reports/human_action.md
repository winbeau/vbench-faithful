# Counterfactual Pair Accuracy: human_action

- family: `filename_invariance`
- transformation: Byte-identical copies named with a correct, wrong and neutral action.
- expected relation: Invariance: only the filename changes.
- bases: 25 (dev 5, test 20)
- derived clips: 75 (dev 15, test 60)
- levels: `filename_neutral`, `filename_correct`, `filename_wrong`
- code SHA: `f54fe67c53b44b39d5874df28632317b69477b52`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 75 | 75 | none |
| repair | 75 | 75 | none |
| official (dev) | 15 | 15 | — |
| official (test) | 60 | 60 | — |
| repair (dev) | 15 | 15 | — |
| repair (test) | 60 | 60 | — |

## CPA by contract half

This family mixes an inequality half (the counterfactual must move the
score) with an invariance half (relocated variants must tie). A pooled CPA
is dominated by whichever half is easier, so each is scored separately.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) |
|---|---|---:|---:|---:|
| official | invariance | 75 | 1.0000 | 0.4933 |
| repair | invariance | 75 | 1.0000 | 1.0000 |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 15 | 0.4667 | [0.3333, 0.7333] |
| official | test (zero-margin) | 0 | 60 | 0.5000 | [0.4000, 0.6333] |
| official | test (tie-aware) | 1 | 60 | 1.0000 | [1.0000, 1.0000] |
| repair | dev (zero-margin) | 0 | 15 | 1.0000 | [1.0000, 1.0000] |
| repair | test (zero-margin) | 0 | 60 | 1.0000 | [1.0000, 1.0000] |
| repair | test (tie-aware) | 0 | 60 | 1.0000 | [1.0000, 1.0000] |

## Invariance statistics

Every level of this family carries the same expected rank, so the
tie-aware CPA above is degenerate — widening the dev margin until all
differences count as ties yields CPA 1.0 regardless of how unstable the
metric is. The dispersion below is the meaningful invariance measure
(plan sections 6.4 and 8.3).

| backend | split | bases | mean within-base CV | mean relative range |
|---|---|---:|---:|---:|

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `filename_correct` | 25 | 0.7600 | 0.4271 | 0.0000 | 1.0000 | 2 |
| official | `filename_neutral` | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | `filename_wrong` | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| repair | `filename_correct` | 25 | 0.9291 | 0.1493 | 0.4326 | 0.9999 | 25 |
| repair | `filename_neutral` | 25 | 0.9291 | 0.1493 | 0.4326 | 0.9999 | 25 |
| repair | `filename_wrong` | 25 | 0.9291 | 0.1493 | 0.4326 | 0.9999 | 25 |

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
