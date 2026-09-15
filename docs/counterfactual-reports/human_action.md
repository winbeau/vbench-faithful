# Counterfactual Pair Accuracy: human_action

- family: `filename_invariance`
- transformation: Byte-identical copies named with a correct, wrong and neutral action.
- expected relation: Invariance: only the filename changes.
- bases: 25 (dev 5, test 20)
- derived clips: 75 (dev 15, test 60)
- levels: `filename_wrong`, `filename_correct`, `filename_neutral`
- code SHA: `5c4a13091893273510794e45872eac3c59c3ad21`
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 75 | 75 | none |
| repair | 75 | 75 | none |
| official (dev) | 15 | 15 | — |
| official (test) | 60 | 60 | — |
| repair (dev) | 15 | 15 | — |
| repair (test) | 60 | 60 | — |

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
| official | dev | 4 | 1.4142 | 3.0000 |
| official | test | 15 | 1.4142 | 3.0000 |
| repair | dev | 5 | 0.0000 | 0.0000 |
| repair | test | 20 | 0.0000 | 0.0000 |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | split | level | n | mean | std | min | max | distinct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| official | dev | `filename_correct` | 5 | 0.8000 | 0.4000 | 0.0000 | 1.0000 | 2 |
| official | dev | `filename_neutral` | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | dev | `filename_wrong` | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | test | `filename_correct` | 20 | 0.7500 | 0.4330 | 0.0000 | 1.0000 | 2 |
| official | test | `filename_neutral` | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | test | `filename_wrong` | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| repair | dev | `filename_correct` | 5 | 0.9629 | 0.0440 | 0.8821 | 0.9997 | 5 |
| repair | dev | `filename_neutral` | 5 | 0.9629 | 0.0440 | 0.8821 | 0.9997 | 5 |
| repair | dev | `filename_wrong` | 5 | 0.9629 | 0.0440 | 0.8821 | 0.9997 | 5 |
| repair | test | `filename_correct` | 20 | 0.9206 | 0.1644 | 0.4326 | 0.9999 | 20 |
| repair | test | `filename_neutral` | 20 | 0.9206 | 0.1644 | 0.4326 | 0.9999 | 20 |
| repair | test | `filename_wrong` | 20 | 0.9206 | 0.1644 | 0.4326 | 0.9999 | 20 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| filename_invariance | 1.0000 | 1.0000 | +0.0000 | [+0.3667, +0.6000] |

Paired zero-margin delta: `+0.5000` over 20 test bases; paired tie-aware delta: `+0.0000`.

## Frame evidence

No per-clip evidence was recorded for this run, so a low repair score cannot be
attributed to detector drop-outs rather than to a wrong direction. Re-score with the
current `score.py` to populate it.

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
