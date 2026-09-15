# Counterfactual Pair Accuracy: subject_consistency

- family: `temporal_relocation`
- transformation: One fixed subject-region corruption placed at the start, middle and end.
- expected relation: clean > corrupted, and the three positions should tie.
- bases: 25 (dev 5, test 20)
- derived clips: 100 (dev 20, test 80)
- levels: `corrupt_start`, `corrupt_middle`, `corrupt_end`, `clean`
- code SHA: `bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 100 | 100 | none |
| repair | 100 | 100 | none |
| official (dev) | 20 | 20 | — |
| official (test) | 80 | 80 | — |
| repair (dev) | 20 | 20 | — |
| repair (test) | 80 | 80 | — |

## CPA by contract half

This family mixes an inequality half (the counterfactual must move the
score) with an invariance half (relocated variants must tie). A pooled CPA
is dominated by whichever half is easier, so each is scored separately.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) |
|---|---|---:|---:|---:|
| official | sensitivity | 75 | 0.9333 | 1.0000 |
| official | invariance | 75 | 0.2667 | 0.0000 |
| repair | sensitivity | 75 | 0.7867 | 1.0000 |
| repair | invariance | 75 | 0.9333 | 0.0000 |

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

## Contract decomposition

This family declares more than one expected rank, so its CPA is a
mixture of two contracts and is dominated by whichever is easier. Rank
gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of
the levels declared equal, and there the only correct prediction is a
tie, so a widening dev margin raises this half without measuring
anything. Read the two halves separately, never the composite alone.

| backend | split | rank gap | pairs | match rate | tie rate |
|---|---|---:|---:|---:|---:|
| official | test (zero-margin) | 0 | 60 | 0.0000 | 0.0000 |
| official | test (zero-margin) | 1 | 60 | 1.0000 | 0.0000 |
| official | test (tie-aware) | 0 | 60 | 0.2500 | 0.2500 |
| official | test (tie-aware) | 1 | 60 | 0.9333 | 0.0667 |
| repair | test (zero-margin) | 0 | 60 | 0.0000 | 0.0000 |
| repair | test (zero-margin) | 1 | 60 | 1.0000 | 0.0000 |
| repair | test (tie-aware) | 0 | 60 | 0.9167 | 0.9167 |
| repair | test (tie-aware) | 1 | 60 | 0.7833 | 0.2167 |

Rank-gap-0 pairs are the family's actual target. Splitting them out
shows whether a Repair gain in the composite comes from sensitivity
(which both backends usually already have) or from the invariant half.

### Declared-equal subgroups — dispersion, not CPA

Same degeneracy as a same-rank family, applied to each declared-equal
group: the tie-margin CPA of these pairs can be pushed to 1.0 by
widening the margin, so the within-base CV is the meaningful number.

| backend | split | levels | bases | mean within-base CV | mean relative range |
|---|---|---|---:|---:|---:|
| official | dev | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 5 | 0.0414 | 0.0949 |
| official | test | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 20 | 0.0455 | 0.1035 |
| repair | dev | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 5 | 0.0124 | 0.0279 |
| repair | test | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 20 | 0.0141 | 0.0319 |

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
