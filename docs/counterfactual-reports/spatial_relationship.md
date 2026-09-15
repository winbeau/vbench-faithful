# Counterfactual Pair Accuracy: spatial_relationship

- family: `directional_flip`
- transformation: Mirror the axis named by the ordered relation, prompt unchanged.
- expected relation: original > flip.
- bases: 40 (dev 10, test 30)
- derived clips: 80 (dev 20, test 60)
- levels: `vertical_flip`, `horizontal_flip`, `original`
- code SHA: `bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 80 | 80 | none |
| repair | 80 | 80 | none |
| official (dev) | 20 | 20 | — |
| official (test) | 60 | 60 | — |
| repair (dev) | 20 | 20 | — |
| repair (test) | 60 | 60 | — |

## CPA by contract half

This family mixes an inequality half (the counterfactual must move the
score) with an invariance half (relocated variants must tie). A pooled CPA
is dominated by whichever half is easier, so each is scored separately.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) |
|---|---|---:|---:|---:|
| official | sensitivity | 40 | 0.3750 | 0.3750 |
| repair | sensitivity | 40 | 0.1000 | 0.1000 |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 10 | 0.4000 | [0.1000, 0.7000] |
| official | test (zero-margin) | 0 | 30 | 0.3667 | [0.2000, 0.5333] |
| official | test (tie-aware) | 0 | 30 | 0.3667 | [0.2000, 0.5333] |
| repair | dev (zero-margin) | 0 | 10 | 0.2000 | [0.0000, 0.5000] |
| repair | test (zero-margin) | 0 | 30 | 0.0667 | [0.0000, 0.1667] |
| repair | test (tie-aware) | 0 | 30 | 0.0667 | [0.0000, 0.1667] |

## Contract decomposition

This family declares more than one expected rank, so its CPA is a
mixture of two contracts and is dominated by whichever is easier. Rank
gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of
the levels declared equal, and there the only correct prediction is a
tie, so a widening dev margin raises this half without measuring
anything. Read the two halves separately, never the composite alone.

| backend | split | rank gap | pairs | match rate | tie rate |
|---|---|---:|---:|---:|---:|
| official | test (zero-margin) | 1 | 30 | 0.3667 | 0.3667 |
| official | test (tie-aware) | 1 | 30 | 0.3667 | 0.3667 |
| repair | test (zero-margin) | 1 | 30 | 0.0667 | 0.8333 |
| repair | test (tie-aware) | 1 | 30 | 0.0667 | 0.8333 |

Rank-gap-0 pairs are the family's actual target. Splitting them out
shows whether a Repair gain in the composite comes from sensitivity
(which both backends usually already have) or from the invariant half.

### Declared-equal subgroups — dispersion, not CPA

Same degeneracy as a same-rank family, applied to each declared-equal
group: the tie-margin CPA of these pairs can be pushed to 1.0 by
widening the margin, so the within-base CV is the meaningful number.

| backend | split | levels | bases | mean within-base CV | mean relative range |
|---|---|---|---:|---:|---:|

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `horizontal_flip` | 28 | 0.3108 | 0.3694 | 0.0000 | 1.0000 | 17 |
| official | `original` | 40 | 0.3104 | 0.3679 | 0.0000 | 1.0000 | 22 |
| official | `vertical_flip` | 12 | 0.3319 | 0.3583 | 0.0000 | 0.9860 | 12 |
| repair | `horizontal_flip` | 28 | 0.0312 | 0.1127 | 0.0000 | 0.5625 | 4 |
| repair | `original` | 40 | 0.0448 | 0.1700 | 0.0000 | 1.0000 | 6 |
| repair | `vertical_flip` | 12 | 0.0521 | 0.1394 | 0.0000 | 0.5000 | 3 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| directional_flip | 0.3667 | 0.0667 | -0.3000 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
