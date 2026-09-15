# Counterfactual Pair Accuracy: spatial_relationship

- family: `directional_flip`
- transformation: Mirror the axis named by the ordered relation, prompt unchanged.
- expected relation: original > flip.
- bases: 40 (dev 10, test 30)
- derived clips: 80 (dev 20, test 60)
- levels: `horizontal_flip`, `vertical_flip`, `original`
- code SHA: `5c4a13091893273510794e45872eac3c59c3ad21`
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)

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

This family declares at least one group of levels that must tie, so its
CPA mixes an inequality half (the counterfactual must move the score)
with an invariance half (the declared-equal levels must tie). A pooled
CPA is dominated by whichever half is easier, so each is scored
separately, and each half carries its own cluster-bootstrap interval:
the composite's interval says nothing about either half.

| backend | split | half | pairs | CPA (dev margin) | CPA (zero margin) | 95% CI |
|---|---|---|---:|---:|---:|---|
| official | dev | sensitivity | 10 | 0.4000 | 0.4000 | [0.1000, 0.7000] |
| official | test | sensitivity | 30 | 0.3667 | 0.3667 | [0.2000, 0.5333] |
| repair | dev | sensitivity | 10 | 0.2000 | 0.2000 | [0.0000, 0.5000] |
| repair | test | sensitivity | 30 | 0.0667 | 0.0667 | [0.0000, 0.1667] |

Per-half paired difference (the same `base_id` clusters resampled once
and both backends re-scored on each resample, plan 5.4). This is the
interval that decides a half; the two marginal intervals above overlap.

| half | repair − official | paired 95% CI | bases |
|---|---:|---|---:|
| sensitivity | -0.2750 | [-0.4250, -0.1500] | 40 |

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

## Sequence-level order statistics

Per-base Spearman correlation between the declared rank and the score,
and the fraction of bases whose levels come out in the declared strict
order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank
gaps more heavily; levels the family declares equal are not required
to be strictly ordered, and a base with a missing score is excluded
rather than counted as a violation.

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 10 | 0.0000 | 0.0000 | 4/10 (0.4000) |
| official | test | 30 | 0.1579 | 1.0000 | 11/30 (0.3667) |
| repair | dev | 10 | 0.3333 | 1.0000 | 2/10 (0.2000) |
| repair | test | 30 | -0.2000 | -1.0000 | 2/30 (0.0667) |

## Contract decomposition

This family declares levels that must tie, so its CPA is a mixture of
two contracts and is dominated by whichever is easier. Rank
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

Rank-gap-0 pairs are the declared-equal contract. Splitting them out
shows whether a Repair gain in the composite comes from sensitivity
(which both backends usually already have) or from the invariant half —
but an exact tie is only a contract test when the levels in the group
really are interchangeable. When they differ in more than severity (for
example a same-frame occlusion against a clip in which no frame holds
both targets), the tie rate measures the metric's saturation, not its
fidelity, and the predicate section below is the test that applies.

### Declared-equal subgroups — dispersion, not CPA

Same degeneracy as a same-rank family, applied to each declared-equal
group: the tie-margin CPA of these pairs can be pushed to 1.0 by
widening the margin, so the within-base CV is the meaningful number.

| backend | split | levels | bases | mean within-base CV | mean relative range |
|---|---|---|---:|---:|---:|

## Score sensitivity

Per-level score distribution. A metric with a single distinct value at
every level is insensitive rather than ordered: it cannot detect the
transformation at all, so no level pair can match.

| backend | split | level | n | mean | std | min | max | distinct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| official | dev | `horizontal_flip` | 7 | 0.5080 | 0.3948 | 0.0000 | 1.0000 | 7 |
| official | dev | `original` | 10 | 0.3908 | 0.4100 | 0.0000 | 1.0000 | 7 |
| official | dev | `vertical_flip` | 3 | 0.2488 | 0.2689 | 0.0126 | 0.6250 | 3 |
| official | test | `horizontal_flip` | 21 | 0.2451 | 0.3357 | 0.0000 | 1.0000 | 12 |
| official | test | `original` | 30 | 0.2836 | 0.3487 | 0.0000 | 1.0000 | 17 |
| official | test | `vertical_flip` | 9 | 0.3597 | 0.3795 | 0.0000 | 0.9860 | 9 |
| repair | dev | `horizontal_flip` | 7 | 0.0446 | 0.0866 | 0.0000 | 0.2500 | 3 |
| repair | dev | `original` | 10 | 0.1428 | 0.3129 | 0.0000 | 1.0000 | 3 |
| repair | dev | `vertical_flip` | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| repair | test | `horizontal_flip` | 21 | 0.0268 | 0.1198 | 0.0000 | 0.5625 | 2 |
| repair | test | `original` | 30 | 0.0122 | 0.0400 | 0.0000 | 0.1875 | 4 |
| repair | test | `vertical_flip` | 9 | 0.0694 | 0.1571 | 0.0000 | 0.5000 | 3 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| directional_flip | 0.3667 | 0.0667 | -0.3000 | [-0.4667, -0.1333] |

Paired zero-margin delta: `-0.3000` over 30 test bases; paired tie-aware delta: `-0.3000`.

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
