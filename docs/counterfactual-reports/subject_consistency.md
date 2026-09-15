# Counterfactual Pair Accuracy: subject_consistency

- family: `temporal_relocation`
- transformation: One fixed subject-region corruption placed at the start, middle and end.
- expected relation: clean > corrupted, and the three positions should tie.
- bases: 25 (dev 5, test 20)
- derived clips: 100 (dev 20, test 80)
- levels: `corrupt_end`, `corrupt_middle`, `corrupt_start`, `clean`
- code SHA: `5c4a13091893273510794e45872eac3c59c3ad21`
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)

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

This family declares at least one group of levels that must tie, so its
CPA mixes an inequality half (the counterfactual must move the score)
with an invariance half (the declared-equal levels must tie). A pooled
CPA is dominated by whichever half is easier, so each is scored
separately, and each half carries its own cluster-bootstrap interval:
the composite's interval says nothing about either half.

| backend | split | half | pairs | CPA (dev margin) | CPA (zero margin) | 95% CI |
|---|---|---|---:|---:|---:|---|
| official | dev | sensitivity | 15 | 0.9333 | 1.0000 | [0.8000, 1.0000] |
| official | dev | invariance | 15 | 0.3333 | 0.0000 | [0.0667, 0.6667] |
| official | test | sensitivity | 60 | 0.9333 | 1.0000 | [0.8500, 1.0000] |
| official | test | invariance | 60 | 0.2500 | 0.0000 | [0.1500, 0.3500] |
| repair | dev | sensitivity | 15 | 0.8000 | 1.0000 | [0.4000, 1.0000] |
| repair | dev | invariance | 15 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| repair | test | sensitivity | 60 | 0.7833 | 1.0000 | [0.6167, 0.9167] |
| repair | test | invariance | 60 | 0.9167 | 0.0000 | [0.8167, 1.0000] |

Per-half paired difference (the same `base_id` clusters resampled once
and both backends re-scored on each resample, plan 5.4). This is the
interval that decides a half; the two marginal intervals above overlap.

| half | repair − official | paired 95% CI | bases |
|---|---:|---|---:|
| sensitivity | -0.1467 | [-0.2400, -0.0667] | 25 |
| invariance | +0.6667 | [+0.5333, +0.7733] | 25 |

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

## Sequence-level order statistics

Per-base Spearman correlation between the declared rank and the score,
and the fraction of bases whose levels come out in the declared strict
order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank
gaps more heavily; levels the family declares equal are not required
to be strictly ordered, and a base with a missing score is excluded
rather than counted as a violation.

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 5 | 0.7746 | 0.7746 | 5/5 (1.0000) |
| official | test | 20 | 0.7746 | 0.7746 | 20/20 (1.0000) |
| repair | dev | 5 | 0.7746 | 0.7746 | 5/5 (1.0000) |
| repair | test | 20 | 0.7746 | 0.7746 | 20/20 (1.0000) |

## Contract decomposition

This family declares levels that must tie, so its CPA is a mixture of
two contracts and is dominated by whichever is easier. Rank
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
| official | dev | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 5 | 0.0414 | 0.0949 |
| official | test | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 20 | 0.0455 | 0.1035 |
| repair | dev | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 5 | 0.0124 | 0.0279 |
| repair | test | rank 0: `corrupt_end`, `corrupt_middle`, `corrupt_start` | 20 | 0.0141 | 0.0319 |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value at
every level is insensitive rather than ordered: it cannot detect the
transformation at all, so no level pair can match.

| backend | split | level | n | mean | std | min | max | distinct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| official | dev | `clean` | 5 | 0.8935 | 0.0582 | 0.8285 | 0.9923 | 5 |
| official | dev | `corrupt_end` | 5 | 0.8471 | 0.0461 | 0.7888 | 0.9171 | 5 |
| official | dev | `corrupt_middle` | 5 | 0.8370 | 0.0544 | 0.7436 | 0.9030 | 5 |
| official | dev | `corrupt_start` | 5 | 0.7766 | 0.0491 | 0.6925 | 0.8300 | 5 |
| official | test | `clean` | 20 | 0.9044 | 0.0872 | 0.6957 | 0.9882 | 20 |
| official | test | `corrupt_end` | 20 | 0.8548 | 0.0840 | 0.6506 | 0.9542 | 20 |
| official | test | `corrupt_middle` | 20 | 0.8436 | 0.0841 | 0.6700 | 0.9550 | 20 |
| official | test | `corrupt_start` | 20 | 0.7731 | 0.0914 | 0.6303 | 0.9459 | 20 |
| repair | dev | `clean` | 5 | 0.9018 | 0.0556 | 0.8314 | 0.9933 | 5 |
| repair | dev | `corrupt_end` | 5 | 0.8306 | 0.0464 | 0.7616 | 0.8883 | 5 |
| repair | dev | `corrupt_middle` | 5 | 0.8246 | 0.0539 | 0.7247 | 0.8744 | 5 |
| repair | dev | `corrupt_start` | 5 | 0.8393 | 0.0443 | 0.7641 | 0.8886 | 5 |
| repair | test | `clean` | 20 | 0.9164 | 0.0734 | 0.7743 | 0.9910 | 20 |
| repair | test | `corrupt_end` | 20 | 0.8407 | 0.0761 | 0.6892 | 0.9507 | 20 |
| repair | test | `corrupt_middle` | 20 | 0.8310 | 0.0781 | 0.6854 | 0.9532 | 20 |
| repair | test | `corrupt_start` | 20 | 0.8479 | 0.0733 | 0.6882 | 0.9545 | 20 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| temporal_relocation | 0.5917 | 0.8500 | +0.2583 | [+0.0000, +0.0000] |

Paired zero-margin delta: `+0.0000` over 20 test bases; paired tie-aware delta: `+0.2583`.

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
