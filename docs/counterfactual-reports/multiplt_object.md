# Counterfactual Pair Accuracy: multiplt_object

- family: `weakest_object_visibility`
- transformation: Alpha-blend the tracked weaker target towards its local mean at 0-100%.
- expected relation: Monotone decreasing as the weak target disappears.
- bases: 25 (dev 5, test 20)
- derived clips: 150 (dev 30, test 120)
- levels: `conjunction_control`, `occlusion_100`, `occlusion_075`, `occlusion_050`, `occlusion_025`, `occlusion_000`
- code SHA: `a660756` (the tree it names is content-identical to the scoring tree)
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 150 | 150 | none |
| repair | 150 | 150 | none |
| official (dev) | 30 | 30 | — |
| official (test) | 120 | 120 | — |
| repair (dev) | 30 | 30 | — |
| repair (test) | 120 | 120 | — |

## CPA by contract half

This family declares at least one group of levels that must tie, so its
CPA mixes an inequality half (the counterfactual must move the score)
with an invariance half (the declared-equal levels must tie). A pooled
CPA is dominated by whichever half is easier, so each is scored
separately, and each half carries its own cluster-bootstrap interval:
the composite's interval says nothing about either half.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) | 95% CI |
|---|---|---:|---:|---:|---|
| official | sensitivity | 350 | 0.5371 | 0.5371 | [0.4171, 0.6457] |
| official | invariance | 25 | 0.6000 | 0.6000 | [0.4000, 0.8000] |
| repair | sensitivity | 350 | 0.8171 | 0.8171 | [0.7343, 0.8829] |
| repair | invariance | 25 | 0.0000 | 0.0000 | [0.0000, 0.0000] |

Per-half paired difference (the same `base_id` clusters resampled once
and both backends re-scored on each resample, plan 5.4). This is the
interval that decides a half; the two marginal intervals above overlap.

| half | repair − official | paired 95% CI | bases |
|---|---:|---|---:|
| sensitivity | +0.2800 | [+0.1886, +0.3771] | 25 |
| invariance | -0.6000 | [-0.8000, -0.4000] | 25 |

### Conjunction control — the plan's actual predicate

Plan section 11.4 does not ask the temporal-conjunction control to
*tie* the fully occluded clip. It asks that a clip in which no frame
holds both targets is scored as incomplete, because that is what
catches a repair which silently converts same-frame conjunction into
temporal union. Equality between two different corruption geometries
on a continuous score is not achievable by any continuous estimator —
and a hard 0/1 metric reaches it only by saturating, which is what the
Official column below does. The predicate is therefore reported
directly, next to the tie-based CPA that misrepresents it.

| backend | bases | control mean | `occlusion_100` mean | control ≤ `occlusion_100` |
|---|---:|---:|---:|---:|
| official | 25 | 0.1275 | 0.1825 | 0.8800 |
| repair | 25 | 0.2009 | 0.2436 | 0.7200 |

Read the tie-based invariance CPA as *not applicable* whenever the two
levels in the declared-equal group are different corruption geometries:
an exact tie requires the metric to be blind to the difference between
them, which is a property of saturation rather than of contract
fidelity.

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 75 | 0.3200 | [0.0800, 0.5600] |
| official | test (zero-margin) | 0 | 300 | 0.5967 | [0.4833, 0.6967] |
| official | test (tie-aware) | 0 | 300 | 0.5967 | [0.4833, 0.6967] |
| repair | dev (zero-margin) | 0 | 75 | 0.7600 | [0.5733, 0.9067] |
| repair | test (zero-margin) | 0 | 300 | 0.7633 | [0.6767, 0.8300] |
| repair | test (tie-aware) | 0 | 300 | 0.7633 | [0.6767, 0.8300] |

## Sequence-level order statistics

Per-base Spearman correlation between the declared rank and the score,
and the fraction of bases whose levels come out in the declared strict
order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank
gaps more heavily; levels the family declares equal are not required
to be strictly ordered, and a base with a missing score is excluded
rather than counted as a violation.

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 5 | 0.4244 | 0.6858 | 0/5 (0.0000) |
| official | test | 20 | 0.6657 | 0.8402 | 0/20 (0.0000) |
| repair | dev | 5 | 0.6783 | 0.8117 | 2/5 (0.4000) |
| repair | test | 20 | 0.7131 | 0.9131 | 3/20 (0.1500) |

## Contract decomposition

This family declares levels that must tie, so its CPA is a mixture of
two contracts and is dominated by whichever is easier. Rank
gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of
the levels declared equal, and there the only correct prediction is a
tie, so a widening dev margin raises this half without measuring
anything. Read the two halves separately, never the composite alone.

| backend | split | rank gap | pairs | match rate | tie rate |
|---|---|---:|---:|---:|---:|
| official | test (zero-margin) | 0 | 20 | 0.6000 | 0.6000 |
| official | test (zero-margin) | 1 | 100 | 0.4900 | 0.3900 |
| official | test (zero-margin) | 2 | 80 | 0.5750 | 0.3000 |
| official | test (zero-margin) | 3 | 60 | 0.6500 | 0.2333 |
| official | test (zero-margin) | 4 | 40 | 0.8250 | 0.1000 |
| official | test (tie-aware) | 0 | 20 | 0.6000 | 0.6000 |
| official | test (tie-aware) | 1 | 100 | 0.4900 | 0.3900 |
| official | test (tie-aware) | 2 | 80 | 0.5750 | 0.3000 |
| official | test (tie-aware) | 3 | 60 | 0.6500 | 0.2333 |
| official | test (tie-aware) | 4 | 40 | 0.8250 | 0.1000 |
| repair | test (zero-margin) | 0 | 20 | 0.0000 | 0.0000 |
| repair | test (zero-margin) | 1 | 100 | 0.7300 | 0.0000 |
| repair | test (zero-margin) | 2 | 80 | 0.8125 | 0.0000 |
| repair | test (zero-margin) | 3 | 60 | 0.8833 | 0.0000 |
| repair | test (zero-margin) | 4 | 40 | 0.9500 | 0.0000 |
| repair | test (tie-aware) | 0 | 20 | 0.0000 | 0.0000 |
| repair | test (tie-aware) | 1 | 100 | 0.7300 | 0.0000 |
| repair | test (tie-aware) | 2 | 80 | 0.8125 | 0.0000 |
| repair | test (tie-aware) | 3 | 60 | 0.8833 | 0.0000 |
| repair | test (tie-aware) | 4 | 40 | 0.9500 | 0.0000 |

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
| official | dev | rank 0: `conjunction_control`, `occlusion_100` | 2 | 0.6667 | 1.3333 |
| official | test | rank 0: `conjunction_control`, `occlusion_100` | 9 | 0.4922 | 0.9844 |
| repair | dev | rank 0: `conjunction_control`, `occlusion_100` | 5 | 0.2268 | 0.4535 |
| repair | test | rank 0: `conjunction_control`, `occlusion_100` | 20 | 0.1699 | 0.3398 |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value at
every level is insensitive rather than ordered: it cannot detect the
transformation at all, so no level pair can match.

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `conjunction_control` | 25 | 0.1275 | 0.2012 | 0.0000 | 0.5000 | 5 |
| official | `occlusion_000` | 25 | 0.5575 | 0.4294 | 0.0000 | 1.0000 | 9 |
| official | `occlusion_025` | 25 | 0.4800 | 0.4354 | 0.0000 | 1.0000 | 9 |
| official | `occlusion_050` | 25 | 0.4625 | 0.4261 | 0.0000 | 1.0000 | 11 |
| official | `occlusion_075` | 25 | 0.4150 | 0.4031 | 0.0000 | 1.0000 | 11 |
| official | `occlusion_100` | 25 | 0.1825 | 0.3288 | 0.0000 | 1.0000 | 9 |
| repair | `conjunction_control` | 25 | 0.2009 | 0.1517 | 0.0417 | 0.4759 | 25 |
| repair | `occlusion_000` | 25 | 0.5181 | 0.2595 | 0.0583 | 0.9278 | 25 |
| repair | `occlusion_025` | 25 | 0.4928 | 0.2702 | 0.0749 | 0.9275 | 25 |
| repair | `occlusion_050` | 25 | 0.4777 | 0.2702 | 0.0693 | 0.9221 | 25 |
| repair | `occlusion_075` | 25 | 0.4286 | 0.2630 | 0.0471 | 0.9036 | 25 |
| repair | `occlusion_100` | 25 | 0.2436 | 0.2203 | 0.0473 | 0.7085 | 25 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| weakest_object_visibility | 0.5967 | 0.7633 | +0.1666 | [+0.0900, +0.2500] |

Paired zero-margin delta: `+0.1667` over 20 test bases; paired tie-aware delta: `+0.1667`.

## Frame evidence (test split)

Per-clip detection evidence behind the scores. `joint co-presence` is the
fraction of sampled frames in which both targets pass the official 0.5
gate, read from the separate official-threshold pass for the Repair;
`mean frame score` is the repair's SoftMin value averaged over frames
(the official score is that same co-presence rate, so it has no separate
SoftMin column), and `weakest confidence` is the mean per-frame confidence
of the weaker target. A repair whose `weakest confidence` stays well above
zero at full suppression is reading context, not the object.

| backend | clips | frames | mean joint co-presence | mean frame score | mean weakest confidence | zero-frame rate |
|---|---:|---:|---:|---:|---:|---:|
| official | 120 | 1920 | 0.3802 | — | — | — |
| repair | 120 | 1920 | 0.3802 | 0.4074 | 0.3510 | 0.0276 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
