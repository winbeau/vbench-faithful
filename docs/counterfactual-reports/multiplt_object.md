# Counterfactual Pair Accuracy: multiplt_object

- family: `weakest_object_visibility`
- transformation: Alpha-blend the tracked weaker target towards its local mean at 0-100%.
- expected relation: Monotone decreasing as the weak target disappears.
- bases: 25 (dev 5, test 20)
- derived clips: 150 (dev 30, test 120)
- levels: `occlusion_100`, `conjunction_control`, `occlusion_075`, `occlusion_050`, `occlusion_025`, `occlusion_000`
- control levels (excluded from the ordered ladder): `conjunction_control`
- code SHA: `c73495e6ffdc44b96d55e1e29d7eab21766c080b`
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

| backend | split | half | pairs | CPA (dev margin) | CPA (zero margin) | 95% CI |
|---|---|---|---:|---:|---:|---|
| official | dev | sensitivity | 50 | 0.2200 | 0.2200 | [0.0000, 0.5000] |
| official | test | sensitivity | 200 | 0.5450 | 0.5450 | [0.4300, 0.6500] |
| repair | dev | sensitivity | 50 | 0.8200 | 0.8200 | [0.6400, 0.9800] |
| repair | test | sensitivity | 200 | 0.7850 | 0.7850 | [0.6900, 0.8650] |

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

## Weak-target construction check

Median area of the suppressed target B as a fraction of the frame, over
the test bases. This is the premise check for the occlusion ladder: the
severity levels only mean something if B occupies enough of the frame
that suppressing it is visible at all.

- bases: 20
- median area ratio: **0.1575** (IQR 0.0350–0.3446)
- range: 0.0066 – 0.4101


## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 50 | 0.2200 | [0.0000, 0.5000] |
| official | test (zero-margin) | 0 | 200 | 0.5450 | [0.4300, 0.6500] |
| official | test (tie-aware) | 0 | 200 | 0.5450 | [0.4300, 0.6500] |
| repair | dev (zero-margin) | 0 | 50 | 0.8200 | [0.6400, 0.9800] |
| repair | test (zero-margin) | 0 | 200 | 0.7850 | [0.6900, 0.8650] |
| repair | test (tie-aware) | 0 | 200 | 0.7850 | [0.6900, 0.8650] |

## Sequence-level order statistics

Per-base Spearman correlation between the declared rank and the score,
and the fraction of bases whose levels come out in the declared strict
order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank
gaps more heavily; levels the family declares equal are not required
to be strictly ordered, and a base with a missing score is excluded
rather than counted as a violation.

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 5 | 0.8008 | 0.8008 | 0/5 (0.0000) |
| official | test | 20 | 0.6204 | 0.7225 | 0/20 (0.0000) |
| repair | dev | 5 | 0.7000 | 0.9000 | 2/5 (0.4000) |
| repair | test | 20 | 0.6550 | 0.9000 | 3/20 (0.1500) |

## Contract decomposition

This family declares levels that must tie, so its CPA is a mixture of
two contracts and is dominated by whichever is easier. Rank
gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of
the levels declared equal, and there the only correct prediction is a
tie, so a widening dev margin raises this half without measuring
anything. Read the two halves separately, never the composite alone.

| backend | split | rank gap | pairs | match rate | tie rate |
|---|---|---:|---:|---:|---:|
| official | test (zero-margin) | 1 | 80 | 0.4500 | 0.4250 |
| official | test (zero-margin) | 2 | 60 | 0.5167 | 0.3500 |
| official | test (zero-margin) | 3 | 40 | 0.6250 | 0.2500 |
| official | test (zero-margin) | 4 | 20 | 0.8500 | 0.1000 |
| official | test (tie-aware) | 1 | 80 | 0.4500 | 0.4250 |
| official | test (tie-aware) | 2 | 60 | 0.5167 | 0.3500 |
| official | test (tie-aware) | 3 | 40 | 0.6250 | 0.2500 |
| official | test (tie-aware) | 4 | 20 | 0.8500 | 0.1000 |
| repair | test (zero-margin) | 1 | 80 | 0.7000 | 0.0000 |
| repair | test (zero-margin) | 2 | 60 | 0.7833 | 0.0000 |
| repair | test (zero-margin) | 3 | 40 | 0.8750 | 0.0000 |
| repair | test (zero-margin) | 4 | 20 | 0.9500 | 0.0000 |
| repair | test (tie-aware) | 1 | 80 | 0.7000 | 0.0000 |
| repair | test (tie-aware) | 2 | 60 | 0.7833 | 0.0000 |
| repair | test (tie-aware) | 3 | 40 | 0.8750 | 0.0000 |
| repair | test (tie-aware) | 4 | 20 | 0.9500 | 0.0000 |

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

| backend | split | level | n | mean | std | min | max | distinct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| official | dev | `occlusion_000` | 5 | 0.4500 | 0.4583 | 0.0000 | 1.0000 | 3 |
| official | dev | `occlusion_025` | 5 | 0.4250 | 0.4717 | 0.0000 | 1.0000 | 3 |
| official | dev | `occlusion_050` | 5 | 0.4000 | 0.4899 | 0.0000 | 1.0000 | 2 |
| official | dev | `occlusion_075` | 5 | 0.4000 | 0.4899 | 0.0000 | 1.0000 | 2 |
| official | dev | `occlusion_100` | 5 | 0.2000 | 0.4000 | 0.0000 | 1.0000 | 2 |
| official | test | `occlusion_000` | 20 | 0.5844 | 0.4175 | 0.0000 | 1.0000 | 9 |
| official | test | `occlusion_025` | 20 | 0.4938 | 0.4248 | 0.0000 | 1.0000 | 9 |
| official | test | `occlusion_050` | 20 | 0.4781 | 0.4071 | 0.0000 | 1.0000 | 11 |
| official | test | `occlusion_075` | 20 | 0.4188 | 0.3782 | 0.0000 | 1.0000 | 11 |
| official | test | `occlusion_100` | 20 | 0.1781 | 0.3083 | 0.0000 | 0.9375 | 8 |
| repair | dev | `occlusion_000` | 5 | 0.4435 | 0.3124 | 0.1609 | 0.9278 | 5 |
| repair | dev | `occlusion_025` | 5 | 0.4252 | 0.3326 | 0.1493 | 0.9275 | 5 |
| repair | dev | `occlusion_050` | 5 | 0.4047 | 0.3427 | 0.0693 | 0.9221 | 5 |
| repair | dev | `occlusion_075` | 5 | 0.3846 | 0.3506 | 0.0614 | 0.9036 | 5 |
| repair | dev | `occlusion_100` | 5 | 0.2235 | 0.2380 | 0.0693 | 0.6967 | 5 |
| repair | test | `occlusion_000` | 20 | 0.5368 | 0.2409 | 0.0583 | 0.8780 | 20 |
| repair | test | `occlusion_025` | 20 | 0.5097 | 0.2494 | 0.0749 | 0.8769 | 20 |
| repair | test | `occlusion_050` | 20 | 0.4959 | 0.2454 | 0.0744 | 0.8477 | 20 |
| repair | test | `occlusion_075` | 20 | 0.4396 | 0.2348 | 0.0471 | 0.8288 | 20 |
| repair | test | `occlusion_100` | 20 | 0.2487 | 0.2153 | 0.0473 | 0.7085 | 20 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| weakest_object_visibility | 0.5450 | 0.7850 | +0.2400 | [+0.1550, +0.3300] |

Paired zero-margin delta: `+0.2400` over 20 test bases; paired tie-aware delta: `+0.2400`.

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
