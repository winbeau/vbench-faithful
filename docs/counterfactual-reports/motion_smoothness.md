# Counterfactual Pair Accuracy: motion_smoothness

- family: `temporal_jerk`
- transformation: Duplicate, skip and locally reverse short segments; frame count and rate fixed.
- expected relation: Monotone decreasing in perturbation severity.
- bases: 25 (dev 5, test 20)
- derived clips: 125 (dev 25, test 100)
- levels: `jerk_4_multiple`, `jerk_3_local_reverse`, `jerk_2_duplicate_skip`, `jerk_1_duplicate`, `jerk_0_original`
- code SHA: `feeb77028c5c987b18a6cb2f27a402d757241d5d`
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 125 | 125 | none |
| repair | 125 | 125 | none |
| official (dev) | 25 | 25 | — |
| official (test) | 100 | 100 | — |
| repair (dev) | 25 | 25 | — |
| repair (test) | 100 | 100 | — |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 50 | 0.7600 | [0.5200, 0.9600] |
| official | test (zero-margin) | 0 | 200 | 0.8300 | [0.7700, 0.8900] |
| official | test (tie-aware) | 0 | 200 | 0.8300 | [0.7700, 0.8900] |
| repair | dev (zero-margin) | 0 | 50 | 0.8600 | [0.6800, 0.9800] |
| repair | test (zero-margin) | 0 | 200 | 0.8800 | [0.8050, 0.9450] |
| repair | test (tie-aware) | 0 | 200 | 0.8800 | [0.8050, 0.9450] |

## Sequence-level order statistics

Per-base Spearman correlation between the declared rank and the score,
and the fraction of bases whose levels come out in the declared strict
order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank
gaps more heavily; levels the family declares equal are not required
to be strictly ordered, and a base with a missing score is excluded
rather than counted as a violation.

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 5 | 0.5600 | 0.7000 | 2/5 (0.4000) |
| official | test | 20 | 0.7650 | 0.7500 | 7/20 (0.3500) |
| repair | dev | 5 | 0.7400 | 0.9000 | 2/5 (0.4000) |
| repair | test | 20 | 0.8150 | 0.9500 | 10/20 (0.5000) |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value at
every level is insensitive rather than ordered: it cannot detect the
transformation at all, so no level pair can match.

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `jerk_0_original` | 25 | 0.9565 | 0.0321 | 0.8475 | 0.9851 | 25 |
| official | `jerk_1_duplicate` | 25 | 0.9559 | 0.0324 | 0.8460 | 0.9856 | 25 |
| official | `jerk_2_duplicate_skip` | 25 | 0.9545 | 0.0338 | 0.8446 | 0.9866 | 25 |
| official | `jerk_3_local_reverse` | 25 | 0.9482 | 0.0383 | 0.8418 | 0.9845 | 25 |
| official | `jerk_4_multiple` | 25 | 0.9378 | 0.0474 | 0.8035 | 0.9835 | 25 |
| repair | `jerk_0_original` | 25 | 0.5407 | 0.1156 | 0.4067 | 0.9207 | 25 |
| repair | `jerk_1_duplicate` | 25 | 0.5011 | 0.0689 | 0.4184 | 0.6838 | 25 |
| repair | `jerk_2_duplicate_skip` | 25 | 0.4855 | 0.0515 | 0.4198 | 0.6232 | 25 |
| repair | `jerk_3_local_reverse` | 25 | 0.4760 | 0.0523 | 0.4100 | 0.6077 | 25 |
| repair | `jerk_4_multiple` | 25 | 0.4268 | 0.0330 | 0.3875 | 0.5242 | 25 |

## Repair continuity components (dev + test)

The repair's own mean and upper-tail discontinuity per level (plan 13.3).
`D_video = (1 - tail_weight) * D_mean + tail_weight * D_tail`, so a level
whose mean is flat while its tail rises is a localised failure the score
alone would hide.

| level | n | mean D_mean | mean D_tail | sd(D_mean) |
|---|---:|---:|---:|---:|
| `jerk_0_original` | 25 | 0.5646 | 0.8433 | 0.1856 |
| `jerk_1_duplicate` | 25 | 0.6056 | 0.9818 | 0.1582 |
| `jerk_2_duplicate_skip` | 25 | 0.6314 | 1.0173 | 0.1310 |
| `jerk_3_local_reverse` | 25 | 0.6393 | 1.0744 | 0.1473 |
| `jerk_4_multiple` | 25 | 0.7515 | 1.1627 | 0.0991 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| temporal_jerk | 0.8300 | 0.8800 | +0.0500 | [-0.0150, +0.1100] |

Paired zero-margin delta: `+0.0500` over 20 test bases; paired tie-aware delta: `+0.0500`.

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
