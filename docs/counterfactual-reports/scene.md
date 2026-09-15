# Counterfactual Pair Accuracy: scene

- family: `environment_coverage`
- transformation: 2x2 grid mixing a wrong-scene donor with target-scene quadrants at 0-100%.
- expected relation: Monotone increasing in target coverage.
- bases: 25 (dev 5, test 20)
- derived clips: 125 (dev 25, test 100)
- levels: `coverage_000`, `coverage_025`, `coverage_050`, `coverage_075`, `coverage_100`
- code SHA: `5c4a13091893273510794e45872eac3c59c3ad21`
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
| official | dev (zero-margin) | 0 | 50 | 0.3400 | [0.0600, 0.6200] |
| official | test (zero-margin) | 0 | 200 | 0.3850 | [0.2550, 0.5050] |
| official | test (tie-aware) | 0 | 200 | 0.3850 | [0.2550, 0.5050] |
| repair | dev (zero-margin) | 0 | 50 | 0.9000 | [0.7800, 1.0000] |
| repair | test (zero-margin) | 0 | 200 | 0.9300 | [0.8400, 0.9850] |
| repair | test (tie-aware) | 0 | 200 | 0.9300 | [0.8400, 0.9850] |

## Sequence-level order statistics

Per-base Spearman correlation between the declared rank and the score,
and the fraction of bases whose levels come out in the declared strict
order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank
gaps more heavily; levels the family declares equal are not required
to be strictly ordered, and a base with a missing score is excluded
rather than counted as a violation.

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 5 | 0.4520 | 0.6669 | 0/5 (0.0000) |
| official | test | 20 | 0.7480 | 0.7071 | 0/20 (0.0000) |
| repair | dev | 5 | 0.8400 | 1.0000 | 3/5 (0.6000) |
| repair | test | 20 | 0.8800 | 1.0000 | 14/20 (0.7000) |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value at
every level is insensitive rather than ordered: it cannot detect the
transformation at all, so no level pair can match.

| backend | split | level | n | mean | std | min | max | distinct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| official | dev | `coverage_000` | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | dev | `coverage_025` | 5 | 0.1875 | 0.3750 | 0.0000 | 0.9375 | 2 |
| official | dev | `coverage_050` | 5 | 0.2000 | 0.2915 | 0.0000 | 0.7500 | 3 |
| official | dev | `coverage_075` | 5 | 0.4000 | 0.3636 | 0.0000 | 0.8125 | 3 |
| official | dev | `coverage_100` | 5 | 0.2375 | 0.3881 | 0.0000 | 1.0000 | 3 |
| official | test | `coverage_000` | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |
| official | test | `coverage_025` | 20 | 0.0969 | 0.2908 | 0.0000 | 1.0000 | 3 |
| official | test | `coverage_050` | 20 | 0.1000 | 0.2557 | 0.0000 | 1.0000 | 4 |
| official | test | `coverage_075` | 20 | 0.2125 | 0.3471 | 0.0000 | 1.0000 | 7 |
| official | test | `coverage_100` | 20 | 0.4344 | 0.3995 | 0.0000 | 1.0000 | 6 |
| repair | dev | `coverage_000` | 5 | 0.3618 | 0.0101 | 0.3529 | 0.3805 | 5 |
| repair | dev | `coverage_025` | 5 | 0.3713 | 0.0143 | 0.3540 | 0.3952 | 5 |
| repair | dev | `coverage_050` | 5 | 0.3804 | 0.0139 | 0.3610 | 0.3984 | 5 |
| repair | dev | `coverage_075` | 5 | 0.3886 | 0.0120 | 0.3716 | 0.4047 | 5 |
| repair | dev | `coverage_100` | 5 | 0.3983 | 0.0116 | 0.3837 | 0.4156 | 5 |
| repair | test | `coverage_000` | 20 | 0.3586 | 0.0116 | 0.3363 | 0.3753 | 20 |
| repair | test | `coverage_025` | 20 | 0.3708 | 0.0092 | 0.3566 | 0.3949 | 20 |
| repair | test | `coverage_050` | 20 | 0.3831 | 0.0129 | 0.3648 | 0.4158 | 20 |
| repair | test | `coverage_075` | 20 | 0.3930 | 0.0133 | 0.3636 | 0.4228 | 20 |
| repair | test | `coverage_100` | 20 | 0.4031 | 0.0161 | 0.3661 | 0.4317 | 20 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| environment_coverage | 0.3850 | 0.9300 | +0.5450 | [+0.4249, +0.6800] |

Paired zero-margin delta: `+0.5450` over 20 test bases; paired tie-aware delta: `+0.5450`.

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
