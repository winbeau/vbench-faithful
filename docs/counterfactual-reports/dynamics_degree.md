# Counterfactual Pair Accuracy: dynamics_degree

- family: `fps_resampling`
- transformation: Resample one trajectory to 8/6/4/2 fps with the duration held fixed.
- expected relation: Invariance: every rung should score the same.
- bases: 40 (dev 10, test 30)
- derived clips: 160 (dev 40, test 120)
- levels: `fps6`, `fps4`, `fps2`, `fps8`
- code SHA: `5c4a13091893273510794e45872eac3c59c3ad21`
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 160 | 160 | none |
| repair | 160 | 160 | none |
| official (dev) | 40 | 40 | — |
| official (test) | 120 | 120 | — |
| repair (dev) | 40 | 40 | — |
| repair (test) | 120 | 120 | — |

## Sampling-interval response (primary)

The contract for this family is `score must not depend on the sampling
interval`, i.e. a signed log-log slope of `p = 0`. This is the primary
diagnostic: unlike the unsigned dispersion below it can tell a score that
inflates at low frame rates from one that shrinks.

| backend | fitted p (target 0) | mean per-clip p | sd | levels (score vs rung) |
|---|---:|---:|---:|---|
| official | +0.4908 | 0.5562 | 0.3492 | fps2=32.8744, fps4=24.2540, fps6=19.7201, fps8=16.5108 |
| repair | +0.0187 | 0.0565 | 0.3769 | fps2=0.0906, fps4=0.0932, fps6=0.0910, fps8=0.0878 |

## CPA

`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated
margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.

| backend | split | margin | pairs | CPA | 95% CI |
|---|---|---:|---:|---:|---|
| official | dev (zero-margin) | 0 | 60 | 0.0000 | [0.0000, 0.0000] |
| official | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |
| official | test (tie-aware) | 20.14 | 180 | 0.8333 | [0.7444, 0.9111] |
| repair | dev (zero-margin) | 0 | 60 | 0.0000 | [0.0000, 0.0000] |
| repair | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |
| repair | test (tie-aware) | 0.03721 | 180 | 0.7722 | [0.6665, 0.8667] |

## Invariance statistics

Every level of this family carries the same expected rank, so the
tie-aware CPA above is degenerate — widening the dev margin until all
differences count as ties yields CPA 1.0 regardless of how unstable the
metric is. The dispersion below is the meaningful invariance measure
(plan sections 6.4 and 8.3).

| backend | split | bases | mean within-base CV | mean relative range |
|---|---|---:|---:|---:|
| official | dev | 10 | 0.3838 | 0.9862 |
| official | test | 30 | 0.3177 | 0.8196 |
| repair | dev | 10 | 0.1749 | 0.4375 |
| repair | test | 30 | 0.1822 | 0.4760 |

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | split | level | n | mean | std | min | max | distinct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| official | dev | `fps2` | 10 | 25.8855 | 19.0859 | 1.4977 | 61.0185 | 10 |
| official | dev | `fps4` | 10 | 17.6140 | 13.5988 | 1.0404 | 38.2962 | 10 |
| official | dev | `fps6` | 10 | 12.7531 | 10.0611 | 0.5916 | 30.0854 | 10 |
| official | dev | `fps8` | 10 | 10.7698 | 9.4788 | 0.5281 | 28.2659 | 10 |
| official | test | `fps2` | 30 | 35.2040 | 32.7838 | 0.5438 | 153.1937 | 30 |
| official | test | `fps4` | 30 | 26.4673 | 29.3779 | 0.3743 | 153.0603 | 30 |
| official | test | `fps6` | 30 | 22.0424 | 26.5212 | 0.3515 | 141.0087 | 30 |
| official | test | `fps8` | 30 | 18.4244 | 21.9790 | 0.3038 | 105.4677 | 30 |
| repair | dev | `fps2` | 10 | 0.0845 | 0.0838 | 0.0023 | 0.2735 | 10 |
| repair | dev | `fps4` | 10 | 0.0824 | 0.0778 | 0.0024 | 0.2056 | 10 |
| repair | dev | `fps6` | 10 | 0.0733 | 0.0732 | 0.0016 | 0.2067 | 10 |
| repair | dev | `fps8` | 10 | 0.0721 | 0.0781 | 0.0017 | 0.2170 | 10 |
| repair | test | `fps2` | 30 | 0.0927 | 0.0795 | 0.0010 | 0.2853 | 30 |
| repair | test | `fps4` | 30 | 0.0968 | 0.0894 | 0.0010 | 0.3790 | 30 |
| repair | test | `fps6` | 30 | 0.0970 | 0.0904 | 0.0011 | 0.4214 | 30 |
| repair | test | `fps8` | 30 | 0.0930 | 0.0925 | 0.0011 | 0.4185 | 30 |

## Official vs Repair (test, tie-aware)

`repair - official` is the paired difference over the same `base_id`
clusters; its 95% CI resamples those clusters once and re-scores both
backends on each resample (plan 5.4). The interval is what decides whether
the delta is distinguishable from zero — the two marginal intervals in the
`CPA` table do not.

| metric | official | repair | repair - official | paired 95% CI |
|---|---:|---:|---:|---|
| fps_resampling | 0.8333 | 0.7722 | -0.0611 | [+0.0000, +0.0000] |

Paired zero-margin delta: `+0.0000` over 30 test bases; paired tie-aware delta: `-0.0611`.

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
