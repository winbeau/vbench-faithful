# Counterfactual Pair Accuracy: dynamics_degree

- family: `fps_resampling`
- transformation: Resample one trajectory to 8/6/4/2 fps with the duration held fixed.
- expected relation: Invariance: every rung should score the same.
- bases: 40 (dev 10, test 30)
- derived clips: 160 (dev 40, test 120)
- levels: `fps4`, `fps8`, `fps6`, `fps2`
- code SHA: `8bf0ed5d5874bd27f364177fb53b66434da14d62`
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

## CPA by contract half

This family mixes an inequality half (the counterfactual must move the
score) with an invariance half (relocated variants must tie). A pooled CPA
is dominated by whichever half is easier, so each is scored separately.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) |
|---|---|---:|---:|---:|
| official | invariance | 240 | 0.8500 | 0.0000 |
| repair | invariance | 240 | 0.8042 | 0.0000 |

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

## Score sensitivity

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

| backend | level | n | mean | std | min | max | distinct |
|---|---|---:|---:|---:|---:|---:|---:|
| official | `fps2` | 40 | 32.8744 | 30.2231 | 0.5438 | 153.1937 | 40 |
| official | `fps4` | 40 | 24.2540 | 26.6125 | 0.3743 | 153.0603 | 40 |
| official | `fps6` | 40 | 19.7201 | 23.8540 | 0.3515 | 141.0087 | 40 |
| official | `fps8` | 40 | 16.5108 | 19.8936 | 0.3038 | 105.4677 | 40 |
| repair | `fps2` | 40 | 0.0906 | 0.0807 | 0.0010 | 0.2853 | 40 |
| repair | `fps4` | 40 | 0.0932 | 0.0868 | 0.0010 | 0.3790 | 40 |
| repair | `fps6` | 40 | 0.0910 | 0.0870 | 0.0011 | 0.4214 | 40 |
| repair | `fps8` | 40 | 0.0878 | 0.0896 | 0.0011 | 0.4185 | 40 |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| fps_resampling | 0.8333 | 0.7722 | -0.0611 |

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
