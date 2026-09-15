# Counterfactual Pair Accuracy: multiplt_object

- family: `weakest_object_visibility`
- transformation: Alpha-blend the tracked weaker target towards its local mean at 0-100%.
- expected relation: Monotone decreasing as the weak target disappears.
- bases: 25 (dev 5, test 20)
- derived clips: 150 (dev 30, test 120)
- levels: `occlusion_100`, `conjunction_control`, `occlusion_075`, `occlusion_050`, `occlusion_025`, `occlusion_000`
- code SHA: `bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`

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

This family mixes an inequality half (the counterfactual must move the
score) with an invariance half (relocated variants must tie). A pooled CPA
is dominated by whichever half is easier, so each is scored separately.

| backend | half | pairs | CPA (dev margin) | CPA (zero margin) |
|---|---|---:|---:|---:|
| official | sensitivity | 350 | 0.5371 | 0.5371 |
| official | invariance | 25 | 0.6000 | 0.6000 |
| repair | sensitivity | 350 | 0.8171 | 0.8171 |
| repair | invariance | 25 | 0.0000 | 0.0000 |

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

## Contract decomposition

This family declares more than one expected rank, so its CPA is a
mixture of two contracts and is dominated by whichever is easier. Rank
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

Rank-gap-0 pairs are the family's actual target. Splitting them out
shows whether a Repair gain in the composite comes from sensitivity
(which both backends usually already have) or from the invariant half.

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

Per-level score distribution. A metric with a single distinct value is
insensitive rather than invariant: it cannot detect the transformation at
all, so its CPA on an invariance family is vacuous (plan section 7.4).

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

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| weakest_object_visibility | 0.5967 | 0.7633 | +0.1666 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
