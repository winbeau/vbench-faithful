# Counterfactual Pair Accuracy: spatial_relationship

- family: `directional_flip`
- transformation: Mirror the axis named by the ordered relation, prompt unchanged.
- expected relation: original > flip.
- bases: 40 (dev 10, test 30)
- derived clips: 80 (dev 20, test 60)
- levels: `horizontal_flip`, `vertical_flip`, `original`
- code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 80 | 80 | none |
| repair | 80 | 80 | none |
| official (dev) | 20 | 20 | — |
| official (test) | 60 | 60 | — |
| repair (dev) | 20 | 20 | — |
| repair (test) | 60 | 60 | — |

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
