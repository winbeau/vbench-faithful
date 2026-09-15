# Counterfactual Pair Accuracy: dynamics_degree

- family: `fps_resampling`
- transformation: Resample one trajectory to 8/6/4/2 fps with the duration held fixed.
- expected relation: Invariance: every rung should score the same.
- bases: 40 (dev 10, test 30)
- derived clips: 160 (dev 40, test 120)
- levels: `fps6`, `fps2`, `fps8`, `fps4`
- code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 160 | 160 | none |
| repair | 160 | 160 | none |
| official (dev) | 40 | 40 | — |
| official (test) | 120 | 120 | — |
| repair (dev) | 40 | 40 | — |
| repair (test) | 120 | 120 | — |

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
| repair | test (tie-aware) | 0.1466 | 180 | 0.8444 | [0.7388, 0.9333] |

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| fps_resampling | 0.8333 | 0.8444 | +0.0111 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
