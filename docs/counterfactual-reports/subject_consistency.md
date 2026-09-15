# Counterfactual Pair Accuracy: subject_consistency

- family: `temporal_relocation`
- transformation: One fixed subject-region corruption placed at the start, middle and end.
- expected relation: clean > corrupted, and the three positions should tie.
- bases: 25 (dev 5, test 20)
- derived clips: 100 (dev 20, test 80)
- levels: `corrupt_start`, `corrupt_middle`, `corrupt_end`, `clean`
- code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`

## Score coverage

| backend | scored clips | expected | incomplete shards |
|---|---:|---:|---|
| official | 100 | 100 | none |
| repair | 100 | 100 | none |
| official (dev) | 20 | 20 | — |
| official (test) | 80 | 80 | — |
| repair (dev) | 20 | 20 | — |
| repair (test) | 80 | 80 | — |

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

## Official vs Repair (test, tie-aware)

| metric | official | repair | repair - official |
|---|---:|---:|---:|
| temporal_relocation | 0.5917 | 0.8500 | +0.2583 |

## Status and limitations

- Weights are local; nothing was downloaded and no upstream checkout was modified.
- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these
  scores come from the current metric packages, so treat absolute values as
  first-run measurements rather than reproduced official baselines.
- A clip whose backend raised is recorded as `failed` and stays out of the CPA
  denominator rather than being scored as zero.
