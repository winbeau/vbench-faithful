# Table 2 — Counterfactual Pair Accuracy (VBench-CF)

Official VBench 1.0 versus the repaired/Audit metric on metamorphic
counterfactuals. CPA is over all ordered level pairs within a family; the
interval is a 95% cluster bootstrap over `base_id`; `delta` is Repair minus
Official on the test split.

Scoring code SHA: `66c4a99dba05aceaebe80276ffbffc607c3d2e40`.

| dimension | family | bases | clips (test) | Official CPA | Repair CPA | delta | delta 95% CI (paired) |
|---|---|---:|---:|---:|---:|---:|---|
| dynamics_degree | `fps_resampling` | 40 | 120 | 0.8333 | 0.7722 | -0.0611 | — |
| subject_consistency | `temporal_relocation` | 25 | 80 | 0.5917 | 0.8500 | +0.2583 | — |
| human_action | `filename_invariance` | 25 | 60 | 1.0000 | 1.0000 | +0.0000 | — |
| spatial_relationship | `directional_flip` | 40 | 60 | 0.3667 | 0.0667 | -0.3000 | — |
| scene | `environment_coverage` | 25 | 100 | 0.3850 | 0.9300 | +0.5450 | — |
| multiplt_object | `weakest_object_visibility` | 25 | 120 | 0.5967 | 0.7633 | +0.1666 | — |
| motion_smoothness | `temporal_jerk` | 25 | 100 | 0.8300 | 0.8800 | +0.0500 | [-0.0150, +0.1100] |

## Invariance families — dispersion, not CPA

For these families every level shares one expected rank, so a tie-margin
CPA is degenerate: widening the dev margin until every difference counts
as a tie yields 1.0 regardless of how unstable the metric is. The
within-base coefficient of variation is the meaningful number, and a
Repair CV of 0 with a large Official CV is the actual result.

| dimension | Official CV | Repair CV | Official CPA | Repair CPA |
|---|---:|---:|---:|---:|
| dynamics_degree | 0.3177 | 0.1822 | 0.8333 | 0.7722 |
| human_action | 1.4142 | 0.0000 | 1.0000 | 1.0000 |

## Coverage

| dimension | Official scored / test | Repair scored / test |
|---|---:|---:|
| dynamics_degree | 120 / 120 | 120 / 120 |
| subject_consistency | 80 / 80 | 80 / 80 |
| human_action | 60 / 60 | 60 / 60 |
| spatial_relationship | 60 / 60 | 60 / 60 |
| scene | 100 / 100 | 100 / 100 |
| multiplt_object | 120 / 120 | 120 / 120 |
| motion_smoothness | 100 / 100 | 100 / 100 |

## Limitations

- These are **first-run measurements from the current metric packages**. Model,
  CUDA and weight parity against the frozen E0 baselines has **not** been
  verified, so absolute values are not reproduced official numbers.
- Clips whose backend raised are recorded as `failed` and excluded from the CPA
  denominator; they are never scored as zero, so coverage is reported separately.
- A same-rank family's CPA must not be quoted without its dispersion statistics.
- Overall Consistency is out of scope for this round.
