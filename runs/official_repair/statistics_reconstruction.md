# Frozen E0 Baseline Statistics Reconstruction

## Result

The frozen official predictions, canonical pair metadata, and embedded split column reproduce every requested core baseline statistic to the displayed precision without evaluator inference and without dimension-specific result overrides.

## Inputs and schema

| Input | Relevant columns / confirmed use |
|---|---|
| `data/processed/pairwise_master_split.csv` | `dimension`, `group_id`, `model_a`, `model_b`, `video_a_uid`, `video_b_uid`, `human_label`, `split`; 4 models and the six unordered model pairs per group |
| `splits/e0_prompt_split.csv` | `dimension`, `prompt_id`, `split`, `seed`; its frozen prompt split agrees with the pair metadata split used below |
| `data/processed/e0_scoring_manifest.csv` | `video_uid`, `dimension`, `split`, `prompt_id`, `group_id`, `generator`, `relative_video_path`; canonical per-video inventory |
| `scores/official/e0/<dimension>/results.csv` | `video_uid`, `score`, `status`, plus identity columns; successful, nonempty scores join by `video_uid` |
| `scripts/04_s1_consistency_split.py` | Confirms the four-model / six-pair-per-group construction and prompt-disjoint split generation |

The pair labels are exactly `{0, 0.5, 1}`. Direct comparison against official score differences confirms their direction: `1` means video/model A wins, `0` means B wins, and `0.5` is a human tie.

## Reconstructed rules

For a valid pair, define `d = score_A - score_B`.

- Pair outcome at margin `delta`: A wins if `d > delta`; B wins if `d < -delta`; otherwise tie.
- `Acc0` is exact three-class pair accuracy using `delta = 0`, including human ties.
- Tie margin candidates are the finite dev-derived set `{0} union {|d|}`; select the margin maximizing dev three-class accuracy, then the smallest exact maximizer.
- Tie-aware test accuracy applies that fixed dev margin to valid test pairs.
- Kendall tau-b is `scipy.stats.kendalltau(human_label, raw d)` on valid test pairs.
- Coverage is pairs whose A and B both have successful numeric scores divided by all canonical pairs.
- Model-level Pearson uses four per-model means. Each valid pair contributes metric A-side `outcome(d, 0)`, B-side `1-outcome(d, 0)`, human A-side `human_label`, and B-side `1-human_label`; Pearson is computed from those two four-element vectors.

## Core regression result

| Dimension | Dev Acc0 | Test Acc0 | Delta | Test tie-aware | tau-b | Valid / total | Coverage | Pearson (n=4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dynamic Degree | 0.603448 | 0.684496 | 0.000000 | 0.684496 | 0.461214 | 2160 / 2160 | 1.000000 | 0.815453 |
| Subject Consistency | 0.626437 | 0.584496 | 0.000302 | 0.585271 | 0.382951 | 2160 / 2160 | 1.000000 | 0.960580 |
| Human Action | 0.663333 | 0.553333 | 0.000000 | 0.553333 | 0.265161 | 3000 / 3000 | 1.000000 | 0.904245 |
| Spatial Relationship | 0.531429 | 0.505442 | 0.344890 | 0.525170 | 0.291660 | 2520 / 3240 | 0.777778 | 0.725736 |

All values match the supplied baseline after normal display rounding (absolute difference below 0.001). Spatial coverage follows from 720 unavailable pairs, not a hard-coded percentage.

## Confidence intervals

The original CI sampling unit, bootstrap count, seed, and percentile variant cannot be uniquely inferred from retained point estimates. They are not claimed as recovered. The reconstruction fixes a reproducible forward protocol: pair-level iid resampling with replacement, percentile 95% intervals, 2,000 replicates, base seed `2026` (`2027` is the deterministic second stream for tie-aware accuracy). The emitted CI values support consistent future official-versus-repair comparisons, but are not asserted to be the lost original CI implementation.

## Artifacts and verification

- Implementation: `scripts/evaluate_pairwise_statistics.py`
- Regression test: `tests/test_pairwise_statistics.py`
- Metrics: `runs/official_repair/reconstructed_baseline_metrics.csv`
- Detailed model vectors and metrics: `runs/official_repair/reconstructed_baseline_details.json`
- Command: `./.venv/bin/python scripts/evaluate_pairwise_statistics.py --bootstrap-iterations 2000 --seed 2026`

`pytest` is not installed in the existing `.venv`; the same test function was loaded and executed directly after `py_compile`, and passed. No model inference was launched.

## Pearson input vectors

Model order is `cogvideo, lavie, modelscope, videocraft`. Each pair below is `(metric mean, human mean)`; these are the two four-element inputs to Pearson.

| Dimension | cogvideo | lavie | modelscope | videocraft |
|---|---:|---:|---:|---:|
| Dynamic Degree | (0.372685185, 0.308796296) | (0.418981481, 0.531018519) | (0.526388889, 0.538425926) | (0.681944444, 0.621759259) |
| Subject Consistency | (0.575000000, 0.557407407) | (0.674074074, 0.682407407) | (0.497222222, 0.591203704) | (0.253703704, 0.168981481) |
| Human Action | (0.418666667, 0.409333333) | (0.545333333, 0.581333333) | (0.516000000, 0.530666667) | (0.520000000, 0.478666667) |
| Spatial Relationship | (0.379761905, 0.453174603) | (0.522619048, 0.457142857) | (0.532142857, 0.527380952) | (0.565476190, 0.562301587) |
