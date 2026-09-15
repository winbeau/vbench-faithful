# Review: `motion_smoothness` counterfactual + natural-preference evidence

Scope, 2026-09-16 (adversarial re-review; supersedes the 2026-09-15 revision):

- `docs/counterfactual-reports/CONSOLIDATED.md` (Raw result row + notes),
  `motion_smoothness.md`, `P1_NATURAL_AND_CONTROL_RUNS.md` §P1.2, `README.md`,
  `table2.csv`/`table2.json`/`SUMMARY.md`.
- Estimator: `metrics/motion-smoothness/src/motion_smoothness/{metric,schemas}.py`
  and `backends/audit.py`; the defaults changed at `4d53fa2`, the frozen scores
  were produced at `feeb770`, the report file was regenerated at `5c4a130`.
- Statistics: `scripts/counterfactual/cpa.py`, `scripts/counterfactual/run_dimension.py`,
  `scripts/evaluate_pairwise_statistics.py`, `scripts/evaluate_paired_backend_delta.py`.
- Frozen trees on `h100-server`: `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/`
  and `/root/wenbiao_zhao/datasets/natural-preference-runs/`.

Verdict: **the `motion_smoothness` row must not be written as a repair win, and the
`4d53fa2` default change is not yet evidence-based.** Three conclusions hold
simultaneously and are all measured below:

1. On the counterfactual ladder the direction-aware estimator is direction-correct
   *within a base* (all four adjacent contrasts positive on average; CPA 0.8300 →
   0.8800 for `feeb770`, and a replay of `4d53fa2` predicts 0.9150).
2. On the natural preference set the same `feeb770` estimator is **significantly
   below chance** (tie-aware 0.3248 against Official 0.6364, paired Δ −0.3116
   [−0.3605, −0.2643] with the correct prompt-clustered unit). This is not a bug:
   the natural label is largely "which clip changes less between frames", and the
   repair score increases with frame-to-frame change.
3. `4d53fa2` has **no end-to-end measurement at all**. Its support is the synthetic
   ladder, the unit tests, and a component replay that predicts test CPA 0.9150
   (Δ +0.0850 [+0.015, +0.155]) — but that same replay shows the strict-order rate
   *falling* from 10/20 to 8/20, so the ladder problem is reduced, not fixed.

Two harness defects are reported in §5 and must be fixed before any of these
numbers is quoted.

## 0. Revision map (why every motion number is stale)

| artifact | revision | when | status |
|---|---|---|---|
| v1 estimator scores (`Δ −0.1050`) | `a044ac9` | — | archived |
| direction-aware scores, counterfactual | `feeb770` | 2026-09-15 18:35 CST | **frozen, reproduced exactly** |
| `motion_smoothness.md` regenerated (`--report-only`) | `5c4a130` | 21:01 CST | report file only |
| natural-preference repair scores (`0.3248`) | `ccbd89b` | 21:57–22:08 CST | **frozen, reproduced exactly** |
| estimator defaults changed | `4d53fa2` | 23:43 CST | **never scored** |
| docs marked pre-`4d53fa2` | `52e4b76` | 23:44 CST | last motion commit; sibling reviews continued on top |

`git log -- metrics/motion-smoothness/src/motion_smoothness/metric.py` is exactly
`1878d39`, `907bc31`, `4d53fa2`, so the counterfactual row (`feeb770`) and the
natural row (`ccbd89b`) were produced by the **same** estimator. The tension in
question 2 is therefore internal to one revision, not a revision mismatch.

Provenance defect found while building this table: the report's `- code SHA:` line
is written by `run_dimension.py:1287` as `git_sha()` at **render** time, and the
score rows carry no revision at all (verified: no `code_sha` key in either frozen
`motion_smoothness__{official,repair}.jsonl`). The published report therefore names
`5c4a130` — a report-generator commit — while the scores it describes were produced
at `feeb770`. A `--report-only` rerun silently relabels the scoring revision. This
must be fixed (write the scoring revision into the score rows, or split the two
fields) before any revision caveat can be trusted.

## 1. Q1 — does `4d53fa2` fix the ladder-realizability problem?

**What is verified.**

- Synthetic ladder: on the canonical constant/ramp/sinusoid trajectories (18
  settings), the new defaults order `L0 > L1 > L2 > L3 > L4` strictly in **18/18**,
  with an `L2 − L3` margin of +0.26 to +0.29 in score.
- Unit tests: `metrics/motion-smoothness/tests/test_temporal_jerk_ladder.py` pins
  the order, the load-bearing direction term, the shipped defaults, and the legacy
  `mean_tail` ablation.

**What the synthetic evidence cannot show.** The synthetic trajectories use
uniform, noise-free flow with a single motion scale; they are exactly the regime in
which a scale-free ratio behaves. Real clips have content motion, flow noise and a
motion scale spanning three orders of magnitude (measured below: `speed_scale`
0.47–1256 px/frame across the 125 derivative clips).

**Real-clip evidence that does exist (a replay, not a score).** The per-transition
components for all 125 `temporal_jerk` clips were cached from a real RAFT pass
(`/root/wenbiao_zhao/tmp/ms_components4/`). Replaying the `feeb770` spec from that
cache reproduces **every frozen repair score to 1.7e-16** (`max|diff| = 1.665e-16`,
`mean|diff| = 2.0e-17`, n=125), which both proves the cache is faithful and proves
the frozen scores are the `feeb770` estimator. Replaying `4d53fa2` from the same
cache gives:

| statistic (test, 20 bases) | Official | Repair `feeb770` (frozen) | Repair `4d53fa2` (replay) |
|---|---:|---:|---:|
| CPA (all 10 pairs) | 0.8300 | 0.8800 | **0.9150** |
| paired Δ vs Official | — | +0.0500 [−0.015, +0.110] | **+0.0850 [+0.015, +0.155]** |
| `L2 > L3` bases | **19/20** | 15/20 | **17/20** |
| strict order `L0>…>L4` | 7/20 | **10/20** | **8/20** |
| `L0>L1` / `L1>L2` / `L3>L4` | 8 / 11 / 20 | 18 / 15 / 18 | 18 / 15 / 18 |
| level means (L0…L4) | — | — | 0.4346 / 0.3732 / 0.3497 / 0.3276 / 0.2750 |

**Answer.** Synthetic + unit evidence is **not sufficient**, and the replay shows
why: `L2 > L3` improves from 15/20 to 17/20 and CPA from 0.880 to 0.915, but the
strict-order rate — the statistic plan 5.2 asks for — **drops from 10/20 to
8/20**. The joint pattern changed on 8 of 20 bases: 3 bases gained exactly the
`L2 > L3` pair (`1101→1111`, `1001→1011`, `0101→1101`) and 5 lost a *different*
adjacent pair, three of them abandoning a previously strict `1111`
(`1111→1011` twice, `1111→0111` once). Top-k aggregation sharpens the average pair
decision while making a single noisy transition more likely to break the full
chain, so the two published order statistics move in opposite directions.
The required real-clip evidence is therefore:

1. an **end-to-end re-score** of the 125 clips at `4d53fa2` (not a replay), with
   the per-base `L2 − L3` sign count, the strict-order rate and the paired CI;
2. the **natural-set re-run** at `4d53fa2` (the replay cannot produce it — those
   1 440 videos have no cached components);
3. a statement of which statistic is the headline. CPA and strict order move in
   opposite directions here, so quoting only CPA hides the regression.

## 2. Q2 — can ladder parity and natural below-chance both be true?

**Yes. They are different statistics against different targets, and the mechanism
is measurable.** Three independent measurements on the frozen data explain it.

### 2.1 CPA is a within-base statistic; the nuisance is between-base

Variance decomposition of the counterfactual test scores (test split, 20 bases ×
5 levels):

| backend | variance share between bases | within base | declared-rank vs within-base level means |
|---|---:|---:|---:|
| Official | 0.9224 | 0.0776 | Spearman **+1.000** |
| Repair `feeb770` | 0.5776 | 0.4224 | Spearman **+1.000** |

Both backends order the levels perfectly *on average within a base*. Official does
so with a within-base effect of only ±0.006…−0.014 riding on 92 % between-base
variance; the repair's effect is roughly 4–9× larger (±0.054…−0.059) but carries
42 % within-base noise, which is what costs it per-base pairs. Every pair CPA
scores compares two levels of the *same* source clip, so any clip-level nuisance
cancels inside the comparison. The natural set compares *different* clips, so it
does not.

### 2.2 The repair score tracks the clip's motion amount, in the wrong direction

Measured on the cached real-clip components (old defaults):

```text
spearman(speed_scale, D_video) = -0.3318     # slower clip -> higher discontinuity
spearman(speed_scale, score)   = +0.3318     # slower clip -> lower "smoothness"
mean score by speed_scale quartile: Q1 0.4628 | Q2 0.4623 | Q3 0.4829 | Q4 0.5345
```

The promoted scale-free normalisation `|Δv| / (|v_t| + |v_{t-1}| + eps)` is
scale-free only where the denominator is signal; near zero it divides flow noise by
flow noise and returns values near 1. Static and slow clips therefore look
*maximally* discontinuous. On the natural videos the same dependence is present and
in the same direction: Spearman(repair score, mean-absolute-frame-difference) =
**+0.2479** overall, **+0.1914** within `.mp4`, **+0.1533** within `.gif`. By MAD
decile the repair score *rises* from 0.4838 (decile 1) to 0.5947 (decile 8).

### 2.3 Human "motion smoothness" is largely "changes less", and Official tracks it

Cheap CPU baseline: mean absolute grayscale frame difference (MAD) over ≤12 frames,
used as a pairwise score with "lower MAD = smoother" and a dev-calibrated tie
margin, on the same 2 160 pairs and split:

| backend | dev margin | test acc0 | tie-aware | tie-aware 95 % CI (pair) | 95 % CI (prompt-clustered) |
|---|---:|---:|---:|---|---|
| Official | 0.0215 | 0.5682 | **0.6364** | [0.6101, 0.6636] | [0.5961, 0.6744] |
| lower-MAD CPU baseline | 3.6924 | 0.5550 | **0.6000** | [0.5736, 0.6264] | [0.5674, 0.6295] |
| Repair `feeb770` | 0.1459 | 0.3388 | **0.3248** | [0.2992, 0.3512] | [0.2907, 0.3628] |

On the 1 501 pairs with a non-tie human label, "the lower-MAD clip is smoother"
agrees with the human label **0.7935** of the time; Official agrees **0.8075**; the
repair agrees **0.4857**. Official's score is a near-monotone function of the same
cue (Spearman(Official, MAD) = **−0.9451**), which is why its accuracy is high.

**Mechanism, stated plainly.** The natural label for this dimension is dominated by
"which clip has less frame-to-frame change" (a static CPU statistic reproduces
56–60 % tie-aware and 79 % of decisive pairs). Official's AMT interpolation error
is an almost perfect proxy for that cue. The repair measures a different thing —
normalised flow-field continuity — whose between-clip variation is driven by motion
scale *opposite* to the human cue, so it lands below chance. No sign bug or plumbing
error is needed to explain the result: predictions.csv matches the audit rows
value-for-value on all 1 440 videos, the Official path reproduces .6364 exactly, and
the below-chance number survives the correct (prompt-clustered) interval.

The two results are therefore compatible, and the counterfactual row is the weaker
evidence: a family whose levels are derived from one source clip cannot detect a
metric that is wrong about every *other* clip.

## 3. Q3 — is this still a "motion smoothness repair"? What should the row say?

**No replacement claim is defensible for this dimension, and the word "repair"
should be dropped from the row.** The estimator is an *audit instrument* with a
demonstrated property and a demonstrated failure:

- it can be made to satisfy the injected within-clip ordering (a real, if partial,
  result: `L2 > L3` 15/20 → 17/20, CPA 0.880 → 0.915 under `4d53fa2`);
- it is *anti*-correlated with the human construct it is named after, and the
  anti-correlation is structural (motion-scale normalisation), not noise;
- the natural-set failure is shared with the *old* estimator (P1.2's own scope
  caveat records the pre-direction repair at .395, already below chance), so it is
  not a regression introduced by the direction fix — the construct mismatch
  predates it.

Sentence for the paper: *"VBench Motion Smoothness on the natural preference set is
largely a frame-difference detector: an unlearned CPU baseline using only the mean
absolute frame difference reaches tie-aware accuracy 0.600 against the Official
metric's 0.636, and the Official score is a near-monotone function of that baseline
(Spearman −0.95). A flow-continuity repair satisfies the injected temporal-jerk
ladder within a clip (0.880 → 0.915 after the `4d53fa2` defaults) but is
significantly below chance on the same human pairs (0.325; paired Δ −0.312
[−0.361, −0.264]), because its scale-free acceleration ratio rewards
frame-to-frame change that the human label penalises. Motion Smoothness is therefore
an audit-only row: the counterfactual contract is satisfiable, the human contract is
not, and no replacement is claimed."*

In the decision table this belongs in the same "better counterfactual, worse
natural" cell as `dynamics_degree` v2 — with the additional, stronger diagnostic
that here the natural metric can be reproduced by a baseline that uses no model.

## 4. Q4 — minimal re-run list

The estimator changed at `4d53fa2`; the counterfactual row and P1.2 both describe
`feeb770`/`ccbd89b`. Nothing below needs weights downloaded; all helpers exist on
`h100-server`.

**Must be re-scored (GPU, ~13 min total)**

| # | scores | command | measured cost |
|---|---|---|---|
| 1 | counterfactual repair, 125 clips | archive `scores/motion_smoothness__repair*.jsonl` → `run_dim.sh motion_smoothness 1,2,3,4,5` | 73 s (18:34:06→18:35:19, 4 shards, GPUs 1,2,3,5) |
| 2 | natural repair, 1 440 videos | archive `natural-preference-runs/motion_smoothness/{repair_results.jsonl,predictions.csv}` → `run_natural.sh motion_smoothness repair 1,2,3,4,5` | 11 min 5 s (21:57:35→22:08:40, 5 shards) |

**Can be reused unchanged**

- Official counterfactual scores and their CPA (Official did not change; verified
  0.8300 from the frozen tree).
- Official natural scores (`official_scores/motion_smoothness/results.csv`) and all
  Official natural statistics (0.6364; the file is unchanged by a repair rerun).
- Human labels, split, pair census (`data/processed/pairwise_master_split.csv`).
- The frozen v1 archives under `scores/archive/`.

**Must be recomputed from the new repair scores (CPU, ~2 min)**

- Counterfactual: per-backend CPA, dev margin, rank-gap table, strict-order rate,
  Spearman, `D_mean`/`D_tail`, and the paired `base_id`-clustered interval
  (`scripts/counterfactual/run_dimension.py` then `summarize.py`).
- Natural: dev margin, tie-aware accuracy, Kendall tau-b, model-level Pearson, the
  marginal and paired intervals — **with `prompt_id` clustering** (§5).
- Then regenerate `motion_smoothness.md`, `CONSOLIDATED.md` Raw result row,
  `table2.csv`/`table2.json`/`SUMMARY.md`, `README.md` and the `P1.2` tables.

**Fix at the same time**

- Write the scoring revision into the score rows (`run_dimension.py:1287` currently
  records the render revision; the report's `code SHA` is `5c4a130` for `feeb770`
  scores).
- Add the prompt-clustered interval to `evaluate_pairwise_statistics.ci` and
  `evaluate_paired_backend_delta.compare`.
- Keep the lower-MAD baseline as a permanent control row (it is CPU-only and it is
  the finding that makes the natural result interpretable).

## 5. Q5 — audit of the paired intervals

### 5.1 Counterfactual interval — correct

`scripts/counterfactual/cpa.py:paired_bootstrap_ci` resamples **`base_id` clusters**
(`keys = sorted(set(official) & set(repair))`), draws `len(keys)` clusters with
replacement, and recomputes *both* backends on the same draw, so the pairing is
preserved; 2 000 iterations, seed 2026, percentile interval, each backend using its
own dev-calibrated margin. That matches plan 5.4 ("clustered by `base_id` for
VBench-CF"). Independently recomputed on the frozen tree: Δ **+0.0500**,
95 % CI **[−0.0150, +0.1100]**, 20 clusters — identical to the published value.
The only residual issue is the provenance defect of §0, not the algorithm.

### 5.2 Natural intervals — wrong cluster unit

`scripts/evaluate_paired_backend_delta.py:69-75` and
`scripts/evaluate_pairwise_statistics.py:94-99` both resample **pairs**
(`test_keys[rng.randrange(count)]`). The natural set has **43 test prompts × 30
pairs** (72 prompts × 30 = 2 160 pairs overall), and plan 5.4 requires the Natural
Set to be clustered by `prompt_id` and states that pairs within a model group must
never be resampled as independent observations. Consequence, recomputed on the same
frozen scores:

| interval | published (pair unit) | prompt-clustered | width inflation |
|---|---|---:|---:|
| Official tie-aware | [0.6101, 0.6636] | [0.5961, 0.6744] | 0.0535 → 0.0783 (+46 %) |
| Repair tie-aware | [0.2992, 0.3512] | [0.2907, 0.3628] | 0.0519 → 0.0721 (+39 %) |
| Paired Δ | [−0.3473, −0.2760] | [−0.3605, −0.2643] | 0.0713 → 0.0961 (+35 %) |

Every published interval is therefore too narrow by 35–46 %. All three still
exclude their null, so the *conclusions* survive; the *widths* do not, and the
`P1.2` numbers as printed overstate precision. Two smaller defects in the same
file: `model_level_pearson_n4` is computed over all valid pairs (dev + test) while
the accuracy columns are test-only, and the Kendall tau-b is reported without a
clustered interval or p-value. Seed handling is deterministic and reproducible
(seed 2026 for the marginal zero-margin and paired intervals, seed+1 = 2027 for the
tie-aware intervals), and the paired point estimate correctly lets each backend keep
its own dev-calibrated margin.

## 6. Findings from the previous review that still stand

- **The family is ordered, not mixed.** `temporal_jerk` has five distinct expected
  ranks and no rank-gap-0 pairs; the mixed-contract template was generated by a
  `len(ranks) > 1` gate. Fixed in `b2117a5`; the current report has no
  `Contract decomposition` and no `CPA by contract half` section.
- **The plan-required statistics now exist.** Sequence-level Spearman and
  strict-order rate, per-level `D_mean`/`D_tail`, and a paired interval are in the
  report (commits `dc3739c`, `feeb770`, `4d53fa2`). Section 3.1 of the previous
  review (no paired CI) is resolved for the counterfactual row.
- **The severity ladder is still an unvalidated design assumption.** Level 2
  rewrites `[f7…f11]` to `[f7,f7,f7,f11,f11]` (`transforms.py`), i.e. three frames
  excised plus holds — not the "skip two frames" the plan describes — and no
  `manual_validity_status` exists for the 25 bases. The L2/L3 comparison remains
  ambiguous between "metric defect" and "mis-declared level"; `4d53fa2` reduces the
  observable inversion without settling which it was.
- **The tie-aware column is degenerate for this family** (dev margin 0 for both
  backends), so the report's `zero-margin` and `tie-aware` rows are identical by
  construction. The split hygiene issue is fixed (the profile table is now
  split-aware).
- **Model/CUDA/weight parity against the frozen E0 baselines is still unverified**,
  and no weights were downloaded for this review.

## 7. Previously published conclusions that are overturned or downgraded

| previous statement | status now |
|---|---|
| "the v1 deficit is fixed; the repair matches Official on the declared ladder (parity)" | **Downgraded twice.** It is not parity — it is parity *on one statistic computed within a base* while being significantly below chance on natural preference. And the number describes `feeb770`, not the shipped defaults. |
| "estimator fixed (`907bc31`/`feeb770`), strict order 10/20 vs 7/20, Spearman 0.815 vs 0.765" | **Superseded by `4d53fa2`.** The replay predicts strict order **8/20** (worse than 10/20) with CPA 0.915; the pair of statistics disagree. |
| "§3.1 no paired CI — the −0.1050 is under-evidenced" | **Resolved** for the counterfactual row (paired CI exists and is correct). The natural-row intervals are still mis-clustered (§5.2). |
| "the deficit is partly structural; the estimator cannot realise `L2 > L3`" | **Partially resolved.** 15/20 → 17/20 on real clips (replay); still not 20/20. |
| "the report's code SHA does contain the generator (unlike the sibling reports)" | **Overturned.** The report's `code SHA` is the *render* revision (`5c4a130`), not the scoring revision (`feeb770`), and the score rows carry no revision at all. |
| `CONSOLIDATED.md` motion row and `P1.2` "describe the shipped estimator" with a revision caveat | **Insufficient.** Both are `feeb770`/`ccbd89b`; `4d53fa2` has never been scored, so neither describes the shipped estimator even with the caveat. |

## 8. Status of the numbers in this review

Everything below was recomputed on `h100-server` from the frozen trees; nothing was
re-scored and no frozen file was modified. The committed verifier
`scripts/counterfactual/verify_motion_review.py` reproduces every row of this table
in one run (the MAD row needs `--mad-cache` or `--compute-mad`):

```bash
cd /root/wenbiao_zhao/vbench-audit
/root/wenbiao_zhao/venvs/vbench/bin/python -B scripts/counterfactual/verify_motion_review.py \
    --output /root/wenbiao_zhao/tmp/mv/verify.json
```

| claim | verifier section |
|---|---|
| frozen scores are `feeb770` (replay reproduces all 125 to 1.665e-16) | `revision.replay_feeb770_vs_frozen` |
| no score row carries a revision; report `code SHA` is `5c4a130` | `revision.*_rows_carry_code_sha`, `revision.report_code_sha` |
| counterfactual CPA 0.8300 / 0.8800, paired +0.0500 [−0.015, +0.110] | `counterfactual.{official,repair}` |
| `4d53fa2` replay 0.9150, Δ +0.0850 [+0.015, +0.155], `L2>L3` 17/20, strict 8/20 | `revision.replay_4d53fa2_replay` |
| natural 0.6364 / 0.3248, margins 0.0215 / 0.1459, tau, pair-unit CIs | `natural.backends`, `natural.published_paired_pair_unit` |
| prompt-clustered intervals and the paired delta | `natural.backends.*_ci_prompt`, `natural.paired.delta_ci_prompt` |
| predictions.csv ↔ repair_results.jsonl, 0 mismatches on 1 440 videos | `natural.predictions_consistency` |
| Spearman(Official, Repair) = −0.1032, Pearson = +0.0198; ranges | `natural.per_video` |
| lower-MAD baseline 0.6000 [0.5674, 0.6295]; non-tie direction accuracies 0.7935 / 0.8075 / 0.4857; Spearman +0.2479 / −0.9451 | `mad_baseline` |
| variance decomposition 0.9224 / 0.5776; Spearman(speed_scale, D) = −0.3318 | `counterfactual.variance_*`, `normalisation` |
| timings 73 s / 11 min 5 s | `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/run_motion_smoothness_v2.log`; `natural-preference-runs/run_mot_repair.log` |

The MAD proxy (≤12 grayscale frames per video, 1 440 videos, CPU) is cached at
`/root/wenbiao_zhao/tmp/motion_natural_proxy.json`; `--compute-mad` rebuilds it and
needs no weights. One cosmetic defect in `finalize_natural.py:52` (`NameError` in
the closing `print`, after `results.csv` is written) makes a successful official
natural run end in a traceback; it does not affect the written file.

## 9. Executable checklist

1. **Do not quote the `motion_smoothness` row as a repair win** in
   `CONSOLIDATED.md`, `README.md` or the paper; mark it audit-only and cite the
   natural-set result next to the counterfactual one.
2. Fix score-row provenance: record the scoring revision in the score rows and
   split `code SHA` from the render revision in `run_dimension.py`.
3. Re-score counterfactual repair (`run_dim.sh motion_smoothness 1,2,3,4,5`, ~1 min)
   after archiving the `feeb770` shards; recompute CPA, strict order, Spearman,
   `D_mean`/`D_tail` and the paired CI.
4. Re-run natural repair (`run_natural.sh motion_smoothness repair 1,2,3,4,5`,
   ~11 min) after archiving `natural-preference-runs/motion_smoothness/`; recompute
   the dev margin, tie-aware accuracy, tau-b and model-level Pearson.
5. Switch both natural `ci()` and `compare()` to `prompt_id`-clustered resampling;
   regenerate the `P1.2` intervals and note the 35–46 % width correction.
6. Add the lower-MAD CPU baseline as a permanent control row in the natural tables.
7. Report CPA, strict-order rate and the `L2 − L3` sign count together — never CPA
   alone — and state the `L2`/`L3` level declaration as unvalidated.
8. Keep the direction-aware estimator only if the natural re-run is the acceptance
   test; if it stays below chance, the honest conclusion is that VBench Motion
   Smoothness needs a different target definition, not a better flow statistic.
