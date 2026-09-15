# Review: `dynamics_degree` counterfactual report (adversarial re-review)

**Scope.**

- Artifact under review: `docs/counterfactual-reports/dynamics_degree.md`,
  regenerated at code SHA `5c4a13091893273510794e45872eac3c59c3ad21` (an ancestor of
  the tree's current `HEAD`), plus its row in
  `docs/counterfactual-reports/CONSOLIDATED.md`
  (`0.8333` / `0.7722` / `−0.0611 [−0.150, +0.017]`, verdict `parity`) and the
  `dynamics_degree` note in `docs/counterfactual-reports/README.md`.
- Data reviewed: frozen score tree
  `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/` (40 bases, 160 clips,
  100% coverage both backends), its `archive/dynamics_degree__repair_v1_archived.jsonl`
  and the `v2/fixed05__shard*.jsonl` / `v2/fixed1__shard*.jsonl` re-measurements, the
  P1.3 holdout `validation_dynamic/` (30 bases, 120 clips) and the P1.1 natural run
  `/root/wenbiao_zhao/datasets/natural-preference-runs/dynamic_degree/`.
- Code reviewed: `scripts/counterfactual/{cpa,run_dimension,build_dynamic_validation,score_dynamic_validation}.py`,
  `scripts/evaluate_paired_backend_delta.py`, `metrics/dynamic-degree/`, and
  `P1_NATURAL_AND_CONTROL_RUNS.md` §P1.1 and §P1.3.
- Method: every statistic below was **recomputed on `h100-server` from the frozen
  trees**. The CPA margin, the tie-aware CPA and the paired cluster bootstrap were
  re-derived from their definitions in
  `scripts/counterfactual/verify_dynamics_review.py` **without importing
  `scripts.counterfactual`**, so a bug in `cpa.py` could not reproduce itself; the two
  implementations agree to the printed precision. No GPU was used, no weights were
  downloaded, no upstream checkout was modified, and nothing was written into the
  frozen trees. Written 2026-09-15.
- New material this round: the P1.1 natural-set run, the P1.3 independent holdout, the
  split-aware report at `5c4a1309`, and two attribution experiments written for this
  review (`scripts/counterfactual/verify_dynamics_review.py`,
  `scripts/counterfactual/natural_alpha_attribution.py`).

**Verdict.** The archived-row criticism stands and is now stronger, but three claims
must change. (1) **v2 does fix the contract at the level VBench scores** (the dataset
aggregate): the fitted exponent moves from `+0.458 [+0.334, +0.632]` (Official) and
`−0.511 [−0.646, −0.367]` (archived v1) to `−0.011 [−0.146, +0.133]` (shipped v2) on
the test split — the interval contains 0 and excludes both alternatives. (2) **Per-clip
invariance is not achieved and cannot be achieved by any exponent**: the exponent is a
per-rung multiplier, so it recentres the population and provably cannot compress its
spread. (3) The P1.3 holdout does not validate `0.5` as a *calibration* in the sense
§P1.3 claims: its three exponent rows are one distribution rescaled by exactly
`1, ½, ¼`, so the holdout contains a single independent measurement, and on that
measurement the ±20% differences between exponents are 6/30 vs 4/30 vs 5/30. Finally, a
new attribution experiment shows the natural-preference deficit (`−0.1155`) is **not**
caused by the invariance fix, which makes the audit-not-replacement reading stronger
rather than weaker. Three rendering/provenance defects are reported in §8.

## 1. What survives from the previous review, what is overturned

| previous claim | status | evidence |
|---|---|---|
| §1 the family is a pure invariance family; all 6 pairs per base expect a tie | **survives** | one `expected_rank` across the four levels; `family_pairs` expands 6 tie pairs per base |
| §2.1 `zero-margin` CPA needs exact float equality, so it is `0.0000` for any continuous score | **survives** | independently reproduced: both backends `0.0000` on dev and test |
| §2.2 the dev tie margin is the 0.9 quantile of dev deltas by construction | **survives** | `calibrate_margin` candidates `{0} ∪ q10..q90`, monotone for a tie-only family |
| §2.3 the CPA is sign-blind; a metric can score *higher* CPA while violating the contract | **survives, now demonstrated on shipped data** | v2 (`p=+0.019`) scores `0.7722` while the mirror-broken v1 (`p=−0.511`) scores `0.8444` |
| §3 "the report's own level table proves the repair fixed nothing" | **overturned for v2, kept for v1** | v1 `−0.511` is a genuine mirror; v2 `−0.011 [−0.146, +0.133]` is not (§3) |
| §3 the exact identity `repair == official / dt × const` | **downgraded to an empirical near-identity, and replaced by an exact one** | exact: `v1 == v2_fixed1` value for value, and `fixed05 == fixed1 × dt**0.5`; the "official / dt" constant held only as the ratio `0.015–0.016` across rungs |
| §2.4 / §4 "the dispersion improvement is consistent with a pure sign flip, so it is not evidence" | **overturned for the v1→v2 step, reframed** | CV `0.2360 → 0.1822` (test) is a real reduction, but the statistic still cannot certify per-clip invariance (§6) |
| §5 the clips are not the cause (rung re-encoding excluded; both estimators sublinear; `straightness` ≪ 1) | **survives, not re-run here** | unchanged; the flow measurements were taken as frozen in this review |
| §5 the two construction defects (coarse-rung span truncation, GIF duration drift) | **survive** | unchanged, still unfixed in the published dataset |

## 2. The report's own numbers, re-verified

Level profile and fitted exponent `score ~ dt**p` (target `p = 0`), recomputed from the
frozen score files; the 95% interval is a cluster bootstrap over `base_id` (1 000
resamples):

| backend | split | fitted `p` | 95% CI | `P(p<0)` | level profile (8→6→4→2) |
|---|---|---:|---|---:|---|
| Official | test | **+0.4581** | [+0.334, +0.632] | 0.000 | 1.000 / 1.196 / 1.437 / 1.911 |
| Official | dev | +0.6412 | [+0.514, +0.814] | 0.000 | 1.000 / 1.184 / 1.635 / 2.404 |
| Official | pooled | +0.4908 | [+0.371, +0.637] | 0.000 | 1.000 / 1.194 / 1.469 / 1.991 |
| v1 archived | test | **−0.5107** | [−0.646, −0.367] | 1.000 | 1.000 / 0.902 / 0.736 / 0.498 |
| v1 archived | pooled | −0.4813 | [−0.598, −0.360] | 1.000 | 1.000 / 0.898 / 0.750 / 0.516 |
| v2 shipped | test | **−0.0107** | [−0.146, +0.133] | 0.525 | 1.000 / 1.042 / 1.040 / 0.996 |
| v2 shipped | dev | +0.1241 | [−0.007, +0.279] | 0.037 | 1.000 / 1.017 / 1.143 / 1.172 |
| v2 shipped | pooled | +0.0187 | [−0.098, +0.140] | 0.373 | 1.000 / 1.037 / 1.061 / 1.032 |

The report's `+0.4908` / `+0.0187` and its per-level means reproduce exactly. So do the
CPA numbers: the independent implementation gives dev margins `20.1433` (Official) and
`0.03721` (repair), test tie-aware CPA `0.8333` / `0.7722`, paired delta `−0.0611` with
CI `[−0.150, +0.0167]`, and `[0, 0]` for the zero-margin paired statistic — identical to
`table2.csv` and to `cpa.py`, and stable across two bootstrap seeds.

Three identities were checked because they decide what the evidence can mean:

| identity | result |
|---|---|
| `v1_archived` vs `v2_fixed1`, 160 shared values | max relative difference **0.0** — the v2 plumbing changes nothing at exponent 1 |
| `v2_fixed05 / v2_fixed1` per clip at `fps8` | median **0.35355** = `0.125**0.5` exactly — the exponent is a pure per-rung rescaling |
| P1.3 holdout `ratio(α) = ratio(0) · 4**−α` | max absolute error **1.1e−15** over 30 bases × 2 exponents |

## 3. Q1 — Did v2 fix it? What must the previous conclusion become?

The previous review's §3 conclusion ("the report's own level table proves the repair did
not fix anything") was correct for the evidence available then, and is now **half
wrong**: it holds for the archived v1 and not for v2.

- **At the level the benchmark scores, yes.** VBench's dynamic degree is a dataset-level
  statistic (a proportion over videos), and at that level v2's fitted exponent is
  `−0.011 [−0.146, +0.133]` on the test split: consistent with exact invariance, and
  inconsistent with both the Official metric (`+0.458 [+0.334, +0.632]`) and the
  archived v1 (`−0.511 [−0.646, −0.367]`). The bootstrap distribution is centred at
  `−0.005` with `P(p<0) = 0.525`, i.e. no residual direction at all.
- **Per clip, no, and not by any exponent.** The per-base exponent spread is `sd = 0.392`
  on the test split (mean `−0.001`), and the per-base `fps2/fps8` ratio has median
  `1.163`, IQR `[0.731, 1.520]` with 6/30 bases inside ±20%. Since the score at rung `L`
  is the raw displacement times the constant `dt_L**−α`, changing `α` translates the
  whole ratio distribution on a log axis; it cannot narrow it. Any sentence of the form
  "v2 makes the metric frame-rate invariant" must carry "aggregate" and must not be read
  per video.
- **Two caveats the interval does not remove.** (a) `α = 0.5` is a default taken from a
  bootstrap over these same 40 bases
  (`exponent_source = diffusive_sqrt_lag_default_bootstrap_ci_0p39_0p65_not_independently_calibrated`),
  so the in-sample CI is not a held-out result — the holdout in §4 is the closest
  available substitute and is weaker than §P1.3 claims. (b) On the 10 dev bases alone the
  v2 interval is `[−0.007, +0.279]` with `P(p<0) = 0.037`, i.e. the fix is marginal on
  the split a calibration protocol would actually use; the dev split's own optimal
  exponent is `+0.624`.

**Corrected §3 sentence:** *the archived v1 repair mirrored the Official violation
(`−0.511` against `+0.491`) and fixed nothing; the shipped v2 repair removes it from the
reported aggregate (`−0.011 [−0.146, +0.133]`, test), while individual clips remain
dispersed and the exponent is a same-data default rather than an independent
calibration.*

## 4. Q2 — Does the P1.3 holdout validate `alpha = 0.5`?

**No, not in the sense §P1.3 claims, and the table cannot do so by construction.**

The holdout's three exponent rows are **one measurement rescaled**. Because
`score(α) = displacement / dt**α` and `dt` is constant inside a clip,
`ratio(α) = ratio(0) · 4**−α` exactly. Verified to `1.1e−15` on the holdout, and visible
in the published table itself:

| method | median | IQR | relative IQR (q75/q25) | within ±20% | `P(ratio<1)` |
|---|---:|---|---:|---:|---:|
| Official | 1.8474 | [1.253, 3.052] | 2.436 | 4/30 | 0.23 |
| α = 0 | 1.9352 | [1.273, 3.037] | **2.385** | 4/30 | 0.23 |
| α = 0.5 | 0.9676 | [0.637, 1.518] | **2.385** | 6/30 | 0.53 |
| α = 1 | 0.4838 | [0.318, 0.759] | **2.385** | 5/30 | 0.97 |

- Every α row is `ratio(0) × {1, ½, ¼}`: `1.9352 → 0.9676 → 0.4838`,
  `1.2733 → 0.6367 → 0.3183`, `3.0369 → 1.5185 → 0.7592`.
- The **relative** dispersion is therefore identical (`2.385`) for all three: the exponent
  cannot compress the spread, only move the centre. The ±20% column is a sliding window
  over the same distribution — 6 vs 4 vs 5 bases out of 30 — and is not evidence about α.
- Consequently the sentence "the ordering of the three exponents on unseen prompts is
  exactly the ordering the contract predicts, which is what the earlier in-sample CI
  could not establish" is **not supportable**: the ordering is algebraic. What the
  holdout contains is exactly one independent quantity, the per-base raw `fps2/fps8`
  ratio distribution (median `1.9352`, bootstrap CI `[1.370, 2.822]`).

What that single measurement does support, with intervals:

| comparison (paired, 30 bases) | improved | paired mean delta of abs log-ratio | 95% CI | exact sign test |
|---|---:|---:|---|---:|
| α=0 → α=0.5 | 20/30 | −0.214 | [−0.413, −0.002] | p = 0.099 |
| α=1 → α=0.5 | 20/30 | −0.317 | [−0.490, −0.129] | p = 0.099 |
| Official → α=0.5 | 19/30 | −0.210 | [−0.420, +0.012] | p = 0.201 |

plus the aggregate statement that `α = 0.5` is the only setting whose median ratio is
consistent with 1.0 (median `0.9676`, CI `[0.685, 1.411]`).

**Verdict for Q2:** the holdout supports *"α = 0.5 recentres the aggregate on unseen
prompts and is clearly better than the ballistic α = 1"* — and only weakly
(`p ≈ 0.10`) that it improves the typical per-base deviation over α = 0. It does **not**
support *"α = 0.5 is calibrated"*: a calibration claim needs a tolerable error the data
cannot meet (80% of bases are outside ±20%), and it cannot support any claim about
dispersion, because dispersion is α-invariant by construction. A holdout design that
*could* validate α would have to score a fresh sample of clips per setting, or use a
criterion that is not a monotone re-window of the same ratio distribution.

## 5. Q3 — The natural-set deficit (`−0.1155`) versus "v2 satisfies the contract"

The P1.1 numbers reproduce exactly when the paired script is re-run on the frozen natural
tree: Official `0.684496`, v2 `0.568992`, paired delta `−0.115504`, 95% CI
`[−0.149612, −0.080620]`, 1 290 test pairs, Official tie margin `0.0`, repair margin
`0.272133`.

**New attribution (written for this review).** The natural run stored per-transition
`residual_displacement` and `dt_seconds` for all 1 440 videos in
`structured_evidence.jsonl`, and the shipped score is exactly the duration-weighted mean
of `residual_displacement / dt**α`. Reconstructing that expression and re-running the
paired statistics gives — with the fidelity check that α = 0.5 reproduces the shipped
score **exactly**, max `|shipped − reconstructed| = 0.0` over all 1 440 videos:

| α | repair tie-aware acc | paired Δ vs Official | 95% CI |
|---:|---:|---:|---|
| 0 (no time normalisation) | 0.5667 | −0.1178 | [−0.1519, −0.0829] |
| 0.25 | 0.5667 | −0.1178 | [−0.1519, −0.0829] |
| **0.5 (shipped)** | **0.5690** | **−0.1155** | [−0.1496, −0.0806] |
| 0.75 | 0.5915 | −0.0930 | [−0.1271, −0.0589] |
| 1 (archived ballistic) | 0.5915 | −0.0930 | [−0.1264, −0.0581] |

**The deficit is not caused by the invariance fix.** It is present at every exponent, and
it is slightly *larger* at α = 0 (where the metric has no time normalisation at all) than
at the shipped α = 0.5. The total effect of the exponent on natural accuracy is ≤ 0.025,
against a gap of 0.093–0.118 to Official. The deficit therefore belongs to the audit
score *family* — a continuous residual-channel displacement with a dev-calibrated tie
margin — not to `dt**0.5`.

This strengthens the audit-not-replacement reading and corrects the reading that §P1.1
invites. Both statements hold simultaneously, and they are about different things:

- **contract (measurement property):** the reported aggregate must not depend on the
  sampling interval — v2 satisfies this (§3);
- **predictive validity (human preference):** the audit score is a worse predictor than
  Official's Boolean on unmodified generations, at every exponent — so the audit refines
  the official metric's *measurement* and does not replace it.

§P1.1's sentence "v2 does not improve, and slightly degrades, agreement with human
preference" is accurate as a description of the shipped rule, but placed next to the
invariance result it invites the false causal reading that the fix costs accuracy; the
attribution above rules that out. What remains unattributed is the rest of the bundle —
the residual-vs-apparent channel, the diagonal normalisation, the continuous-vs-Boolean
score and the dev-calibrated margin — because P1.1 varies all of them at once (checklist
item 6).

## 6. Q4 — The split-aware `Invariance statistics`, recomputed

| backend | split | mean within-base CV | mean relative range | per-base `fps2/fps8` median | IQR | within ±20% |
|---|---|---:|---:|---:|---|---:|
| Official | dev | 0.3838 | 0.9862 | 2.748 | [2.264, 3.309] | 0/10 |
| Official | test | 0.3177 | 0.8196 | 2.189 | [1.487, 3.043] | 2/30 |
| v1 archived | dev | 0.1800 | 0.4662 | 0.669 | [0.585, 0.779] | 1/10 |
| v1 archived | test | 0.2360 | 0.6107 | 0.582 | [0.365, 0.760] | 6/30 |
| v2 shipped | dev | 0.1749 | 0.4375 | 1.338 | [1.169, 1.558] | 2/10 |
| v2 shipped | test | 0.1822 | 0.4760 | 1.163 | [0.731, 1.520] | **6/30** |

The report's four published cells reproduce exactly (it prints dev/test for both
backends; the CV columns match to four decimals).

**My §2.4 conclusion must be split in two.**

- *Overturned for the v1 → v2 step:* the CV improvement `0.2360 → 0.1822` is a real
  reduction, not a sign flip, and it agrees with the exponent. Had this statistic been
  available then, it would have counted in v2's favour.
- *Downgraded, not overturned, as a criterion:* the statistic still cannot certify
  per-clip invariance, and the reason is now exact rather than rhetorical — (i) it is
  unsigned; (ii) the α-family cannot change the *relative* dispersion of the per-base
  ratio (identical `2.385` on the holdout, and `2.079` test in-sample for v1 and v2
  alike), so the CV gain comes from re-centring; (iii) the in-sample ±20% fraction is
  **identical** for v1 and v2 on the test split (6/30 both), i.e. the column that looks
  most like a per-clip measure does not move at all; (iv) a constant score would minimise
  CV while measuring nothing, which the report's own `Score sensitivity` section already
  warns about.

**Corrected wording for the report's `Invariance statistics` paragraph:** the dispersion
block is a *necessary companion* to the signed exponent, not "the meaningful invariance
measure" — it corroborates v2's aggregate fix, and it cannot by itself distinguish
invariance from recentring or from insensitivity.

## 7. Q5 — Is "30 of the 32 unused candidates" the fairest holdout?

The arithmetic is right and the claim is weaker than it sounds.

- The pool has **72 candidates with 72 distinct prompts and 72 distinct video UIDs**
  (verified by re-running the selector's pool construction). Therefore **any** subset is
  prompt-disjoint from its complement — prompt disjointness is a property of the pool, not
  a constraint that forces a 30-base holdout. The binding constraint is the pre-existing
  choice of the 40 in-use bases by the selector's own hash ranking, which leaves a
  leftover of 32.
- That leftover is **generator-skewed**: `{cogvideo 8, lavie 11, modelscope 9,
  videocraft 4}`, i.e. videocraft 12.5% against 27.5% (11/40) in the in-use set. The
  holdout's official `fps8` median (`15.72`) is above the in-use median (`11.12`), though
  a permutation test on the medians does not separate them at n = 30 vs 40 (`p = 0.26`),
  so the sets are not demonstrably different in score level; the compositional difference
  is real regardless.
- Taking 30 instead of 32 saves 8 clips and buys nothing; the "largest honest holdout"
  phrasing suggests a limit that is not binding.

**Fairer designs, in increasing order of cost:**

1. Use all 32 unused candidates and report the generator composition next to the result
   (trivial, no re-scoring of the counterfactual family).
2. Re-split the 72 **at random (or stratified by generator) into 40 / 32**, which is
   prompt-disjoint by construction and balances the arms; re-score only the new 40 on the
   counterfactual ladder to check the exponent. This changes a fixture choice in a
   published dataset, so it belongs to a dataset revision, not to this round.
3. **Cross-fit the exponent**: choose α on a random half of the bases and evaluate on the
   other half, over the same 72. This converts "default from the same 40 bases" into a
   cross-validated estimate and costs one CPU pass over the existing displacements.
4. Report the holdout's source-fps mix (`22× 8 fps + 8× 10 fps` vs `30 + 10` in-use) for
   the same reason.

None of this changes the direction of the P1.3 result; it changes how strongly the result
can be worded.

## 8. Defects to fix before this row is quoted

1. **The paired interval printed in the report belongs to the wrong statistic.**
   `dynamics_degree.md` §"Official vs Repair (test, tie-aware)" prints the tie-aware delta
   `−0.0611` with `[+0.0000, +0.0000]`, which is the *zero-margin* paired interval:
   `run_dimension.py` reads `cpa["paired"]["zero_margin"]["ci_low"/"ci_high"]` while the
   row's delta comes from `test_tie_aware`. The correct interval is `[−0.150, +0.0167]`
   (independently reproduced, and already correct in `table2.csv`). The `parity` verdict
   survives — the correct interval crosses zero — but the report shows an interval that
   does not describe the delta beside it.
2. **The `Frame evidence` section is stale.** It says "No per-clip evidence was recorded
   for this run … Re-score with the current `score.py`", but the shipped tree
   `scores/dynamics_degree__repair.jsonl` carries `lag_scaling` (`lags_frames`,
   `chord_displacement_diagonals`, `path_displacement_diagonals`, `straightness`,
   `fit_rmse_log_log`), `coverage`, `threshold`, `exponent` and `channel_means` for all
   160 clips. The generator's evidence lookup does not read that layout.
3. **The frozen tree's `mode` label contradicts its `exponent`.** All 160 shipped rows
   record `mode = "measured"` while `exponent = 0.5` with
   `exponent_source = diffusive_sqrt_lag_default_bootstrap_ci_0p39_0p65_not_independently_calibrated`
   — the applied normalisation was the *fixed* default, not a per-clip fit. The scoring
   driver recorded its raw CLI argument. `exponent_source` is correct and authoritative;
   the row should record the resolved mode, or CONSOLIDATED should state explicitly that
   `mode` is the driver argument.
4. Minor: the report's `levels` line is unordered (`fps6, fps4, fps2, fps8`).

## 9. What this dimension can now claim

Ranked, each with the interval it needs:

1. **Official dynamic degree is frame-rate dependent.** Fitted exponent
   `+0.458 [+0.334, +0.632]` (test) and `+0.491 [+0.371, +0.637]` (pooled): the same
   trajectory scores ~2× higher at 2 fps than at 8 fps, and the fixed-pixel `check_move`
   threshold makes the static/moving decision easier to cross at low frame rates.
   Independent of any repair, and independently reproduced here.
2. **The shipped v2 repair removes that dependence from the reported aggregate.**
   `−0.011 [−0.146, +0.133]` (test), against the archived v1's mirror
   `−0.511 [−0.646, −0.367]`.
3. **The composite CPA must not be quoted for this family**, and v2 is the proof: the
   variant that satisfies the contract scores *lower* CPA (`0.7722`) than the variant that
   mirrors the violation (`0.8444`).
4. **Per-clip invariance is not claimed.** Per-base exponent `sd = 0.392`; per-base
   `fps2/fps8` median `1.163`, IQR `[0.731, 1.520]`, 6/30 inside ±20%. No exponent can
   change this, because the exponent is a per-rung constant.
5. **On natural preference the audit family is worse than Official at every exponent**
   (`−0.093 … −0.118`), so this is a measurement audit, not a replacement candidate.

## 10. Executable checklist

1. **Fix the paired-interval rendering.** In `scripts/counterfactual/run_dimension.py`,
   take `ci_low`/`ci_high` from `cpa["paired"]["tie_aware"]` for the tie-aware row (or
   print both intervals when both deltas are shown), then regenerate `dynamics_degree.md`
   and re-check `table2.csv` / `SUMMARY.md`.
2. **Repair the `Frame evidence` section.** Either read the evidence the shipped tree
   already carries (`lag_scaling`, `coverage`, `threshold`), or state that the section
   applies only to runs scored by a driver that did not persist it.
3. **Fix the provenance label.** Record the resolved exponent mode in the frozen rows (or
   document `mode` as the driver argument) so `mode = "measured"` cannot be read as "the
   shipped scores used per-clip exponents".
4. **Rewrite §P1.3's claim.** Replace "the ordering of the three exponents … is what the
   earlier in-sample CI could not establish" with: the holdout measures the per-base raw
   `fps2/fps8` ratio distribution (median `1.9352`, CI `[1.370, 2.822]`); `α = 0.5`
   recentres it (`0.9676`, CI `[0.685, 1.411]`) and is clearly better than `α = 1`; the
   relative dispersion (`2.385`) is α-invariant, so no dispersion claim can be made, and
   the ±20% counts (4/6/5 of 30) are not evidence.
5. **Restate the family's claim in `CONSOLIDATED.md`** as "aggregate-level fix, per-clip
   dispersion unresolved" (the note already says this; add the interval
   `−0.011 [−0.146, +0.133]`) and keep the v1 mirror beside it.
6. **Add the natural-set attribution table to §P1.1** (`α = 0, 0.25, 0.5, 0.75, 1` →
   `−0.118, −0.118, −0.116, −0.093, −0.093`) so the deficit is not read as the price of
   invariance; then attribute the residual by varying the channel (residual vs apparent),
   the diagonal normalisation and the score margin at fixed `α = 0.5` — the natural
   evidence is already on disk, so this is CPU-only.
7. **Adopt a fairer calibration split** (item 2 or 3 of §7) and use all 32 unused
   candidates before the holdout sentence is quoted as "the largest honest holdout".
8. **Fix the ordering of the `levels` line** in the generator.

## 11. Residual uncertainty in this review

- No GPU run was performed: the per-transition flow measurements are taken as frozen, so
  the exponent intervals are bootstrap uncertainty over the existing 40 (and 30 holdout)
  bases, not over new clips. A re-measurement would add flow-level variance these
  intervals do not contain.
- The independent CPA re-implementation shares the *definitions* with `cpa.py` (tie-margin
  CPA, 0.9-quantile margin search, cluster bootstrap); it is the code path that is
  independent, and it reproduced every published number.
- The natural-set attribution varies `α` within the v2 bundle only; the channel,
  normalisation and margin contributions remain unmeasured (§5).
- The `0.5` exponent has no mechanism behind it in this repository: `~sqrt(dt)` scaling is
  consistent with the data (both RAFT and Farnebäck are sublinear, and `straightness` ≪ 1
  says the trajectories turn), but no experiment here separates that from estimator
  behaviour at large displacements.
- The two construction defects (coarse-rung span truncation; GIF container-duration drift)
  are unchanged, and the published dataset remains as scored.
