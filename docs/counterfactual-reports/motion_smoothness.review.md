# Review: `motion_smoothness` counterfactual report

Scope: `docs/counterfactual-reports/motion_smoothness.md` (code SHA
`bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`), the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `run_dimension.py`), the `temporal_jerk`
transform (`scripts/counterfactual/transforms.py:472-556`) and the Audit
estimator it scores against
(`metrics/motion-smoothness/src/motion_smoothness/metric.py:analyze_motion_fields`).

Verdict: the headline row `motion_smoothness | temporal_jerk | 0.8300 | 0.7250 |
-0.1050` in `docs/counterfactual-reports/README.md` is a **real deficit on the
declared ladder, but the report does not establish it and does not describe what
it is measuring.** Three defects, in descending order of severity:

1. The report's framing is wrong: `temporal_jerk` is a *pure ordered* family, not
   a "mixture of two contracts", and it has **no rank-gap-0 pairs**. The
   "Contract decomposition" narrative and the rank-gap-0 sentence are
   boilerplate emitted for any family with more than one rank; they contradict
   the report's own table.
2. The deficit is at least partly **structural in the Audit estimator**: on
   canonical smooth trajectories the estimator ties or inverts the declared
   ordering of `jerk_2_duplicate_skip` vs `jerk_3_local_reverse`, which the
   contract requires to be strict. The report's own level table already shows
   the repair placing level 2 *below* levels 3 and 4.
3. The plan-required statistics are missing — a sequence-level order statistic,
   mean/tail discontinuity, and the **paired** `Repair - Official` confidence
   interval — so the `-0.1050` is quoted at a precision the evidence does not
   carry (the two marginal 95% CIs overlap on `[0.77, 0.79]`).

Nothing below requires a model re-run. Section 2 uses the repository's own
transform and estimator with synthetic constant-flow trajectories; sections 1,
3 and 4 are derived from the published report and the committed CPA code.

## 1. This family is ordered, not mixed — the report contradicts itself

`temporal_jerk` emits five levels with five **distinct** expected ranks
(`transforms.py:489,501,516,534,551`):

| level | `expected_rank` | contract |
|---|---:|---|
| `jerk_0_original` | 4 | highest smoothness |
| `jerk_1_duplicate` | 3 | " |
| `jerk_2_duplicate_skip` | 2 | " |
| `jerk_3_local_reverse` | 1 | " |
| `jerk_4_multiple` | 0 | lowest smoothness |

`C(5,2) = 10` pairs per base; 20 test bases -> 200 test pairs, 5 dev bases ->
50 dev pairs. The rank-gap census of the report is exactly what that implies:
gaps 1/2/3/4 have 4/3/2/1 pairs per base -> 80/60/40/20 test pairs. **Every pair
has `expected ±1`; there are no ties and no rank-gap-0 pairs.**

The generator gates the mixed-contract text on `len(ranks) > 1`
(`run_dimension.py:470` at `bdfda5c`), which is true for *any* family with at
least two levels, including pure ordered ones:

```text
run_dimension.py:475  "This family declares more than one expected rank, so its CPA is a"
run_dimension.py:476  "mixture of two contracts and is dominated by whichever is easier..."
run_dimension.py:477  "gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of"
run_dimension.py:496  "Rank-gap-0 pairs are the family's actual target. ..."
```

For this family lines 475-480 and 496-498 are false. The correct predicate is
already computed four lines later as `dispersion_rows` (`run_dimension.py:501-503`:
ranks shared by more than one level). A family is a contract mixture iff
`dispersion_rows` is non-empty; `subject_consistency` qualifies,
`motion_smoothness` does not.

Two further boilerplate artefacts in the same report:

- "CPA by contract half" (report lines 22-31) says both halves "are scored
  separately", then prints only the `sensitivity` rows for both backends. The
  invariance half has `n_pairs = 0` and is silently skipped
  (`run_dimension.py:412` at `bdfda5c`). The prose describes a table that is not there.
- The `Score sensitivity` note about "a single distinct value ... its CPA on an
  invariance family is vacuous (plan section 7.4)" (report lines 81-83) does not
  apply: every level has `distinct = 25`.

None of this is a scoring bug; it is report-generator copy that a reader will
mistake for the family's design. It must be fixed before the report is cited.

## 2. The estimator cannot realise the declared level-2 > level-3 ordering

The contract is a strict monotone order
`L0 > L1 > L2 > L3 > L4`. The Audit estimator's response is a second difference
of a *normalised* first difference: `relative = |v_t - v_{t-1}| /
(|v_t| + |v_{t-1}| + eps)` saturates near 1 for any large change, and `D_t` is
then `|relative_t - relative_{t-1}|`, aggregated as
`0.75 * mean(D_t) + 0.25 * mean(top 10% of D_t)` (`metric.py:303-354`).

`output/counterfactual/_scratch/ms_level_response.py` feeds the repository's own
`transforms.temporal_jerk` index surgery and `analyze_motion_fields` the exact
per-step flows of a canonical smooth trajectory (a translating pattern; the flow
between two reordered source frames is the known displacement). No RAFT, no
CUDA, no content noise. Sweep over clip length, speed profile and step size:

| n | profile | step px | L0 | L1 | L2 | L3 | L4 | strict order | L2-L3 |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 16 | constant | 0.5 | 1.0000 | 0.7235 | 0.6184 | 0.6183 | 0.4797 | yes | +0.0000 |
| 16 | constant | 1.0 | 1.0000 | 0.7235 | 0.6183 | 0.6183 | 0.4797 | yes | +0.0000 |
| 16 | constant | 2.0 | 1.0000 | 0.7235 | 0.6183 | 0.6183 | 0.4797 | yes | +0.0000 |
| 16 | ramp | 1.0 | 0.9809 | 0.7223 | 0.6174 | 0.6313 | 0.5112 | **no** | **-0.0139** |
| 16 | sinusoid | 1.0 | 0.8435 | 0.6873 | 0.6089 | 0.6238 | 0.5131 | **no** | **-0.0149** |
| 33 | constant | 1.0 | 1.0000 | 0.8052 | 0.7047 | 0.7179 | 0.5770 | **no** | **-0.0132** |
| 33 | ramp | 1.0 | 0.9949 | 0.8080 | 0.7054 | 0.7208 | 0.5851 | **no** | **-0.0154** |
| 33 | sinusoid | 1.0 | 0.9674 | 0.8219 | 0.7109 | 0.7507 | 0.5870 | **no** | **-0.0399** |

(All 18 sweeps in the script; the pattern is identical at 0.5/2.0 px and for
both frame lengths, and L2 vs L3 is the *only* violated adjacent pair in every
row.)

- At constant velocity, n=16, the two levels score `0.6183308` and `0.6183150`:
  a tie of `1.6e-5`, not the strict `L2 > L3` the family declares.
- At every other setting the ordering **inverts** by `0.013`-`0.040` in favour
  of level 3.

The mechanism is visible in `D_t` for n=16, constant, 1 px:
`jerk_2` gives `[.., 1.00, 1.00, 1.00, .., 1.00, ..]` and `jerk_3` gives
`[.., 0.60, 0.40, 1.00, .., 1.00, 0.40, 0.60, ..]` — the same total
discontinuity mass and the same top-tail, hence the same `D_video = 0.4808`.
A five-frame excise-and-hold and a four-frame in-place reversal are operationally
different edits of similar temporal extent; the mean+tail aggregation of a
saturated second difference cannot tell them apart.

This is not a synthetic-only artefact. The report's own per-level means
(lines 92-96) are:

| backend | L0 | L1 | L2 | L3 | L4 | order |
|---|---:|---:|---:|---:|---:|---|
| official | 0.9565 | 0.9559 | 0.9545 | 0.9482 | 0.9378 | 0>1>2>3>4 |
| repair | 0.7562 | 0.7292 | **0.6586** | 0.7094 | 0.6732 | 0>1>3>4>**2** |

The repair's mean ordering puts level 2 *last*, below both level 3 and level 4.
If the per-base ordering followed the means, the repair would lose 2 of the 10
pairs per base — `(L2,L3)` and `(L2,L4)` — before any noise, capping the CPA at
`0.80`. The observed `0.7250` is consistent with that plus ordinary noise; the
official means violate nothing.

The honest reading is therefore not "the repair is worse at smoothness". It is
"the repair ranks the two middle levels in the opposite order to the one the
family declares, and the family's declared order is an unvalidated design
assumption" — see section 4.

## 3. The headline delta is not established, and the required statistics are missing

### 3.1 No paired confidence interval

Plan section 5.4 requires a 95% cluster bootstrap for `Ours - Official`. The
report's final table (lines 98-102) prints only the point estimate
`-0.1050`. The two marginal intervals the report *does* give overlap:

```text
official test [0.7700, 0.8900]   repair test [0.6500, 0.7900]   overlap [0.77, 0.79]
```

`bootstrap_ci` (`cpa.py:135-149`) resamples each backend independently, so no
paired interval exists anywhere in the harness, and `run_dimension.py:562-575`
computes the delta by subtracting two rounded point estimates. A `-0.105` on
20 clusters with an overlap in the marginals cannot be called a "genuine Repair
loss" (README) until the paired interval is reported.

### 3.2 Missing sequence-level and discontinuity statistics

Plan 5.2 ("for ordered levels ... also report a sequence-level statistic such as
Spearman correlation or strict-order rate") and plan 13.3 ("Report monotonic CPA,
Spearman correlation, mean discontinuity, and tail discontinuity") are both
explicit. The report contains:

- no Spearman correlation and no strict-order rate anywhere (`grep` over
  `scripts/counterfactual/` finds neither term);
- only the per-level *score* distribution, not the repair's own
  `mean_discontinuity` / `tail_discontinuity` diagnostics, which
  `analyze_motion_fields` already computes (`metric.py:363-364`) and which are
  exactly the two numbers plan 13.3 asks for.

For an ordered family the strict-order rate is the statistic that answers the
question the CPA is being used to answer, and it is the one the report omits.

### 3.3 Split hygiene and a redundant column

- "CPA by contract half" reports `250` pairs and labels the column "dev margin",
  but never says those 250 pairs are **50 dev + 200 test pooled**
  (`run_dimension.py:683` calls `contract_split_cpa(rows, ...)` with all rows).
  The pool mixes the split used to calibrate the margin into the reported
  number. The rank-gap table, by contrast, is test-only. Pick one split and say
  which.
- Both `tie-aware` rows equal the `zero-margin` rows exactly, because the dev
  margin is `0`. That is expected for a family where every pair expects `±1`:
  `calibrate_margin` (`cpa.py:121-132`) searches margins and widening one turns
  correct sign predictions into ties, so the first candidate (`0`) wins. The
  report should state that the tie-aware column is degenerate here rather than
  print two identical columns as if they were checks on each other.

## 4. Why the declared ladder itself needs justification

Even with a perfect estimator, the CPA would reward whichever backend agrees
with the declared order. Here the order is an assumption, not ground truth:

- Plan 13.3 declares level 2 ("duplicate and skip two frames") milder than
  level 3 ("reverse a short local segment"). The code does not implement "two
  duplicated and two skipped frames" literally: for a 16-frame clip it rewrites
  `[f7,f8,f9,f10,f11]` to `[f7,f7,f7,f11,f11]` (`transforms.py:506-510`), i.e.
  three frames are excised and replaced by two holds — a stop-and-go that is
  arguably *harsher* than reversing four frames in place, which is what level 3
  does. Whether `L2 < L3` in smoothness is an empirical question the family
  never answers.
- The L2-vs-L3 inversion of section 2 is therefore ambiguous by construction:
  it is consistent with a metric defect *and* with the repair correctly
  disagreeing with a mis-declared level.
- Plan section 14 requires human validation of the constructed relation, and
  `output/counterfactual/bases.jsonl` carries no `manual_validity_status` for
  any of the 25 bases; the ladder has never been checked against annotators.

The report does not mention this, and the README's "genuine Repair loss" reads
the composite as if the ladder were ground truth.

## 5. What the report should say instead

1. **Fix the generator gate first.** Emit the "contract mixture / rank gap 0"
   block iff `dispersion_rows` is non-empty (`run_dimension.py:501-503`), and
   label the ordered family's table "rank-gap decomposition" only. The current
   text is indistinguishable from a design description and is wrong here.
2. **Lead with the sequence-level statistic.** Per base: the strict-order rate
   over `L0 > L1 > L2 > L3 > L4`, the Spearman correlation, and the signed
   `L2 - L3` contrast. Report the composite CPA as a secondary number.
3. **Report the repair's `D_mean` and `D_tail` per level** (plan 13.3), which
   also makes the level-2/level-3 behaviour legible.
4. **Add the paired cluster-bootstrap `Repair - Official` CI** (plan 5.4) beside
   `-0.1050`, and stop quoting the delta alone.
5. **Either justify or fix level 2.** If the intent is "duplicate and skip two
   frames", implement a two-frame edit (not a three-frame excision); if the
   intent is "a hold-and-jump", declare it as such and pre-register the expected
   order against a validated reference. Until then, report the `L2`/`L3`
   declaration as unverified.
6. **State the split** for the contract-half table, and drop or footnote the
   redundant tie-aware column.

The honest headline sentence is: *on the declared jerk ladder the Official
interpolation score separates the five levels in the declared order (test CPA
0.8300, 95% CI [0.77, 0.89]), while the Audit repair inverts the order of the
two middle levels and scores 0.7250 [0.65, 0.79]; the paired difference is not
yet reported, the estimator has a demonstrated level-2/level-3 degeneracy on
clean motion, and the ladder's own severity ordering has not been validated.*

## 6. Status of the numbers in this review

- Sections 1, 3 and 4 are derived from the published report and the committed
  code at `bdfda5c`. Unlike the two sibling reports (`dynamics_degree`,
  `subject_consistency`, both naming `66c4a99`), this report's `code SHA`
  *does* contain the generator that produced it (`cpa.py`, `contract_split_cpa`,
  `rank_gap_groups` are all present at `bdfda5c`); only the report file itself
  was committed later (`a044ac9`). The reproducibility complaint from the
  sibling reviews does not apply here.
- Section 2 uses the repository's own `transforms.temporal_jerk` and
  `analyze_motion_fields` on synthetic constant-flow trajectories
  (`output/counterfactual/_scratch/ms_level_response.py`), so it isolates the
  estimator's response to the transform. It does **not** re-score the 125 real
  clips: those scores and the derived clips live on the scoring host
  (`/root/wenbiao_zhao/datasets/counterfactual-vbench`), which is not reachable
  from this workspace, and this workspace has no RAFT weight or CUDA device.
- The claim that the repair's *per-base* ordering inverts `L2`/`L3` is inferred
  from the published per-level means plus the probe. A scoring host can confirm
  it directly from the cached `<dimension>__official.jsonl` /
  `__repair.jsonl` by computing the strict-order rate, the per-base `L2 - L3`
  sign counts, and the paired bootstrap; `run_dimension.py --report-only` will
  rebuild the report from those caches once the two missing blocks are added.
- The probe scripts are gitignored under `output/counterfactual/_scratch/`
  (consistent with the sibling reviews); `ms_level_response.py` reruns in
  seconds with `.venv/bin/python` and no model assets.

## 7. Addendum (2026-09-15): all three defects fixed and the dimension re-scored

The findings above describe the shipped estimator and report at `a044ac9`. Both
have since been repaired, and the dimension was re-scored at `feeb770`:

- **Estimator (issue 2, commit `907bc31`).** `D_t` now follows the plan 13.5 Full
  variant, `magnitude_weight * flow acceleration + direction_weight * direction
  change`, aggregated with mean + upper tail; the previously computed but unused
  `direction_change_t` is now load-bearing. `magnitude_weight`/`direction_weight`
  are configurable (`0.7`/`0.3`) and threaded through the CLI and the sharded
  worker. `metrics/motion-smoothness/tests/test_temporal_jerk_ladder.py` pins the
  strict `L0 > L1 > L2 > L3 > L4` order on canonical constant/ramp/sinusoid
  trajectories; the `L2 - L3` margin moves from `[−0.040, 0.000]` to
  `[+0.040, +0.122]` across the 18-setting sweep. The claim in section 2 that the
  estimator "cannot realise the declared ordering" therefore describes v1 only.
- **Report generator (issue 1, commit `b2117a5`).** The mixed-contract template
  is now emitted only when a family has more than one rank *and* a rank shared by
  several levels. `temporal_jerk` gets no `Contract decomposition` and no
  `CPA by contract half` section; the ordered-family regression is pinned in
  `tests/test_counterfactual_cpa.py`.
- **Missing statistics (issue 3, commits `dc3739c`, `feeb770`).** The report now
  carries per-base Spearman and strict-order rate (plan 5.2/13.3), the repair's
  per-level `D_mean`/`D_tail` (plan 13.3), and a **paired** cluster-bootstrap
  `Repair − Official` interval (plan 5.4). The v1 paired interval was
  `[−0.200, −0.020]`: the −0.1050 deficit the review could only call
  "under-evidenced" was in fact significant.
- **Re-scored numbers (`feeb770`, 125/125 clips both backends).** Official
  0.8300 `[0.770, 0.890]`; Repair 0.8800 `[0.805, 0.945]`; delta +0.0500 with
  paired 95% CI `[−0.015, +0.110]` (crosses zero). Sequence-level: strict order
  10/20 (Repair) vs 7/20 (Official), mean Spearman 0.815 vs 0.765. The repair's
  per-level means are now monotone (`0.5407 > 0.5011 > 0.4855 > 0.4760 > 0.4268`),
  and its `D_mean`/`D_tail` rise with severity (`0.5646 → 0.7515` and
  `0.8433 → 1.1627`). The v1 repair scores are archived at
  `scores/archive/motion_smoothness__repair_v1_archived.jsonl`.
- **What the result now supports.** The v1 deficit is gone; the honest reading is
  *parity*, not a win — the paired interval crosses zero and the family's severity
  ladder (in particular the `L2 > L3` declaration, section 4) still has no human
  validation. `CONSOLIDATED.md` and `table2.csv` carry these numbers and the
  paired interval.
