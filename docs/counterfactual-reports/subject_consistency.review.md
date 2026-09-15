# Review: `subject_consistency` counterfactual report (adversarial re-review)

**Scope.**

- Artifact under review: `docs/counterfactual-reports/subject_consistency.md`,
  regenerated at code SHA `5c4a13091893273510794e45872eac3c59c3ad21`, and its row
  in `docs/counterfactual-reports/CONSOLIDATED.md`
  (`0.5917` / `0.8500` / `+0.2583 [+0.158, +0.350]`) and `README.md`.
- Data reviewed: frozen score tree
  `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/` (25 bases, 100 clips,
  100% coverage both backends), the P1.5 confirmatory arms
  `subject_confirmatory/` (equal-area, 23 bases × 4 levels × 2 backends) and
  `subject_tracked/` (tracked-box control, same 23 bases), and
  `P1_NATURAL_AND_CONTROL_RUNS.md` §P1.5.
- Code reviewed: `scripts/counterfactual/{cpa,run_dimension,summarize,transforms,score}.py`,
  `subject_position_profile.py`, `metrics/subject-consistency/`.
- Method: every statistic below was **recomputed independently on `h100-server`**
  with a fresh implementation that does not import `scripts.counterfactual`, so a
  bug in `cpa.py` could not reproduce itself. Commands and outputs are quoted
  inline. This review is written on 2026-09-15 against
  `local = origin/winbeau = h100-server = 52e4b76` plus the review commit.
- Prior version of this file reviewed the same artifact at `66c4a99` and is
  superseded; §1 states exactly which of its findings survive.

**Verdict.** The published numbers are arithmetically correct — every figure in
`subject_consistency.md` reproduced exactly on independent recomputation. What
does **not** survive is the interpretation. Three claims must change:

1. The **`+0.2583` composite is not a Repair win**. It is the exact average of a
   sensitivity half where Repair is *worse* (−0.1500) and an invariance half
   where the comparison is **margin-dependent and inverts** (§3.3). It
   should be marked *declining*, alongside `scene` and `spatial_relationship`.
2. The **`0.2500 → 0.9167` headline is not established** and is not comparable
   with the confirmatory run: it uses a different base set (9/25 overlap) and a
   different construction, and the equal-area arms give a *different* ordering
   under a different margin (Official 0.9825 vs Repair 0.9474, §3.3).
3. What **is** established is the **margin-free position dispersion**: on
   identical corruption with identical boxes, Official's relative range is
   **4.7× the Repair's** (0.1017 vs 0.0215), and Official is wider on **21/23**
   bases, exact sign test **p = 6.6 × 10⁻⁵** (§4.4). That is the claim the paper
   can carry.

Two concrete defects in the report itself must be fixed before it is quoted
(§6): the paired interval printed for the headline belongs to a *different*
statistic, and the per-half paired interval is computed over a *different* base
set than its table claims.

---

## 1. What survives from the previous review, and what is superseded

In the table below, `§N` citations in the *claim* column refer to the **previous
version of this file** (the `66c4a99` review, in git history), not to this
document. `§N` citations in the *why* column refer to this document.

| previous claim | status | why |
|---|---|---|
| Pooled CPA is half sensitivity, half invariance; the composite must never be quoted alone | **survives, strengthened** | verified arithmetically and re-derived from frozen scores (§3.1) |
| Official is position-dependent; the position spread is real | **survives** | holds on both base sets and both constructions, margin-free (§4.4) |
| The repair reduces but does not remove position sensitivity | **survives** | equal-area: 0.1017 → 0.0215 is a reduction, not zero |
| §4.2 "tracked boxes make the three placements differ in area *and* position, so the gap is confounded" | **superseded** | P1.5 equal-area removes the confound by construction and I verified it: **23/23** bases now use byte-identical window boxes at all three positions, constant across frames; the tracked control has median max/min area **1.33**, max **5.08** (§4.1). The confinement was real and is now fixed. |
| §3 "repair penalises the corruption least at the start / `middle < end < start`" | **superseded / withdrawn** | those level means came from the tracked construction on the old base set. On the equal-area arms the per-position picture is different and the claim is not re-established; it must not be repeated. |
| §4.1 "the `0.5 × local` term makes the middle window's corrupted frames carry more weight (0.3917 vs 0.3583), a structural position dependence" | **downgraded to untested, and predicted to be tiny** | the pair-count algebra is right and I re-verified it for both window sizes (`T=16,w=4`: 0.3917 middle vs 0.3583 ends; `T=33,w=8`: 0.3565 vs 0.3409). But the all-pairs term is exactly position-neutral, and the local gap is only 0.0333 / 0.0156 of score weight — at a similarity drop of 0.1 that is 0.0033 / 0.0016, i.e. a small fraction of the Official spread and comparable to the *whole* residual repair spread. It is below the detection floor at n = 23 and is not the explanation for the headline gap. Treat as an **open hypothesis**, not a finding. |
| §5 remedy 1 "make the intervention temporally uniform" | **done** | `VBENCH_AUDIT_SUBJECT_BOX=median`, default since `61a65bc`; verified (§4.1) |
| §5 remedy 3 "report the halves separately" | **done** | `contract_split_cpa` now exists and is emitted; the two bugs in §6 are in its *rendering*, not its computation |

---

## 2. Verification that the report's own numbers are correct

Independent recomputation from `scores/subject_consistency__{official,repair}.jsonl`
(margins recalibrated from the 5 dev bases; 2000-replicate cluster bootstrap over
`base_id`, seed 2026) reproduced every published figure:

| quantity | published | independently recomputed |
|---|---:|---:|
| official test pooled, dev margin | 0.5917 [0.5583, 0.6250] | 0.5917 [0.5583, 0.6250] |
| repair test pooled, dev margin | 0.8500 [0.7583, 0.9333] | 0.8500 [0.7583, 0.9333] |
| official test sensitivity (dev margin) | 0.9333 [0.8500, 1.0000] | 0.9333 [0.8500, 1.0000] |
| official test invariance (dev margin) | 0.2500 [0.1500, 0.3500] | 0.2500 [0.1500, 0.3500] |
| repair test sensitivity (dev margin) | 0.7833 [0.6167, 0.9167] | 0.7833 [0.6167, 0.9167] |
| repair test invariance (dev margin) | 0.9167 [0.8167, 1.0000] | 0.9167 [0.8167, 1.0000] |
| pooled paired delta, tie-aware margins | +0.2583 | +0.2583, 95% CI [+0.1818, +0.3333] |
| dev margins | 0.01389 / 0.04457 | 0.01389385 / 0.04456929 |
| P1.5 equal-area medians (Official / Repair) | 0.1017 / 0.0215 | 0.1017 / 0.0215 |
| P1.5 tracked medians (Official / Repair) | 0.1299 / 0.0296 | 0.1299 / 0.0296 |
| P1.5 paired CI, Official / Repair | [−0.0586, +0.0135] / [−0.0213, +0.0132] | [−0.0625, +0.0135] / [−0.0227, +0.0132] |

The two P1.5 paired intervals differ in the lower bound because
`subject_position_profile.py` reports the bootstrap **percentile quantile of the
resampled medians**, while I report the quantile of the bootstrap **difference**;
both include zero, so no conclusion depends on the discrepancy, but the report
should state which one it prints. Everything else matches to four decimals.

**Status of the numbers in this review:** all of §2, §3, §4 and §5 are
recomputed from the frozen score tree; nothing here needs a new model run except
where explicitly marked "needs a re-score".

---

## 3. Q1 — Does the CONSOLIDATED row / headline survive the equal-area control?

**The composite delta survives as a number and not as a claim.**

### 3.1 The composite is the average of two halves that move in opposite directions

At the report's own margins the pooled test deltas are exactly the half-average:

```text
sensitivity  official 0.9333  repair 0.7833  delta −0.1500
invariance   official 0.2500  repair 0.9167  delta +0.6667
pooled delta = (−0.1500 + 0.6667) / 2 = +0.2583   ✓
```

Both entries reproduce the frozen `subject_consistency__cpa.json` and the
report's own `Contract decomposition` table. So `+0.2583` is not "Repair is
better on this family"; it is the net of a genuine **regression** on the
inequality half and a margin-dependent gain on the tie half. A composite whose
sign depends on how two opposing effects are averaged is not evidence for either
contract, and `CONSOLIDATED.md` already applies exactly this rule to
`multiplt_object` (where the control level was removed so the composite *is* one
half).

### 3.2 The composite's advantage is not margin-dependent, but it is margin-inflated

Comparing both backends at a **shared** margin removes the asymmetric-margin
advantage Repair gets from its looser dev margin:

| shared margin | official | repair | delta | paired 95% CI |
|---:|---:|---:|---:|---|
| 0.0139 | 0.5917 | 0.7417 | +0.1500 | [+0.097, +0.205] |
| 0.0200 | 0.6000 | 0.7833 | +0.1833 | [+0.117, +0.256] |
| 0.0446 | 0.6167 | 0.8500 | +0.2333 | [+0.141, +0.321] |
| 0.0600 | 0.5667 | 0.7833 | +0.2167 | [+0.143, +0.292] |
| 0.0900 | 0.4917 | 0.6917 | +0.2000 | [+0.111, +0.283] |

The sign is stable, so the composite gain is real in the sense that no shared
margin erases it. But 35 % of the published +0.2583 (0.2583 → 0.1500) is
produced by letting each backend keep its own dev margin, and the published
paired CI is computed under that configuration. The row must therefore be quoted
either with the margin-symmetric number or with an explicit note that the
margins differ.

### 3.3 The headline invariance pair `0.2500 → 0.9167` does not survive contact with the confirmatory run

Three independent reasons, in increasing severity:

1. **Different base set.** The published 25 bases and the P1.5 23 bases overlap
   in **9** bases (verified: `published-only = 16`, `P1.5-only = 14`). The
   published row and the P1.5 arms do not describe the same clips.
2. **Different construction.** The published row was scored on the **tracked**
   construction; P1.5 rebuilt the family with a fixed median box under
   `code_sha ccbd89b` (`subject_confirmatory/metadata/build_summary.json`).
3. **The invariance CPA is not a stable statistic.** Sweeping the margin over the
   equal-area arms on their own 19 test bases, with the *same* pairs:

   | margin | 0.01 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 |
   |---|---:|---:|---:|---:|---:|---:|
   | official | 0.5580 | 0.5942 | 0.5725 | 0.5362 | 0.5362 | 0.5362 |
   | repair | 0.6884 | 0.7826 | 0.7971 | 0.6304 | 0.5580 | 0.5000 |

   The ordering is **not** robust: at margin 0.02 Repair leads Official by 0.19,
   at margin 0.15 they are equal, at margin 0.20 they are equal at 0.50. A
   dev-margin recalibration on the equal-area arm (4 dev bases) gives
   **Official 0.9825 vs Repair 0.9474** — the *opposite* conclusion from the
   published `0.2500 → 0.9167`. The margin for repair (0.0446) is larger than
   the median relative range it is meant to absorb (0.0279), which is the
   mechanical reason this happens.

**Answer to Q1:** the row's arithmetic stands; its use as a Repair win does not.
The published `+0.2583` and `0.2500 → 0.9167` should both be marked *declining*
in `CONSOLIDATED.md`, and replaced by the margin-free dispersion result of §4.4.

---

## 4. Q2 — Does the equal-area arm actually remove the confound?

**Yes, mechanically, and I verified it rather than assuming it.** It does not,
however, make the position gap vanish — which is itself the informative result.

### 4.1 The construction change is real

From `subject_confirmatory/manifest.jsonl` and `subject_tracked/manifest.jsonl`,
per base, over the three `corrupt_*` rows:

| arm | identical window boxes across the 3 positions | each window box constant across frames | per-base max/min window area (median; min–max) |
|---|---:|---:|---|
| equal-area | **23 / 23** | **23 / 23** | **1.000** (1.000–1.000) |
| tracked | 1 / 23 | 1 / 23 | 1.333 (1.000–5.076) |

Window placement is identical in both arms (`start=0`, `middle=(T−w)//2`,
`end=T−w`, `w = round(0.25·T)`), so the only difference between the arms is the
box. In the equal-area arm the intervention is now **exactly** "same corruption,
different when", which is the premise the family needs. The previous review's
§4.2 confound is confirmed and closed.

### 4.2 Same-base paired comparison (not two different base sets)

Same 23 bases, same clips, only the box differs:

| backend | equal-area | tracked | delta (equal-area − tracked) | 95% CI | established? |
|---|---:|---:|---:|---|---|
| official | 0.1017 | 0.1299 | −0.0282 | [−0.0625, +0.0135] | **no** |
| repair | 0.0215 | 0.0296 | −0.0081 | [−0.0227, +0.0132] | **no** |

The point estimates are a 22 % and 27 % reduction in median relative range, and
the tolerance bands move the right way (Official ±10 %: 0.348 → 0.435; Repair
±5 %: 0.696 → 0.783), but **both intervals include zero**. At n = 23 the claim
"removing the confound reduced the measured spread" is not established, only
suggested. This matches `P1_NATURAL_AND_CONTROL_RUNS.md` §P1.5's own hedge
("reduced, not removed") — the report is honest here.

### 4.3 The direction of the residual is what matters, and it is not a construction artefact

With identical boxes at all three positions:

- **Official is still far from position-invariant**: median relative range
  0.1017, only **26 %** of bases within ±5 %, **43 %** within ±10 %.
- **Repair is roughly five times tighter**: 0.0215, **78 %** within ±5 %,
  **96 %** within ±10 %.

So the published direction — Official is position-dependent, Repair much less so
— survives on a construction where position is the *only* difference. What does
**not** survive is the specific published CPA pair, per §3.3.

### 4.4 The margin-free statistic is the one strong result in this dimension

Because the CPA invariance half is margin-dependent, the claim must rest on a
statistic that needs no margin. Two independent ones agree:

| statistic (same 23 bases) | official | repair |
|---|---:|---:|
| median relative range, equal-area | 0.1017 | 0.0215 (**4.7×**) |
| median relative range, tracked | 0.1299 | 0.0296 (**4.4×**) |
| bases where official is wider, equal-area | **21 / 23** | exact sign test **p = 6.6 × 10⁻⁵** |
| bases where official is wider, tracked | **20 / 23** | exact sign test **p = 4.9 × 10⁻⁴** |

The same comparison on the **published** 25 bases (margin-free, so it is
comparable despite the different base set): official 0.1154 vs repair 0.0279
(**4.1×**), official wider on **21/25**, p = 9.1 × 10⁻⁴. The effect is therefore
not an artefact of either base set.

Full per-base dispersion, for the report's plan-§7.3 block (test+dev bases as
available; median with IQR in parentheses):

| arm | backend | n | `max − min` | `(max − min) / mean` | CV |
|---|---|---:|---:|---:|---:|
| published | official | 25 | 0.0856 (0.0444–0.1129) | 0.1154 (0.0538–0.1337) | 0.0476 |
| published | repair | 25 | 0.0239 (0.0119–0.0324) | 0.0279 (0.0131–0.0388) | 0.0118 |
| equal-area | official | 23 | 0.0886 (0.0301–0.1224) | 0.1017 (0.0357–0.1497) | 0.0461 |
| equal-area | repair | 23 | 0.0183 (0.0083–0.0296) | 0.0215 (0.0089–0.0387) | 0.0097 |

Cluster bootstrap of the paired official − repair relative-range difference
(10 000 replicates, seed 2026), to check the sign test is not just
over-precision from treating bases as independent:

| arm | cluster unit | clusters | 95% CI |
|---|---|---:|---|
| published | `generator` | 4 | [+0.042, +0.099] |
| published | `group_id` / prompt | 25 | [+0.026, +0.114] |
| equal-area | `generator` | 4 | [+0.047, +0.112] |
| equal-area | `group_id` / prompt | 23 | [+0.037, +0.118] |

Every interval excludes zero, including the conservative 4-cluster one. Also
checked: **0/23** bases have an exactly-zero position spread in either backend,
so Repair's tighter spread is not an artefact of snapping to exact ties.

**Answer to Q2:** yes, the equal-area arm removes the confound (verified at
construction level); the *reduction* it buys is not significant at n = 23, but
the headline direction it was built to test survives it, margin-free and
cluster-robust.

---

## 5. Q3 — Do the report's contract-half numbers agree with P1.5?

**They are not the same measurement, and where they overlap the report is
arithmetically right but statistically fragile.** Point by point:

| question | report (`subject_consistency.md`) | P1.5 | consistent? |
|---|---|---|---|
| base set | 25 published bases (20 test) | 23 *different* bases (19 test), 9 overlap | **no — not comparable** |
| construction | tracked box (published corpus) | equal-area re-build + tracked control | no |
| statistic | tie-margin CPA per half | margin-free relative range | different instruments |
| sensitivity | 1.000 zero-margin, 0.9333 / 0.7833 at dev margins | 1.000 on all 23 bases × 2 backends × 2 arms | **yes, both saturate** |
| invariance direction | official 0.2500 < repair 0.9167 | official 4.7× wider (margin-free) | **yes in direction** |
| invariance ordering under a different margin | not tested | **inverts** (official 0.9825 > repair 0.9474) | **no — report's ordering is margin-specific** |

**Which is right?** The *margin-free* one. A tie-margin CPA answers "did the
score move by more or less than δ", and δ is a free parameter whose choice
decides the answer here (the sweep in §3.3 spans 0.50 → 0.80 for the same
repair arm). The relative range answers "how far apart are the three placements
relative to their own level", which is what plan §7.3 actually asks for
(maximum position gap and within-family score variance). The report's per-half
table is still worth keeping — it is the honest decomposition of the composite —
but it cannot settle the invariance question, and it should say so.

---

## 6. Two rendering defects that must be fixed before the report is quoted

Both were confirmed by reading the frozen `subject_consistency__cpa.json`
alongside the generated Markdown.

**6.1 The headline's paired interval is the wrong statistic's interval.**
`run_dimension.py` renders the `Official vs Repair (test, tie-aware)` table with
`ci_low`/`ci_high` taken from `paired["zero_margin"]`:

```text
report:      | temporal_relocation | 0.5917 | 0.8500 | +0.2583 | [+0.0000, +0.0000] |
cpa.json:    paired.tie_aware  = {delta +0.2583, ci [+0.1583, +0.3500]}   <- correct
             paired.zero_margin = {delta 0.0000, ci [0.0000, 0.0000]}     <- what got printed
```

The published `[+0.0000, +0.0000]` is impossible for a +0.2583 delta and directly
contradicts `CONSOLIDATED.md`, which quotes `[+0.158, +0.350]`. `CONSOLIDATED.md`
and `README.md` are right; `subject_consistency.md` is wrong.

**6.2 The per-half paired interval is computed over a different base set than its
table.** `run_dimension.py` pools **all** bases (`all_rows = rows`, both splits)
for `paired_halves`, and reports `n_bases = 25`, while the table above it is
labelled test-only with `n = 20` bases / 60 pairs:

```text
report:  | sensitivity | −0.1467 | [−0.2400, −0.0667] | 25 |   <- 25 bases, but the table above says "test"
cpa.json paired_halves.sensitivity = {delta −0.14667, ci [−0.24000, −0.06667], n_bases 25}
         paired_halves.invariance  = {delta +0.66667, ci [+0.53333, +0.77333], n_bases 25}   <- printed in the report
test-only equivalent (independently computed): invariance delta +0.6667, ci [+0.5333, +0.8000]
```

Either compute the halves on the test split only, or label the column
`dev+test (25 bases)`. As printed, a reader cannot tell which base set produced
the interval that "decides" the half.

**6.3 Minor.** The report header prints
`repair variant: ordered_role_identity_assignment (detection-conditioned: false)`.
That string is the **Spatial Relationship** repair mode, hardcoded as the
`--repair-mode` default and forwarded to every dimension
(`run_dimension.py` `--repair-mode`, `default=os.environ.get("VBENCH_AUDIT_SPATIAL_MODE", ...)`).
`subject_consistency`'s repair scorer ignores it (`score.py`
`dimension == "subject_consistency"` builds `backend_name = "audit"` with no
spatial config), so no score is affected, but the header is misleading and
should either be dimension-gated or dropped.

---

## 7. Q4 — Is the saturated sensitivity half unfalsifiable?

**At the current corruption strength, yes — and the report should say so
explicitly rather than presenting 1.000 as a pass.**

- On the published family the sensitivity half is **60/60** pairs at zero margin
  and 0.9333 / 0.7833 at the dev margins; on P1.5 it is **1.000 on 23/23 bases
  × 2 backends × 2 constructions**. A proportion of 23/23 has a 95%
  Clopper-Pearson interval of [0.852, 1.000]; it cannot distinguish "the metric
  is good" from "the corruption is overwhelming".
- The corruption is `GaussianBlur(6)` plus a 120° hue rotation over the subject
  box — a very strong edit. Both a first-frame-anchored metric and an all-pairs
  metric will rank it below the source.
- Therefore the half has **no discriminative power for the paper's comparison**:
  it cannot separate the two backends, and (worse) the one place it does move —
  the published dev-margin CPA, Official 0.9333 vs Repair 0.7833 — is evidence
  *against* the repair, which the pooled composite hides.

**How the paper should phrase it.** Not "both backends satisfy the sensitivity
contract and Repair improves invariance", but a single joint statement with the
ceiling named:

> On a single fixed corruption applied to identical subject regions at three
> temporal placements, both VBench 1.0's subject-consistency score and the
> trajectory-wide repair rank the clean clip above every corrupted placement in
> all 23 clips (a saturated result that does not discriminate between the two).
> The metrics do separate on *where* the corruption sits: Official's three
> placements differ by a median of 10.2% of their mean score, the repair's by
> 2.2%, and Official's spread is larger on 21 of 23 clips (exact sign test
> p < 10⁻⁴, cluster bootstrap over generators excludes zero).

Sensitivity should be demoted to a **precondition / ceiling check** (one sentence,
"not informative at this corruption strength"), and the falsifiable content of
the dimension should be the dispersion plus the tolerance band. If a
*discriminative* sensitivity half is wanted, the family needs a severity ladder
(e.g. blur radius or hue shift at 4–5 rungs) so that the relation
`clean > severity₁ > … ` is off the ceiling — that is a **new dataset
construction**, not a re-analysis, and I did not run it.

---

## 8. Q5 — Do 23 bases support "the invariance half is a win"? (power analysis)

Two different questions, and they have different answers.

**(a) "Does equal-area reduce the spread vs tracked?" — not established at
n = 23.** Per-base paired differences of relative range (10 000-replicate
bootstrap, seed 2026):

| backend | mean per-base diff | sd | minimum detectable difference (80% power, α = .05, 2.80·sd/√n) | n needed for the observed effect |
|---|---:|---:|---:|---:|
| official | −0.0198 | 0.0645 | **0.038** | **≈ 83 bases** |
| repair | −0.0034 | 0.0224 | **0.013** | **≈ 338 bases** |

The observed reduction is below the detection floor for both backends. The
honest statement is the report's own: *reduced, not established*.

**(b) "Official is more position-dependent than Repair?" — yes, comfortably at
n = 23.** The relevant paired statistic is much larger relative to its variance:

| arm | mean official − repair | sd | MDE at n = 23 | n needed for the observed effect |
|---|---:|---:|---:|---:|
| equal-area | +0.0762 | 0.0669 | 0.039 | **≈ 6 bases** |
| tracked | +0.0926 | 0.0828 | 0.048 | **≈ 6 bases** |

So the "win" that is supported is the **between-backend contrast** (4.7×, 21/23
signs, cluster-robust CI), not the **between-construction reduction**. Note also
that the *magnitude* "≈ 5×" is much less precise than its direction: 23 bases
put a wide interval around the ratio even though they settle the sign.

**Minimum for a paper claim:** 23 bases support "Official's spread exceeds the
repair's" with a non-parametric sign test and cluster-robust CI; they do **not**
support a numeric claim like "the repair reduces the spread by 27 %" or a
construct claim like "the equal-area construction fixed the confound". If the
construction point needs to be made, the required n is ~83 (official) or ~338
(repair) — i.e. the entire subject-consistency prompt pool, which the
detector-eligibility pass (25 scanned, 25 kept) says does not exist for the
tracked-box variant. Recommendation: **drop the construction-reduction claim
from the paper** and keep it in the appendix as a fixture note.

---

## 9. What this report should now say — executable checklist

**`subject_consistency.md` (regenerate with `run_dimension.py --report-only`):**

1. Fix §6.1: print `paired["tie_aware"]` in the `Official vs Repair` table; the
   cell must read `[+0.1583, +0.3500]`, matching `CONSOLIDATED.md`.
2. Fix §6.2: either compute `paired_halves` on the test split, or relabel the
   table `dev+test (25 bases)`; state the base count and pair count in the same
   table, not in a different one.
3. Fix §6.3: gate the `repair variant` header on dimension, or remove it.
4. In `CPA by contract half`, add one sentence: the invariance half is a
   tie-margin statistic whose value is a function of the margin (cite the sweep),
   and the margin-free relative range is the statistic that decides the contract.
5. Add the per-base `max − min` and `(max − min)/mean` medians with IQRs
   (plan §7.3 asks for both); they are already computable from the frozen scores.
6. Mark the sensitivity half as **saturated / not discriminative**, with the
   23/23 ceiling and the exact binomial interval, and note that the only
   non-saturated sensitivity reading (dev-margin 0.9333 vs 0.7833) favours
   Official.
7. Keep the `Frame evidence` caveat that no per-clip evidence exists for this
   run, so detector drop-outs cannot be ruled out; note that the equal-area arms
   make drop-outs less able to explain a *position* effect, since the box is
   identical at all three positions.

**`CONSOLIDATED.md` and `README.md`:**

8. Change the `subject_consistency` verdict from "the invariance half is the win"
   to: *composite is a mixture with opposite-sign halves; the established result
   is the margin-free position dispersion (Official 4.7× wider, 21/23 bases,
   p < 10⁻⁴); the `0.2500 → 0.9167` pair is margin- and base-set-specific and is
   superseded by P1.5.*
9. Mark the row's `+0.2583` as **declining** (like `scene` and
   `spatial_relationship`), keeping the number visible for provenance, and add
   the margin-symmetric counterpart (+0.15 to +0.23 depending on the shared
   margin).
10. Add the base-set provenance note next to the row: published 25 bases vs P1.5
    23 bases, **9 overlap**; the confirmation run is not a re-measurement of the
    published row.
11. If the paper quotes one number for this dimension, quote
    **0.1017 vs 0.0215 relative range, 21/23, p < 10⁻⁴** — never the CPA pair.

**Claim boundary for the manuscript:**

12. Allowed: "VBench 1.0's subject-consistency score is position-dependent: the
    same corruption applied to identical regions at the start, middle and end of
    a clip moves the score by ~10 % of its own level, and an all-pairs
    trajectory-wide aggregation reduces that to ~2 %."
13. Not allowed: "the repair improves subject consistency" (composite mixes a
    −0.15 sensitivity regression), "position invariance 0.25 → 0.92" (margin- and
    base-set-specific), or "the equal-area construction removes the confound"
    (reduced, not established at n = 23).

## 10. What would be needed to settle this dimension

| open point | data/experiment needed |
|---|---|
| Is the residual repair spread (> 0) aggregation-boundary or representation noise? | score the equal-area arm with the local term removed and with endpoint-replicated local pairs; same 23 bases, no new clips |
| Is the sensitivity half discriminative at all? | a severity ladder (4–5 corruption strengths) on the same bases; requires rebuilding the family and re-scoring ×2 backends |
| Does the position effect generalise beyond 23 bases? | the detector pool for `subject_consistency` is exhausted (25 scanned, 25 kept); a larger claim needs a looser subject definition or a different detector, i.e. new construction |
| Does the position effect survive on natural (unmodified) VBench generations? | natural-set run for this dimension; not attempted in P1 |

## 11. Residual uncertainty in this review

- I recomputed from the frozen score tree only; no model was re-run, so the
  underlying DINO features and the upstream VBench checkout were not re-verified.
  Model/CUDA/weight parity against the frozen E0 baselines remains unverified,
  exactly as the report states.
- The per-half bootstrap and the margin sweep use 2000–10 000 replicates with
  seed 2026; I did not test seed sensitivity, but the sign tests and
  cluster-bootstrap intervals in §4.4 do not depend on the bootstrap seed.
- The `subject_confirmatory` and `subject_tracked` arms were built at
  `ccbd89b`/`b8e1512`, i.e. before the report's `5c4a1309`; the P1.5 arm was
  scored with the current `score.py` (evidence fields absent, so that path is
  unchanged), but the arms were not re-scored at `5c4a1309`. The statistics
  script was run at the revision recorded in `position_profile.json`.
