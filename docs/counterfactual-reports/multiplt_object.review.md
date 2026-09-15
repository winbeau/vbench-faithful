# Review: `multiplt_object` counterfactual report

Scope: `docs/counterfactual-reports/multiplt_object.md` (code SHA
`c73495e6ffdc44b96d55e1e29d7eab21766c080b`), the `multiplt_object` row of
`docs/counterfactual-reports/{CONSOLIDATED.md,README.md,SUMMARY.md,table2.csv}`,
the instrument that produces them (`scripts/counterfactual/{cpa,run_dimension,
summarize,transforms,score,pick_detectable}.py`), `configs/counterfactual/`
(`README.md`, `bases_published.jsonl`), and the plan's sections 5.2, 11.3, 11.4
and 15.2.

Reviewed 2026-09-15 as an adversarial re-review at revision `52e4b76`, against
the frozen score tree
`/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/` on `h100-server`.
Previous revision of this file reviewed the 6-level ladder at `bdfda5c`; every
verdict it reached is re-tested below and the changes are called out explicitly.

**Verdict.** The published numbers are arithmetically exact and the headline is
the strongest ordered result in the audit: on the occlusion-only ladder,
Official 0.5450 → Repair 0.7850 (109/200 → 157/200 pairs), paired Δ **+0.2400**
[+0.155, +0.330], 17/20 test bases move Repair's way, 0 move Official's way, and
a leave-one-out sweep keeps the delta in [+0.216, +0.253]. Three things must
travel with it, and one of them is a defect:

1. **More than half the delta is tie-handling, and only an eighth is new
   discrimination.** Exact outcome decomposition of the 200 test pairs (every
   pair's label is `+1`):

   | Official | Repair | pairs | contribution to Δ |
   |---|---|---:|---:|
   | correct | correct | 104 | 0 |
   | tie | correct | 42 | **+0.2100** |
   | tie | inverted | 25 | **−0.1250** |
   | wrong | correct | 11 | **+0.0550** |
   | correct | wrong | 5 | −0.0250 |
   | wrong | wrong | 13 | 0 |

   Net tie-handling is +0.0850, net strict-versus-strict discrimination is
   +0.0300. So the Repair does not order 48 more pairs than Official because it
   *sees* more ordering — it wins 42 ties, loses 25 of them the other way, and
   only 12 pairs in 200 are decided by a strict comparison the Repair wins that
   was not already a tie or a loss.
2. **The report is internally mixed-generation.** It is labelled `ordered` in the
   tables while its own `## CPA by contract half` and `## Contract decomposition`
   sections still describe a mixture, and the `invariance` row of the per-half
   paired table reports a level pairing (`conjunction_control` vs
   `occlusion_100`) that the same report's `contract_split` declares to have
   **0 pairs**. That row describes a comparison that is no longer in the ladder
   and must be removed or relabelled.
3. **The fixture is detector-conditioned.** `multiplt_object` kept 25 of 49
   scanned candidates (24 rejected because GRiT could not ground the named
   targets) and four test bases suppress under 2 % of the frame (62× area span).
   The dimension therefore measures ranking *among clips GRiT can see*, which is
   a weaker claim than the tables' unqualified phrasing suggests.

Net: **+0.2400 is real and reproducible, and it is not a pure ordering gain.**
It is "Repair is a strictly finer estimator than Official on this ladder, on the
subset of bases where GRiT grounds both targets". That is worth claiming — it is
the only clean zero-excluded interval in the audit — but not as "the cleanest
ordered Repair win" without the saturation, fixture and mislabelling caveats.
The area objection raised in the previous round is now tested and **does not**
survive: dropping the small-area tail or the non-responsive bases leaves the
interval above zero (+0.2000 to +0.2375; §4).

## 1. Scope of the change under review

The ladder is now occlusion-only. `conjunction_control` is listed in the report
header as `control levels (excluded from the ordered ladder)` and
`run_dimension.py` implements that by building every rank-based statistic from
`ladder = [row for row in rows if row["level"] not in CONTROL_LEVELS]`
(`run_dimension.py:1196`), with `CONTROL_LEVELS = ("conjunction_control",)` at
line 49 and the rationale in the comment at lines 43-48. `--report-only`
regenerates the artefact; nothing was re-scored this round, and the per-clip
score files are timestamped 19:49 while the report is 21:37, consistent with a
report-only refresh over unchanged scores.

## 2. Q1 — Is removing `conjunction_control` from the ladder justified?

**Yes, and it is what this reviewer asked for — but the arithmetic effect on the
headline is larger than "presentational", and neither the report nor
`CONSOLIDATED.md` quantifies it.** Two separate questions:

**(a) Is it legitimate?** Yes.

- The plan does not declare the control equal to `occlusion_100` anywhere.
  §11.3 is an ordered family; §11.4 is a *control* whose stated requirement is
  "Both Official and Audit should score this as incomplete". Equality between a
  same-frame full occlusion and a clip in which no frame holds both targets was
  a construction choice made in the manifest (`expected_rank=0` on both rows),
  not a read-off from the plan.
- The two levels differ in far more than severity: `occlusion_100` suppresses
  the weak target in all 16 frames, the control suppresses A in the first half
  and B in the second. A tie between them is not a severity tie.
- The previous review reached the same conclusion (§3 there, and its §6 asked
  for the tie to be demoted to a separate statistic). The implementers' comment
  at `run_dimension.py:43-48` matches that reasoning.

**(b) What did it cost?** Removing the control is not neutral for the headline,
because the composite changed *pool*, not only composition:

| statistic (test split) | Official | Repair | Δ |
|---|---:|---:|---:|
| 6-level pool, as previously published (300 pairs) | 179/300 = 0.5967 | 229/300 = 0.7633 | +0.1667 |
| occlusion-only ladder (200 pairs) | 109/200 = 0.5450 | 157/200 = 0.7850 | **+0.2400** |

The 20 control pairs contributed `12/20` to Official and `0/20` to the Repair, so
removing them *lowers* the measured gap by 0.04; the remaining **+0.0733** is a
composition effect — moving from a 5-rung ladder to a 4-rung ladder re-weights
which rank gaps the composite averages, and Official collapses to 0 on the
twice-counted rung it used to be scored against. Both effects are legitimate,
but a reader comparing the two vintages will see the delta rise from +0.1666 to
+0.2400 and should be told that 45 % of that rise is the re-weighted pair
composition, not new evidence. `CONSOLIDATED.md`'s footnote describes *why* the
control left but not *what it did to the number*.

**Alternatives, in order of preference.** (i) Keep the ordered ladder as the
headline and report the 6-level composite beside it as a deprecated comparator
with the one-line accounting above. (ii) Report the literal plan §5.2 pairing
(base vs each counterfactual) as well: on test that reads Official 47/80 = 0.5875
vs Repair 64/80 = 0.8000 (Δ +0.2125), i.e. the win survives the alternative
pairing and is 0.028 smaller. (iii) Do **not** simply keep the old mixture and
hope the reader splits it; the previous round showed that does not happen.

## 3. Q2 — Can +0.2400 and the old §2 reading coexist?

**They coexist only if the tie contract is formally dropped, and the frozen
artefact does not currently drop it consistently.**

- The old review §2 said the headline was "80 % sensitivity and 20 % an
  invariance contract the Repair fails". The 20 % is now provably not a contract:
  §11.4 never asks for a tie, and the previous review's own §3 established that
  the pairing is confounded. So §2's *conclusion* ("the composite nets a win
  against a loss") is **superseded** — the composite no longer contains that
  pairing.
- But the tie criterion has not been removed, only relocated.
  `multiplt_object.md` still prints `| invariance | -0.6000 | [-0.8000, -0.4000] |`
  in the per-half paired table while its own `## CPA by contract half` table
  shows `invariance n_pairs = 0` for both splits and both backends. Both numbers
  are literally reproducible from the frozen tree — `paired_halves` is computed
  over the 6-level pool (`run_dimension.py:1266-1284` builds it from
  `_base_pairs(rows, …)` on all rows, control included) while `contract_split`
  runs on `ladder` (line 1196) — but a report that says "0 pairs" three lines
  above "-0.6000" is not a single measurement.
- Whether the excluded tie is *still a contract requirement*: **no**. The
  contract that remains is §11.4's level predicate, and by that standard the
  Repair is not merely passing, it is strong on the level scale: control mean
  0.2009 against its own clean level 0.5368 (−63 %), with the control scored no
  higher than `occlusion_100` in 18/25 bases against Official's 22/25 and a
  lower control mean (0.2009 vs 0.2436).
- What the exclusion *legitimately* hides is a separate, real property: the
  Repair never reaches zero on this dimension. On test, `occlusion_100` scores
  exactly 0.0 in 13/20 Official clips but **0/20 Repair clips** (Repair median
  0.1106); the control scores exactly 0.0 in 12/20 Official but 0/20 Repair
  (median 0.1534), and the two are separated by only 0.043 in the mean. The
  earlier review framed this as "no exact tie"; the sharper statement is that
  **absence is not a fixed point of the repair's score**, so a strength-2 union
  and a strength-4 absence land within 0.05 of each other. That is worth an
  explicit line in the report and is a candidate improvement for the metric
  (a floor or a detection-conditioned gate), not a reason to restore the tie.

**Answer in one line:** the tie criterion was never a contract, so excluding it
from the composite is right; it is still being reported inconsistent with the
ladder it now sits outside, and the non-saturation finding it was gesturing at
has not been carried over.

## 4. Q3 — Does the 62× weak-target area span break the ladder, and is +0.2400 just "bigger box, easier"?

**The span is real; the simple area explanation is not supported; a subtler area
confound is not excluded.**

Verified from the manifest's per-frame tracked boxes (`transformation_parameters
.boxes` of the `occlusion_000` row, median area over 16 frames divided by
`width × height`) — the frozen `weak_target_area` block agrees: n = 20 test
bases, median **0.1575**, IQR **0.0350–0.3446**, range **0.0066–0.4101**, span
**62.2×** (the same span holds over all 25 bases). Four test bases are under 2 %
and six are under 5 %.

Stratifying the 200 test pairs by area does **not** show the win living in the
large boxes:

| stratum | Official | Repair | Δ | paired Δ (95 % CI) |
|---|---:|---:|---:|---|
| smaller half (n=10) | 0.590 | 0.810 | +0.220 | +0.220 [+0.000, +0.450] |
| larger half (n=10) | 0.500 | 0.760 | +0.260 | +0.260 [+0.090, +0.440] |
| Q1 (0.007–0.033) | 0.440 | 0.700 | +0.260 | — |
| Q2 (0.036–0.155) | 0.740 | 0.920 | +0.180 | — |
| Q3 (0.160–0.342) | 0.400 | 0.680 | +0.280 | — |
| Q4 (0.351–0.410) | 0.600 | 0.840 | +0.240 | — |

`Pearson(area, per-base Δ) = +0.101`, `Spearman = +0.110` — no monotone "larger
box wins bigger" trend, and the smallest-area quartile carries a gain as large
as the largest. Per-base deltas are +0.240 mean, +0.200 median, 17 positive, 0
negative, 3 zero.

**The cheap area control is now done, and the headline survives it** (paired
bootstrap, 2 000 iterations, frozen tree only — this is checklist item 6,
executed as part of this review rather than deferred):

| subset | bases | Official | Repair | Δ (95 % CI) |
|---|---:|---:|---:|---|
| all | 20 | 0.5450 | 0.7850 | +0.2400 [+0.1550, +0.3300] |
| drop area < 0.02 | 16 | 0.5875 | 0.8250 | +0.2375 [+0.1500, +0.3312] |
| drop area < 0.05 | 14 | 0.5714 | 0.8071 | +0.2357 [+0.1357, +0.3500] |
| drop bases where Official is constant across rungs | 18 | 0.6056 | 0.8056 | +0.2000 [+0.1222, +0.2833] |
| drop bases with Repair rung spread < 0.10 | 17 | 0.5647 | 0.7882 | +0.2235 [+0.1353, +0.3176] |
| drop both (area ≥ 0.02 and spread ≥ 0.10) | 16 | 0.5875 | 0.8250 | +0.2375 [+0.1500, +0.3312] |

Every subset keeps the lower bound above zero, so the plan §15.2 criterion still
holds after the small-area tail and after the two bases where Official is
degenerate. **The +0.2400 is not an artefact of the 62× span.**

**What the controls do show, and what is still not excluded.** The smallest-area
bases are where the *fixture* is stressed, and the effect is visible even though
it does not drive the headline:

- on base `…858774` (area 0.0071) Official returns 0.0 on all five rungs, so all
  ten pairs are ties and it scores 0/10 by construction; the Repair scores 7/10
  on a spread of 0.028. Two bases (`…858774`, `…48250c`) are officially flat.
- on base `…a09a72` (area 0.0066, the smallest) **both** backends order worse
  than chance and the Repair's control score (0.430) sits above every rung, i.e.
  the union clip outranks the clean clip.
- dropping flat/spread-less bases moves the delta from +0.2400 to +0.2000, so
  roughly one sixth of the headline comes from bases where one backend has
  little or nothing to order.

So the remaining caveat is not "bigger box, easier win" but "a few bases carry no
usable ladder, and their disagreement is counted as Repair superiority". A
severity-level validity gate (per-base monotonicity of the *intervention*, not of
the score) is what would remove it; an area filter alone does not.

**Still not excluded / not run:** (a) an area-normalised occlusion that fixes the
*absolute* suppressed area across bases and re-derives the clips; (b) per-base
normalisation of scores by that base's `occlusion_000` level before pairing. Both
would strengthen the claim but require either re-derivation or a new statistic;
neither is needed to make the current interval defensible.

## 5. Q4 — Does the frame evidence support "ordered win"?

**It supports "strictly finer ordering", not "clean ordering", and the report
prints it without that distinction.**

Frozen `order` block, occlusion-only ladder (5 rungs, 10 pairs/base):

| backend | split | bases | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|---:|
| official | dev | 5 | 0.8008 | 0.8008 | 0/5 |
| official | test | 20 | 0.6204 | 0.7225 | 0/20 |
| repair | dev | 5 | 0.7000 | 0.9000 | 2/5 |
| repair | test | 20 | 0.6550 | 0.9000 | 3/20 |

Frozen rank-gap decomposition (test):

| rank gap | pairs | Official match (tie rate) | Repair match (tie rate) |
|---|---:|---:|---:|
| 1 | 80 | 0.4500 (0.4250) | 0.7000 (0.0000) |
| 2 | 60 | 0.5167 (0.3500) | 0.7833 (0.0000) |
| 3 | 40 | 0.6250 (0.2500) | 0.8750 (0.0000) |
| 4 | 20 | 0.8500 (0.1000) | 0.9500 (0.0000) |

Reading:

- **Supports the claim:** at every rank gap the Repair matches more pairs, the
  advantage is largest exactly where Official saturates (gap 1-2), the Repair has
  zero ties, its median Spearman rises 0.7225 → 0.9000, and its strict-order
  count rises 0 → 3/20. Three of twenty bases are perfectly ordered by the
  Repair and none by Official.
- **Undercuts "clean":** the per-base strict-order rate is 15 %, so 17 of 20
  test bases are *not* in the declared order; on dev the Repair's mean Spearman
  (0.7000) is **below** Official's (0.8008); and Official's gap-4 match rate of
  0.8500 shows it does order large severity separations well (the Repair's gap-4
  edge is only +0.10). Combined with the outcome decomposition in the verdict —
  42 ties repaired, 25 ties inverted, 11 strict errors fixed, 5 strict wins lost
  — the rank-gap table is better read as "the Repair separates near-identical
  clips that Official collapses" than as "the Repair orders the ladder". The
  declared relation is monotone in the aggregate means (Repair 0.5368 → 0.5097 →
  0.4959 → 0.4396 → 0.2487) but not per base.
- The frame-evidence table is consistent with the rest of the report (Official
  and Repair agree exactly on mean joint co-presence, 0.3802, because the Repair
  reads the same official-threshold pass), and the Repair's mean weakest-target
  confidence of 0.3510 is the floor the ladder is fighting. What the table does
  not show is the per-level decomposition (verified separately: weakest
  confidence 0.481 → 0.454 → 0.440 → 0.382 → 0.189 → 0.160 control), which is
  the actually informative shape — the detector's confidence falls by only ~0.29
  absolute across the whole ladder.

**Bottom line:** the number is a real improvement over Official on this ladder,
but "ordered Repair win" should be qualified as *relative to a saturated
baseline, with no strict ordering on 85 % of bases*.

## 6. Q5 — Does 24/49 detector rejection mean this dimension tests GRiT, not the metric?

**It means the claim is conditional on GRiT grounding, which the report does not
state anywhere in its own text.**

Durable counts (`configs/counterfactual/README.md`, from `pick_detectable.py
--summary`): `multiplt_object` scanned 49 ranked candidates, kept 25 (5 dev /
20 test), rejected **24** for undetectable targets; `subject_consistency` scanned
25 and rejected 0. My independent check of the published selection against the
current base list finds 25/25 overlap, so the published dataset and the current
selector agree; the earlier 6/25 discrepancy is closed by
`configs/counterfactual/bases_published.jsonl`.

What this does and does not imply:

- It is **not** a scoring artefact: every reported score was computed on a
  published clip, coverage is 120/120 + 30/30 on both backends, and no clip was
  scored as zero or dropped after scoring.
- It **is** a selection effect in two directions. First, the kept bases are
  exactly those where GRiT grounds both bare nouns in the official
  `object_en` annotation, using its own category names — so the sample is
  conditioned on the same detector that the Repair's confidence channel uses.
  Second, the rejected half is not random: two-object prompts whose second
  target GRiT will not name are precisely the compositional cases the dimension
  exists to probe. `CONSOLIDATED.md` and `P1_NATURAL_AND_CONTROL_RUNS.md` record
  the count as a *fixture* finding (P1.4 item 4) but the report itself never
  mentions it, and the tables' "25 bases" gives no hint that half the candidate
  pool was dropped for detector reasons.
- The direction of the bias on the metric claim is not obvious: within the kept
  set the Repair wins, but the rejected set is unmeasured for both backends, so
  nothing here says how either metric behaves on the prompts GRiT cannot ground.
  That is a scope limit on the claim, not a refutation of it.

**Required addition:** one sentence in `multiplt_object.md`'s limitations —
"25 of 49 ranked candidates were rejected because GRiT could not ground both
targets; this dimension measures ranking only on prompts where it can" — plus
the same qualifier wherever the row is called "the cleanest ordered Repair win"
(`CONSOLIDATED.md` "What can still be claimed", `README.md` line 57).

## 7. Conclusions from the previous revision: kept, superseded, added

| previous claim | status now | note |
|---|---|---|
| §1 pair census, profiles, margins consistent | **kept, updated** | 15/base → 10/base on the ordered ladder; profiles are now split dev/test; margins remain 0.0 for both backends |
| §2 "headline is 80 % sensitivity + 20 % a failed invariance contract" | **superseded** | the invariance pairing is no longer in the composite; the ordered half now *is* the composite. The 6-level sensitivity Δ +0.2800 and invariance Δ −0.6000 still reproduce, but as a different (dev+test, 6-level) pool |
| §3 "rank-0 is structural, no confidence rule separates present from suppressed" | **kept and strengthened** | verified again: the Repair scores no test clip at exactly 0.0 on either `occlusion_100` (min 0.0473) or the control (min 0.0473); mean weakest confidence is still 0.3510 and the repair's control mean sits 0.043 above full occlusion |
| §3 "Official's 0.60 is saturation, not fidelity" | **kept, now quantified** | 67/200 test rung pairs tie exactly under Official vs 0/200 under Repair, and 42 pairs are won purely by breaking those ties |
| §4 "the instrument measures a graded ladder as many binary pairs, and the plan's literal pairing is base-vs-rung" | **partly superseded** | the ladder is now occlusion-only, so the pair set is the declared one; the base-vs-rung reading still differs (Official 0.5875 vs Repair 0.8000, Δ +0.2125) and is worth printing |
| §4 "the ordered statistic is likely more favourable than CPA and its absence understates the Repair result" | **falsified** | Spearman mean 0.6204 → 0.6550 and strict order 0/20 → 3/20 are favourable but modest, and the Repair's dev mean Spearman (0.7000) is below Official's (0.8008) |
| §5.1 "no intervention verification" | **closed** | the report now carries construction check + frame evidence, and this review verified per-level weakest confidence and co-presence |
| §5.2 "level profiles pool dev and test" | **closed** | profiles are split by split in the frozen artefact |
| §5.3 "weaker-target choice can be a coin flip, no area table" | **partly closed** | the area table exists; the near-tie choice rule and the 24 rejections are still not in the report body |
| §5.4 "naming: multiplt_object vs multiple_objects" | **open** | unchanged |
| §5.5 "difficulty heterogeneity untested" | **closed here** | per-base deltas: 17 positive, 0 negative, 3 zero; leave-one-out Δ ∈ [+0.2158, +0.2526] |
| §3 / §6 "the 62× weak-target area span makes severity incomparable across bases" (raised as a fixture finding in P1.4) | **tested and downgraded** | the headline survives every area/validity filter (Δ ∈ [+0.2000, +0.2375], all CIs clear zero); the residual issue is 2 officially-flat bases and 3 with repair spread < 0.10, worth ~0.04 of the delta |
| §6 "row must carry the split verdict; CI overlap" | **superseded** | the row now carries a paired interval that clears zero; the older "Repair CI overlaps Official" objection is obsolete because the marginal intervals are no longer the decision statistic |
| §8 "v2 transform prototyped and reverted" | **kept** | still the reason the fixture, not the metric, carries the remaining confound |

## 8. Acceptance checklist (executable)

1. **Fix the mixed-generation report** (`scripts/counterfactual/run_dimension.py`,
   then regenerate the artefact with `--report-only`): (a) when
   `contract_split[*]["invariance"]["n_pairs"] == 0`, drop the `invariance` row
   from the per-half paired table, or relabel it `invariance (6-level control
   pool, not in the composite)`; (b) gate `## Contract decomposition` and the
   "declares levels that must tie" preamble on `mixed_family` computed from the
   *ladder*, so an ordered family does not ship mixture prose; (c) fix the header
   `- levels:` line, which still lists `conjunction_control` first among levels
   while a later line calls it excluded.
2. **Quantify the ladder change where the number is quoted** —
   `CONSOLIDATED.md` footnote and `README.md` ‡: state that the published delta
   moved 6-level → occlusion-only, that the control pairs were worth 12/20 → 0/20,
   and that the remaining +0.0733 is re-weighted pair composition.
3. **Carry the outcome split into the tables**, not just the rank-gap split:
   "of the +0.2400, +0.2100 comes from ordering 42 pairs Official leaves tied and
   −0.1250 from inverting 25 other ties; only +0.0300 is strict-versus-strict
   discrimination". Without it the win reads as a detection advantage it is not.
4. **Add the non-saturation finding** to the report: Repair scores 0.0 on 0/20
   `occlusion_100` and 0/20 control test clips (Official: 13/20 and 12/20), so
   absence is not a fixed point and temporal union sits only 0.043 above full
   occlusion on the mean. This is the honest descendant of the retired tie
   criterion.
5. **State the fixture scope in the report itself**: 49 scanned, 24 rejected,
   62× area span, 4/20 bases under 2 % of frame; and soften the
   "cleanest ordered Repair win" phrasing in `CONSOLIDATED.md` and `README.md`
   to "cleanest zero-excluded ordered result, conditional on GRiT grounding".
6. **Report the area control that this review ran** (drop area < 0.02 → Δ
   +0.2375 [+0.1500, +0.3312]; drop Official-flat bases → +0.2000
   [+0.1222, +0.2833]) so the 62× objection is answered in the artefact. Optionally
   add a per-base severity-validity gate for the 2 flat and 3 near-flat bases,
   which is worth ~0.04 of the delta and is a fixture fix, not a scoring fix.
7. **Print the plan §5.2 base-vs-rung pairing** (Official 0.5875 vs Repair
   0.8000 on test) as a second column, since it is the literal contract reading
   and it is 0.028 smaller.
8. **Reconcile the two "pair sets" in the instrument** so this class of
   inconsistency cannot recur: have `paired_halves` take the same `ladder` as
   `contract_split`, or record the pool it used in its own output.

## 9. How the numbers in this review were obtained

All statistics were recomputed on `h100-server` from the frozen score tree with
the CPU interpreter, no GPU and no model run:

- per-clip scores: `scores/multiplt_object__{official,repair}.jsonl`;
- headline, halves, rank gaps, pair census, control predicate, order statistics:
  recomputed with the same `family_pairs`/`cpa_at_margin` semantics as
  `scripts/counterfactual/cpa.py` and reproduced the frozen
  `scores/multiplt_object__cpa.json` — the 5-rung composite (109/200 vs 157/200,
  paired +0.2400, 6-level composite 179/300 vs 229/300, 6-level sensitivity
  188/350 vs 286/350, 6-level invariance 15/25 vs 0/25) and the frozen profiles
  all match. The paired interval is [+0.155, +0.330] under
  one-sample-per-iteration resampling, which is what `paired_bootstrap_ci`
  implements; an independent resample per backend widens the same interval to
  [+0.100, +0.385], which is the wrong pairing and was discarded;
- areas: from `manifest.jsonl` per-frame `transformation_parameters.boxes`,
  median area / (width × height), matching the frozen `weak_target_area` block;
- selection counts: `configs/counterfactual/README.md` and
  `bases_published.jsonl` vs the manifest's base ids (25/25 overlap);
- scripts used (scratch, not committed):
  `output/counterfactual/_scratch/multiplt_verify.py`,
  `multiplt_verify2.py`, and `/tmp/verify{3,4,5,6,7}.py` (halves and their pools,
  headline accounting, leave-one-out, small-area ladder validity, area/validity
  controls, outcome decomposition).

Standing limitations, unchanged from the consolidated report: these are
first-run measurements from the current metric packages; model/CUDA/weight parity
against the frozen E0 baselines is **not** verified; and the absolute score
levels are not reproduced official baselines.
