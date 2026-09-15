# Review: `multiplt_object` counterfactual report

Scope: `docs/counterfactual-reports/multiplt_object.md` — first reviewed at code
SHA `bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`, refreshed after the diagnosis
below was acted on — the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `run_dimension.py`, `score.py`,
`summarize.py`, `transforms.py`, `build.py`), and the `multiplt_object` row of
`docs/counterfactual-reports/README.md`.

Verdict: **the sensitivity half of this row is a real, significant Repair win
(+0.2800, 95% paired-bootstrap CI [+0.1886, +0.3771]); the tie-based invariance
half is 0.00 against Official 0.60 (paired delta −0.6000, CI [−0.8000, −0.4000])
and is not a criterion any continuous estimator can satisfy.** The composite
`+0.1666` (paired 95% CI [+0.0900, +0.2500], so it clears the plan's §15.2
success criterion) is a sensitivity gain net of an invariance loss. Section 3
settles the cause: not the SoftMin/`beta` configuration and not the aggregation
reading temporal union as co-presence, but the detector's confidence floor — no
confidence-level rule can separate "target present" from "suppressed target
still answered at 0.2-0.5". The plan's actual §11.4 predicate is now reported
directly, and the README's "genuine Repair win with non-overlapping intervals"
claim does not hold for this row on the intervals (they overlap on
[0.6767, 0.6967]).

## 1. Arithmetic that does check out

So the record is clear about what is not in dispute. Derived from the published
integers and the pair census (`family_pairs`, `cpa.py:58-72`; 6 levels → 15
unordered level pairs per base = 1 rank-0 + 14 rank-positive in the 25 bases):

| quantity | value | check |
|---|---|---|
| clips | 150 = 25 × 6 | report header |
| test pairs | 300 = 20 × 15 | 100 + 80 + 60 + 40 + 20 from the rank gaps |
| sensitivity / invariance | 350 / 25 | 14 + 1 per base |
| rank-gap census | `rank1..4` = 100/80/60/40 | spacing counts 5+4+3+2, exact |
| Official test composite | 179/300 = 0.5967 | 167 sensitivity + 12 invariance |
| Repair test composite | 229/300 = 0.7633 | 229 sensitivity + 0 invariance |
| Official whole-family halves | 188/350, 15/25 | published as 0.5371 / 0.6000 |
| dev margin | 0 for both backends | "dev margin" = "zero margin" columns are equal |
| profiles | n=25 per level, min/max/distinct consistent | — |

Three items do **not** quite close and should be re-derived on the scoring
host before publication:

1. The same official sensitivity half appears as **0.5371** in the
   contract-half table (188/350, dev+test pooled) and as **0.5964** in the test
   block (167/280, reconstructed from the 0.5967 composite minus the 12/20
   invariance pairs). Pooling instead from the published dev row (0.3200 over
   75 ⇒ 24 correct) gives dev sensitivity 21/70 = 0.3000, i.e. pooled
   188/350 = 0.5371 — so the two halves are internally consistent with each
   other *only* if dev and test really differ that much (0.30 vs 0.60), which
   is itself a finding left unstated in the report. The residual is ≈ 1.5
   pairs, at the edge of the published 4-decimal rounding; it should still be
   pinned down on the scoring host, because "the same quantity is printed twice
   with two different values" is exactly what a review must not wave through.
2. The report prints no `dev_margin` value at all. Because both margins are 0,
   the zero-margin and tie-aware columns are *identical by construction*
   (`cpa.py:126-132` picks 0.0 whenever it wins), which means the tie-aware
   machinery is inert here and the reader cannot tell that from the tables.

3. Related, and more consequential: the Official backend's own dev and test
   results differ enormously (pooled dev 0.3200 vs test 0.5967), while the
   Repair is stable (0.7600 vs 0.7633). The 5 dev bases the margin is
   calibrated on are not representative of the 20 test bases for Official, so
   "the dev-calibrated margin" (and the whole tie-aware column for this
   dimension) rests on a 5-base calibration set whose pool behaves unlike the
   test pool. This deserves an explicit caveat next to the tie-aware row.

## 2. The headline is 80 % sensitivity and 20 % an invariance contract the Repair fails

Published test/whole-family halves:

- sensitivity: Official 0.5371 → Repair 0.8171 (**+0.28**);
- invariance: Official 0.6000 → Repair 0.0000 (**−0.60**).

Composition into the +0.1666 composite, using the pair weights
P(sensitivity) = 280/300 and P(invariance) = 20/300:

```text
sensitivity: (0.8171 - 0.5371) x 280/300 = +0.2613
invariance:  (0.0000 - 0.6000) x  20/300 = -0.0400
                                            +0.2213
```

(the published accuracies are rounded to 4 decimals and the 0.5371 half is
pooled over dev+test, so the reconstructed total, +0.2213, exceeds the published
composite delta +0.1666 by 0.055 = 16.5 pairs; the *decomposition direction* —
positive sensitivity, negative invariance — is what the tables establish, and
the reconciliation is one more reason to re-derive the halves on the scoring
host, section 1.)

Either way the reading is the same: the composite moves up only on the
sensitivity contract and the invariance half moves *against* the repair. The
invariance half is 20 of the 300 test pairs, i.e. 6.7 % of the composite, and it
is the only half the report calls the family's target. Playing the report's own
instruction ("whether a Repair gain in the composite comes from sensitivity …
or from the invariant half", lines 81-83) straight: **it comes from
sensitivity, and the invariant half supplies none of the gain.** That sentence
has to be in the report and in the README, not left as an exercise.

## 3. The rank-0 pair is a contract the plan never declared, and it is confounded

**Diagnosis (settled on the scoring host).** The Repair's 0/20 rank-0 result is
*neither* a SoftMin/`beta` configuration problem *nor* the aggregation reading
temporal union as co-presence. It is structural: the continuous confidence
channel never reports absence, and the family's rank-0 pair asks a continuous
estimator for an exact tie between two different corruption geometries.

Evidence, all from the archived per-clip detections and re-runs on the H100:

1. **The reconstruction is exact.** Re-running the detector on all 150 clips and
   recomputing `softmin(max per-target official-threshold ROI confidence,
   beta=10)` reproduces the archived Repair scores with `mean|delta| = 0.00000`
   (150/150), and the official-label rule reproduces the archived Official
   scores likewise. The 0.00 is not an artefact of the instrument.
2. **The target is never absent to the detector.** Per-level mean confidence for
   the weak target at full severity is still 0.44/0.64/0.55 on the first three
   bases (natural level 0.66/0.70/0.71), and the official 0.5 gate still accepts
   it in 6-15 of 16 frames. Visual inspection of `occlusion_000` versus
   `occlusion_100` confirms the region really is replaced by a flat patch, so the
   residual confidence is carried by the surrounding scene and by the pose, not
   by the object pixels.
3. **No aggregation setting fixes it.** Sweeping the shipped SoftMin over
   `beta ∈ {1, 3, 10, 30, 100}`, `hard_min`, and confidence gates at 0.2-0.5
   changes rank-0 only from 0.70 to 0.80 (test, `control ≤ occlusion_100`) and
   **never produces an exact tie: `rank0(==) = 0.0000` at every setting**.
4. **No configured threshold fixes it either.** `repair_candidate_threshold` is
   part of `MultipleObjectsConfig` and lowering/raising it is a configuration
   change, not a new rule. At 0.0-0.5 the exact-tie rate stays 0.0000; the best
   ordering rate is 0.80 at 0.2, and sensitivity is flat-to-worse (0.7850 →
   0.8050 → 0.7150).
5. **The Official 0.60 is saturation, not fidelity.** Official's 0/1 score lands
   on exactly equal values in 12 of 20 test bases because it quantises to
   sixteenths; the Repair, being continuous, ties in 0. An exact-tie criterion
   therefore rewards the coarser metric by construction.

The plan's requirement for this control is in section 11.4: "No frame contains
both. Both Official and Audit should score this as incomplete." That is a level
predicate plus an ordering against the ladder's worst rung, not an equality
between the two clips. Both backends pass it (`control ≤ occlusion_100` in
18/25 Repair and 22/25 Official bases; Repair control mean 0.2009 against its
clean level 0.5181), so the report's rank-0 CPA is the wrong statistic for the
contract it claims to test — which is why the instrument now reports the
predicate directly (see section 6) instead of burying it under a tie rate.

The refreshed report also carries the frame evidence that makes the diagnosis
checkable from the artefact: on the test split the official-threshold pass finds
both targets in 38.0 % of sampled frames for *both* backends, the Repair's mean
per-frame SoftMin is 0.4074, and the weaker target's mean per-frame confidence is
**0.3510** — the suppressed target is still answered with substantial confidence,
which is the floor the rank-0 tie is measuring.

`transforms.weakest_object_visibility` (`transforms.py:382-419`) alpha-blends
the tracked weaker target towards its per-frame box mean at severity 0/25/50/
75/100, with `expected_rank = round((1-severity)*4)`. The control
(`transforms.temporal_conjunction_control`, `transforms.py:422-455`) suppresses
A in the first half and B in the second, and is stamped `expected_rank=0`
alongside `occlusion_100`.

Plan 11.3 and 11.4 do **not** declare those two equal:

- 11.3: "completeness decreases monotonically as B becomes less visible";
- 11.4: "No frame contains both. Both Official and Audit should score this as
  incomplete."

"Score this as incomplete" is an absolute-level predicate (score low), not a
tie between two different corruption geometries. The rank-0 relation is a
construction decision made in the manifest, and it is a *confounded* one: the
control suppresses A in the first half and B in the second, while
`occlusion_100` suppresses B in every frame, so the two levels differ in **which
frames hold which target** as well as in severity. (The suppression operator
itself is the same: `weakest_object_visibility` at severity 1.0 blends the
region to its mean, which is what `_suppress` writes.) A rank-0 violation
therefore cannot be attributed to the metric alone.

The published means even run *opposite* to the report's "tie" framing, and in
the direction that matters for the plan's control:

| backend | conjunction_control | occlusion_100 | control − occl100 |
|---|---:|---:|---:|
| official | 0.1275 | 0.1825 | −0.055 |
| repair | 0.2009 | 0.2436 | −0.043 |

Both backends score the never-co-present control **more incomplete** than the
fully-suppressed single-object clip. The absolute levels say the control works
(0.13 / 0.20 are low), so the "should score this as incomplete" requirement is
arguably met at both backends, and the tie contract is the wrong statistic to
judge it with. The defensible statement is the level one: *the repair also
scores the never-co-present clip as incomplete (mean 0.20), but ranks it below
full occlusion, so its soft-min response is not zero for a target that the
frame evidence says is gone.* The report should lead with that predicate and
demote the rank-0 CPA to a secondary statistic, as `dynamics_degree` and
`human_action` already do for their invariance families.

Corroborating dispersion (report lines 85-96): the declared-equal subgroup has
Official test within-base CV 0.4922 (relative range 0.9844) and Repair 0.1699
(0.3398) on 20 bases. Neither backend is remotely invariant on that pair, which
is why the 0.00 Repair tie rate is a real failure of a criterion that should not
have been applied in the first place.

## 4. The instrument measures a graded 6-rung ladder as 14 binary pairs

Plan 5.2 defines the unit as the pair `(base, counterfactual)`:

```text
delta_score = score_counterfactual - score_base
```

which for this family is 5 pairs per base (severity 0/25/50/75/100 against
severity 0), exactly the report's rank-1 count. The runner instead expands all
15 level combinations, so 4 of the 5 "sensitivity" pairs per base are
ladder-versus-ladder comparisons (e.g. `occl_000` vs `occl_075`) whose
"strictly higher" status the plan never declared distinguishable. Consequences
visible in the published table:

- rank-1 precision is 0.49 (official) / 0.73 (repair) — the lowest of all gaps,
  i.e. the pair that comes closest to the plan's declared contract is the one
  both backends pass least often;
- all 15 pairs are weighted equally, so rank-gap-4 comparisons carry the same
  weight as adjacent-rung ones, contradicting plan 5.2's "report a
  sequence-level statistic such as Spearman correlation or strict-order rate"
  and 5.3's macro-averaging rule (5 rungs, equal weight per level, not 15 pairs).

Sequence-level statistics are absent. From the published rank-gap precision,
the strict-order (Spearman = 1) rate is **at most** `min(p1..p4)` = 0.49
(official) and 0.73 (repair), and the report should carry the exact figure.
Given that the pooled means are monotone in severity for both backends
(0.5575 > 0.4800 > 0.4625 > 0.4150 > 0.1825 official; 0.5181 > 0.4928 > 0.4777
> 0.4286 > 0.2436 repair), the ordered statistic is likely more favourable than
CPA and its absence understates the Repair result — which is itself a reason to
report it rather than let a reviewer discover the omission.

## 5. Construction and reporting gaps

1. **No intervention verification.** The plan's 11.3 expects occlusion to make
   B "less visible". The audit backend records per-frame target evidence, and
   the report has no such table. That matters here: `conjunction_control`
   suppresses both objects with the same `_suppress` operator used at full
   severity, yet its score has a floor of 0.20 rather than the softmin floor,
   which suggests the detector keeps returning near-threshold detections for a
   flattened patch. Whether severity actually removed target B — and whether it
   destroyed unrelated content, the plan's "manually reject edits that destroy
   unrelated content" clause — is currently unmeasured in the report.
2. **Level profiles pool dev and test** (`score_profile(rows, …)`,
   `run_dimension.py:795-800`) while CPA separates them; the monotonicity claim
   should be split.
3. **The weaker-target choice can be a coin flip.** `_weaker_target`
   (`build.py:356-366`) takes `argmin` of median tracked box areas with no
   margin; near-tie bases are chosen by pixels. The report has no table of
   selected target, area, or area ratio, nor of clips rejected at build time
   (e.g. `_track_boxes` raising on incomplete tracking). The coverage table's
   "expected 150" only proves that bases which *survived* were all scored.
4. **Naming.** `multiplt_object` and `Multiplt_Object.json` are upstream typos
   (commit `8beec1b` "repaired" it to this), while `docs/upstream-mapping.md`
   and the package use `multiple_objects`/`multiple-objects`. One canonical
   name should be chosen and noted, or the row will not map to the dimension in
   the paper.
5. **Difficulty heterogeneity is untested.** The four sensitivity rank gaps
   could be driven by a few easy bases; per-gap paired CIs would show it.

## 6. What the public tables should say

`docs/counterfactual-reports/README.md`, Table 2 row:

| dimension | family | bases | clips | Official CPA | Repair CPA | delta | 95% CI (Repair) |
|---|---|---:|---:|---:|---:|---:|---|
| `multiplt_object` | `weakest_object_visibility` | 25 | 150 | 0.5967 | 0.7633 | +0.1666 | [0.677, 0.830] |

- The Repair CI **overlaps** the Official one ([0.483, 0.697] vs [0.677, 0.830],
  disjoint only above 0.697 — overlapping on [0.677, 0.697]), so the README's
  "**non-overlapping intervals**" sentence does not hold for this row. The
  sensitivity-half difference is strongly significant (167/280 vs 229/280,
  unpaired `+0.267 ± 0.074`, CI [+0.192, +0.341]), so the fix is to quote the
  half-level difference, not to weaken the win.
- The row must carry the split verdict, e.g. *sensitivity +0.27 (Repair win),
  conjunction control not higher than full occlusion in 18/25 bases (Repair) vs
  22/25 (Official); the rank-0 tie-based CPA is not applicable*, or be split into
  two rows. Under the rule the README already applies to invariance families, the
  rank-0 number cannot stand as a "win" without its dispersion (CV 0.49 / 0.17).

The per-dimension report should add, in order:

1. **Level means with detection evidence** (matched/weakest-target confidences
   per level) and an explicit statement that the conjunction control is an
   absolute-level check, not a tie to `occlusion_100`.
2. **Base-vs-base-vs-rung pairs** per plan 5.2, plus a sequence-level strict-order
   rate and Spearman, macro-averaged per plan 5.3.
3. **Dev margins printed**; dev and test statistics separated; the "margin"
   column labelled with the value actually used (0 for both backends).
4. **Build-time rejections and the chosen weak target's area**, so the
   25 × 6 census is auditable.
5. The rank-0 subgroup and the rank-gap table kept as secondary evidence, clearly
   marked as the confounded pair they are.

Implemented in `scripts/counterfactual/run_dimension.py` for this re-score:

- **`## CPA by contract half` now carries a 95% cluster-bootstrap interval per
  half** (`contract_split_cpa(..., iterations, seed)`), because the composite's
  interval says nothing about either half.
- **`### Conjunction control — the plan's actual predicate`** reports, per
  backend, the control mean, the `occlusion_100` mean and the fraction of bases
  where the control is no higher (`contract_predicate_stats`). This is the
  statistic the family's section 11.4 contract asks for; the tie-based CPA is
  kept beside it as a diagnostic and explicitly marked not applicable when the
  declared-equal group is not interchangeable.
- **The rank-gap template no longer calls the rank-0 pairs "the family's actual
  target"** — a sentence that was false for this family and is now conditional on
  the group really being interchangeable.
- `score.py` records Multiple Objects frame evidence (`joint_detection_rate`,
  `mean_frame_score`, `mean_weakest_confidence`, `zero_frame_fraction`) so the
  report can show whether a suppressed target is absent or merely low-confidence
  instead of only reporting the scalar.

## 7. Status of the numbers in this review

- Sections 1-4 are derived from the published report, the committed instrument
  (`scripts/counterfactual/cpa.py`, `run_dimension.py`, `transforms.py`,
  `build.py`) and the plan.
- Section 3's diagnosis required the scoring host: the per-frame detections were
  re-generated for all 150 clips and reconciled against the archived scores
  (`mean|delta| = 0.00000`, both backends), the aggregation/threshold sweeps ran
  on that cache without a GPU, and the two annotated frame strips were rendered
  from the derived clips.
- The 0.5371/0.5964 reconciliation in section 1 is now closed as far as the
  published artefacts go: the refreshed report recomputes both halves from one
  pass and prints them with their own intervals, and the archived
  `multiplt_object__cpa.json` should be replaced by the refreshed one.
- `docs/counterfactual-reports/SUMMARY.md` still names code SHA `66c4a99`, and
  the README's coverage sentence ("complete for every dimension except
  `human_action` 48/60") contradicts SUMMARY's own 60/60 coverage table. Both
  are stale relative to this report's `a660756` and should be refreshed with
  the same edit that fixes this row.

## 8. Scope note: the dataset was not rebuilt

The 0.00 rank-0 result has a second, independent cause on the intervention side:
`transforms.weakest_object_visibility` suppresses the *raw* tracked box, and the
tracked box is both slightly larger than the object and drifting, so at full
severity parts of the target remain and the surrounding scene is untouched. A v2
transform (dilated support, severity-scaled blur-to-mean fill) was prototyped and
measured on the host: at `occlusion_100` the weaker target's mean confidence
falls on some bases (0.66 → 0.44, 0.71 → 0.00) but rises on another
(0.70 → 0.59), so it does **not** reliably create an absent-target condition and
would move the published dataset's clips and all six score files that reference
them. It was therefore **reverted**; the frozen `data/`, `results/`, `splits/`
and `runs/` trees and the published `counterfactual-vbench` clips are unchanged,
and the delivered fix is instrument-side only (the report code at `a660756`).

Recorded honestly because it shapes what a future round should do: if a v3
intervention is attempted, it must (a) rebuild through
`scripts.counterfactual.build` with reusable boxes rather than editing derived
clips, (b) demonstrate in a pre-registered pilot that the absent-target
condition is actually reached (per-target confidence at the top rung, not a
score-level proxy), and (c) re-score every backend, since the comparison in this
family is between clips and not merely between report columns.
