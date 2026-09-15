# Review: `spatial_relationship` counterfactual report

Scope: `docs/counterfactual-reports/spatial_relationship.md` at code SHA
`5c4a13091893273510794e45872eac3c59c3ad21` (2026-09-15), its row in
`CONSOLIDATED.md` and its caveat in `README.md`, the instrument that produced them
(`scripts/counterfactual/cpa.py`, `run_dimension.py`, `score.py`, `pick_detectable.py`,
`transforms.py`), the directional gate and its tests
(`tests/test_counterfactual_relation_gate.py`), and the repaired metric
(`metrics/spatial-relationship/src/spatial_relationship/`). Upstream is the locked
checkout `/root/wenbiao_zhao/VBench` at
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` (clean); scores are read from the frozen tree
`/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/`. Measured 2026-09-16 on
`h100-server` on physical GPU 1 — cards 2–6 were held by other users' jobs throughout and
nothing of theirs was signalled or killed.

Verdict: the direction of the previous review stands, but three of its load-bearing claims
were estimates and are now measurements, one of them was wrong, and one code defect is new.
The row `spatial_relationship | directional_flip | 0.3667 | 0.0667 | −0.3000` is **not a
Repair deficit and not a repairable fixture**:

- The Repair number rests on **5 of 30 test bases**. The other 25 are exact ties; of the 5
  that carry any signal, **2 are correct and 3 are inverted**.
- Official **cannot** satisfy this family: `get_position_score` reads only `abs(x_distance)`
  versus `abs(y_distance)` and `check_generate` pools the two roles into unordered pairs, so
  the score is invariant under both mirroring and role swap. The 19 discordant test pairs
  split 11/8 — an exact two-sided sign test gives **p = 0.6476**.
- The fixture **cannot be rebuilt from this source**. Run at the declared thresholds the
  directional gate keeps **6 of 84** candidates and fails the dev budget (3/10); relaxing
  both thresholds to 0.50/0.10 keeps **8 of 84** and still fails dev (3/10). 46 of the 84
  candidates fail a threshold-independent box-presence rule: the named object is never
  detected in *any* frame.

The dimension therefore carries one publishable result — the Official metric cannot
represent the ordered relation it is named for — and no Repair comparison. Sections 7 and 8
give the redesign/delete call and the exact claim boundary.

## 1. What this revision changes, and what this re-review adds

`5c4a1309` regenerated the report with a split-aware contract-half table, a per-half paired
bootstrap interval, sequence-level order statistics and the frame-evidence section. The
headline numbers are unchanged from the archived version, and the consolidation row carries
the same verdict. This re-review does not dispute the verdict; it replaces the estimated
parts of it with measurement:

| new evidence | command | where |
|---|---|---|
| every quoted CPU number re-derived from the frozen tree, with a PASS/FAIL reproduction check | `python -m scripts.counterfactual.verify_spatial_review` | §3, §5.2 |
| frame evidence recomputed by re-scoring the dimension with the shipped code | `... score.py --backend repair` (GPU) | §3 |
| the directional gate actually run, twice, and the published 40 bases probed | `... pick_detectable.py --dimension spatial_relationship` | §4 |
| the best-case repair variant re-scored and compared | `... score.py --repair-mode ordered_role --repair-detection-conditioned` | §3, §7 |

Reproduction control: the re-scored repair run is **80/80 identical** to the frozen tree, so
the frame evidence below is attached to the published numbers rather than to a new
estimator.

## 2. Official cannot represent the ordered relation (Q3)

This is a property of the locked upstream source, not a statistical finding. In
`/root/wenbiao_zhao/VBench/vbench/spatial_relationship.py` at `fd18b3d`:

- `:25` `def get_position_score(locality, obj1, obj2, iou_threshold=0.1)`;
- `:51-52` compute `x_distance = box2_center[0] - box1_center[0]` and the `y` analogue;
- `:67-82` are the four branches, and the **only** uses of those two variables in the whole
  file are `abs(x_distance)` at `:69` and `:71` and `abs(y_distance)` at `:76` and `:78`
  (grepping the file for both names returns six hits: the two definitions at `:51-52` and
  these four uses, so no sign is ever read);
- `:98-115` `check_generate` appends boxes for `object_a` **or** `object_b` to one list
  (`:108`), enumerates unordered pairs `c_obj1 < c_obj2` (`:110-111`) and returns
  `max(cur_score)` (`:114`).

Two consequences follow immediately, and neither depends on any data:

1. Mirroring the frame leaves every term unchanged, so `score(original) > score(flip)` is
   unsatisfiable for Official except through detector noise and re-encoding.
2. Swapping `object_a` and `object_b` also leaves the score unchanged, so plan §9.4's
   role-swap family would be equally vacuous against this metric.

Measured on the frozen tree, pooled over dev+test: level means are `original` 0.3104,
`horizontal_flip` 0.3108, `vertical_flip` 0.3319. Of the 30 test rank-gap-1 pairs, 11 are
tied, 11 favour the original and 8 favour the flip. Treating the 19 discordant pairs as a
coin — which is exactly what a sign-blind statistic must produce — the exact two-sided
binomial p-value is **0.6476**. Official's `0.3667` is that coin, not a measurement.

**Answer to Q3: yes — for Official the family's premise is false by construction, and no
sample size can change it.** The corollary is that the Official half of this row should be
argued from `spatial_relationship.py:67-82` and `:98-115`, with the level means and the sign
test as confirmation, never as "0.3667 is at chance".

## 3. The Repair number is a detector floor (Q2)

Re-scoring the dimension with the shipped code and diffing against the frozen tree gives
80/80 identical scores, so the frame evidence below describes the published numbers. Over
the 40 `original` clips (640 frames):

| `frame_reason` | frames | share |
|---|---:|---:|
| `missing_object` | 278 | 43.4% |
| `missing_subject` | 230 | 35.9% |
| `direction_mismatch` | 72 | 11.2% |
| `axis_mismatch` | 22 | 3.4% |
| `relation_satisfied` | 20 | 3.1% |
| `relation_satisfied_with_iou_penalty` | 18 | 2.8% |

**508/640 = 79.4% of frames never reach the geometry check.** The figure is unchanged from
the previous review and is now verified twice over: it reproduces on the current code, and
it is *invariant to the scoring variant* — the `ordered_role` + detection-conditioned run
produces exactly the same 508/640, because conditioning changes the denominator and
`ordered_role` changes only which pair is scored, not whether an object was found. Over all
80 clips the rate is 1030/1280 = 80.5%.

Of the 132 frames that are resolved, 38 (28.8%) satisfy the relation. Per base, over the 40
`original` clips: **23 have no frame in which both roles are detected**, 12 are partial, and
5 are detected in all 16 frames. Those five are `bicycle on the left of car` (satisfied
16/16) and `giraffe on the right of bird` (16/16), `banana on the top of apple` (1/16),
`clock on the left of vase` (0/16), and `bed on the left of tv` (0/16 under the identity-first
contract, 0.9922 under `ordered_role`, which finds a better instance pair).

On the frozen tree, the Repair scores exactly `0.0` on **70 of 80** clips; only **8 of 40**
bases have any non-zero level and only **5 of 40** a non-zero `original`. In the test split,
**only 5 of 30 bases have non-constant scores at all**, and they split 2 correct / 3
inverted; that is the entire content of the `0.0667`. The sequence statistics say the same
thing from the other side: Repair's median per-base Spearman is **−1.0000** (mean −0.2000,
strict order 2/30) against a declared increasing order, versus Official's +1.0000/+0.1579
and 11/30.

**Answer to Q2: the 79.4% figure is still exact, and the diagnosis survives the change of
code.** The zero floor is the detector, and no scoring variant reaches it: the best-case
variant lifts the mean over the 40 originals from 0.0448 to 0.1396 (12 clips improved, none
worse) and still ranks `vertical_flip` (0.1654) **above** `original` (0.1396).

## 4. The gate, actually run — and the fixture is not constructible (Q1)

The gate has **never** been used to re-select this dimension. The published 40 bases were
produced by plain `select_bases.py`; `configs/counterfactual/README.md` records that spatial
is one of the five dimensions where the plain selection and the published inventory agree,
which is only possible because `pick_detectable.py` did not scan it.

It scans it now, so I ran it (`--relation-oracle detector --relation-min-frames 0.75
--min-co-detected-frames 0.25`, GPU 1, tmux, 2026-09-16 01:18–01:27 CST):

```
spatial_relationship/dev: only 3/10 eligible candidates (oracle=detector, scanned 84);
observed relation rates over 38 scanned candidates: min 0.00 median 0.00 max 1.00
```

| quantity | value |
|---|---:|
| ranked pool scanned | **84** (35 dev, 49 test) |
| rejected | **78** |
| kept | **6** (dev 3, test 3) |
| budget | 10 dev + 30 test → **exit 1** |

Rejection taxonomy, by split:

| reason | dev | test | total |
|---|---:|---:|---:|
| box presence: a named object never detected in any frame | 19 | 27 | **46** |
| co-detection floor: both targets in <25% of frames (4 of them in no frame at all) | 6 | 11 | 17 |
| directional: relation holds in <75% of scored frames | 7 | 8 | 15 |

The gate is not a threshold artefact. Re-running with both thresholds relaxed
(`--relation-min-frames 0.50 --min-co-detected-frames 0.10`) gives **kept 8 (dev 3, test 5),
exit 1**, with the box-presence rejections unchanged at 46 — they cannot respond to either
threshold, because the object is absent from the clip rather than rare in it. A third run
isolates the box-presence rule alone (`--relation-oracle none`): it fills dev (10 bases, after
scanning 18 candidates) but exhausts the whole test pool at **22/30** and exits 1 (scanned 67,
35 rejected, every rejection a named object never detected). The shortage is therefore not
created by the directional gate — the pre-existing box-presence rule already caps this
dimension at 22 test bases. The reported relation-rate distribution over the
candidates that were scored at all is `min 0.00, median 0.00, max 1.00`: the median
candidate that can be measured has the arrangement **wrong in every measurable frame**. Of
the 15 candidates rejected for the relation at the strict thresholds, 13 were at exactly
0.00 and the best was 0.24.

Probing the published 40 bases directly through the same `detector_verdict`
(`verify_spatial_review.py --probe-bases configs/counterfactual/bases_published.jsonl`):

| quantity | value |
|---|---:|
| box-presence-only pass | **20 / 40** |
| full directional gate pass | **3 / 40** |
| bases with ≥1 frame where both roles are detected | **18 / 40** |

Of those 18, **10 have the relation holding in 0.00 of their measurable frames** and 4 in
≥0.80; the distribution is bimodal, not noisy. The source clips are not ambiguous about the
relation — they are mostly confidently wrong about it.

**Answer to Q1: the gate was not used before; run now it yields 6/84, and the dimension is
not constructible from VBench-1.0 human-preference clips at the declared budget.** This is
the fixture's failure, not a scoring failure: 46/84 candidates never ground one named
object, and among those that do, the median has the arrangement inverted.

## 5. Defects found in this revision

### 5.1 The gate breaks the documented reproduction of the published selection

`pick_detectable.py:41` now reads
`DETECTOR_DIMENSIONS = ("subject_consistency", "multiplt_object", "spatial_relationship")`
and the oracle defaults to `detector` (`:281`), with `dimensions` taken from that tuple when
`--dimension` is absent (`:313`). `configs/counterfactual/README.md` documents reproduction
as exactly `select_bases.py` followed by `pick_detectable.py` with no `--dimension`, and
claims **205/205 bases reproduced on all 7 dimensions**. With Spatial Relationship inside the
scanned set that claim is now false: 37 of the 40 published spatial bases are rejected by the
gate (20 of them by box presence alone), so the documented command either aborts on the dev
budget or emits a spatial subset of 6. It is not only the directional gate that breaks it —
with `--relation-oracle none` the box-presence rule alone exhausts the test pool at 22/30 and
the command still exits 1. The published dataset at
`xjuIcthub/counterfactual-vbench` can no longer be regenerated by its own instructions.

Minimal fix, for the maintainer: make the directional scan opt-in — keep
`DETECTOR_DIMENSIONS` at the two box-presence dimensions, add spatial to the scanned set
**only** when `--relation-oracle` is requested explicitly, and default
`--relation-oracle` to `none`. Then add the spatial row to the eligibility table in
`configs/counterfactual/README.md` with these counts (scanned 84, kept 3/3 dev/test or
6 total, rejected 78) so the published selection and the gated selection are both
documented and distinguishable. This review does not change shipped code.

### 5.2 The report's own evidence section is empty

`spatial_relationship.md` ends with "No per-clip evidence was recorded for this run"
(`run_dimension.py:1029`). That is accurate: the frozen tree predates `frame_evidence()`, so
the report cannot support the diagnosis it needs, and this review had to re-score the
dimension to get it. Either re-score the 80 clips with the current `score.py` (~8 minutes on
one H100, verified 80/80 score-identical) or point `run_dimension.py` at a side tree.

### 5.3 The README caveat quotes the right numbers but the wrong headline

`docs/counterfactual-reports/README.md` says the Repair "never reaches the geometry check",
which is correct, and quotes 70/80 and 79.4%, which both reproduce exactly. The sharper
statement is the tie decomposition: 25 of 30 test pairs are exact ties, and the remaining 5
split 2/3. As written, a reader can still take `−0.3000` for a deficit.

## 6. Previous conclusions: confirmed, corrected, withdrawn

| previous claim | status | evidence |
|---|---|---|
| 70/80 Repair clips are exactly 0.0 | **confirmed** | frozen tree, `verify_spatial_review.py` |
| 508/640 = 79.4% of `original` frames have no evidence | **confirmed** | re-score 80/80 identical, then frame reasons |
| 23/40 bases have no frame with both roles detected | **confirmed** | `detected_frame_count` per base |
| 28.8% of resolved frames satisfy the relation | **confirmed** | 38/132 |
| Official is direction-blind in the locked source | **confirmed** | `spatial_relationship.py:51-82`, `:98-115`, fd18b3d |
| "the gate admits 2 of 40" | **corrected → 3 of 40** | probe of the published 40 (the earlier figure came from a partial probe; an intermediate run of my own probe was also wrong, see §10) |
| "the repair has not been shown to get the direction wrong once" | **withdrawn as too generous** | 5 test bases carry signal: 2 correct, 3 inverted; median Spearman −1.0000 |
| the gate enforces plan §9.2 before the family is used | **downgraded** | it exists and works, but it was never run for the published fixture, and running it now rejects that fixture (§4) |
| route B (signed-agreement family) was a viable fallback | **withdrawn** | only 18/40 bases have any measurable frame, so it cannot reach the test budget either |

## 7. Redesign or delete (Q4)

Decision criteria, in order:

1. **Does the paper need a Repair-side spatial claim?** If the audit's thesis is that
   official metrics can be blind to their own contract, no: §2 already yields a complete,
   sample-size-independent finding about Official. If a Repair win is needed on this
   dimension, the fixture has to be rebuilt.
2. **Can the existing fixture be salvaged?** No, and this is measured rather than argued:
   46/84 candidates fail a threshold-independent box-presence rule; relaxing both thresholds
   moves kept from 6 to 8 and leaves dev at 3/10; and the best scoring variant still ranks
   `vertical_flip` above `original`. There is no threshold, variant or aggregation setting
   that recovers a 10/30 split.
3. **Does the source have the property at all?** Only for a minority: 4 of the 18 measurable
   published bases satisfy the relation in ≥80% of frames. A family could be built from those
   four, but a 4-base family cannot carry a CPA claim, and selecting on the measured
   arrangement makes the Repair's success circular (the arrangement would be defined by the
   same detector the metric consumes).

Options with cost:

- **A. Controlled composition from VBench crops (rebuild).** Harvest object crops from frames
  in which the existing detector already grounds them, composite two crops onto a background
  at coordinates chosen by construction, mirror for the flip, and register the result as a
  **new** family (for example `controlled_spatial_arrangement`) rather than as
  `directional_flip`. The premise then holds by construction rather than by hope, the base
  count is unlimited, and plan §9.6's two settings still apply. Cost: ~1 day of engineering
  (harvest + compositor + tests, reusing `grit.py`, `common.encode_video` and the transform
  protocol), minutes to build, and ~25 minutes of GPU for 120 clips × 2 backends (measured:
  80 clips ≈ 8 minutes per backend on one H100). It remains detector-conditioned and must be
  reported as such.
- **B. Delete the row from the Repair comparison.** Keep §2 as the dimension's result and the
  gate as the record of why the fixture was rejected. Cost: ~1 hour of writing, no compute.
- **C. Per-video tracking inside the Repair.** Rejected: it changes the metric's contract
  (plan §9.2 puts temporal tracking out of scope) and it does not fix the premise — the
  arrangement in the source clip is still wrong; tracking only removes the drop-out zeros.

**Recommendation: B now, A only if a Repair-side spatial claim is required.** The measured
evidence is that the source clips, not the metric, are the limiting factor, and the current
row is actively misleading — it is the only row in the table that reports a significant
*negative* delta, and that delta is an artefact of 25 ties.

## 8. The minimal publishable claim (Q5)

**Claim (metric, publishable as it stands).** *VBench 1.0's official Spatial Relationship
score cannot represent the ordered relation it is named for.* Evidence, in order: the
geometry is sign-blind (`vbench/spatial_relationship.py:51-82` at `fd18b3d` — `x_distance`
and `y_distance` appear only inside `abs()`), the pair enumeration discards role order
(`:98-115`, both keys pooled into one list and the max taken over unordered pairs), the
measured level means are equal to within noise (0.3104 / 0.3108 / 0.3319), and the 19
discordant pairs split 11/8 (p = 0.6476). This claim needs no fixture and no Repair.

**Claim (fixture, publishable as a construction finding).** *VBench-1.0 human-preference
clips cannot support a directional counterfactual at the declared budget:* 23 of the 40
published bases have no frame in which both named objects are detected; the gate keeps 6 of
84 candidates at the declared thresholds and 8 of 84 with both relaxed; and among the 18
measurable published bases, 10 have the arrangement wrong in every measurable frame. This
belongs in the dataset card and in the review, not in the results table.

**Explicitly not claimable.** That the repaired metric is worse than Official on this
dimension (−0.3000 is 25 ties out of 30), that the repair is position- or direction-invariant
here (5 informative test bases), and that the family is "unverified" in the ordinary sense —
it is verified and it fails.

## 9. Actionable checklist

1. **Fix the gate default** so the published selection still reproduces (§5.1): scan spatial
   only when `--relation-oracle` is given explicitly, default it to `none`, and add a test
   asserting `pick_detectable` copies the published spatial subset through unchanged.
2. **Record the gate counts** in `configs/counterfactual/README.md` (scanned 84; kept 3 dev /
   3 test at 0.75/0.25; kept 3 dev / 5 test at 0.50/0.10; box-presence-only 10 dev / 22 test
   with the test pool exhausted; 46 box-presence rejections) and note that the published
   spatial subset is *ungated* and unreproducible by the current two-step procedure.
3. **Remove the row from `Table 2` / `CONSOLIDATED.md`'s Repair columns**, or mark it
   `fixture invalid` with the tie decomposition inline (25 tied / 2 correct / 3 inverted) and
   the paired CI struck through. Keep the lineage of the archived number in the note.
4. **Rewrite the dimension's result as §8's two claims**, with `spatial_relationship.py`
   line citations and the sign test.
5. **Re-score the frozen tree with the current `score.py`** (~8 minutes, verified
   score-identical) so the published report stops printing an empty `Frame evidence` section,
   or point `run_dimension.py` at a side tree for the evidence.
6. **Keep the gate and its tests.** It is now the artifact that documents *why* the fixture
   was rejected; deleting it would remove the record. `tests/test_counterfactual_relation_gate.py`
   passes unchanged (13 cases).
7. **Decide A vs B in §7 explicitly**, and if B, close the dimension in `AGENTS.md`'s status
   as "family not constructible from this source; Official blindness reported".

## 10. Status of the numbers in this review

- **Measured on the host.** §3's frame evidence, §4's gate runs and published-base probe, and
  the variant comparison in §3/§7 come from runs on `h100-server` GPU 1 on 2026-09-16, writing
  only under `/root/wenbiao_zhao/spatial-verify/`. Nothing wrote to the frozen score tree,
  the repository, or any other user's directories.
- **Recomputed on CPU from the frozen tree.** Every number quoted in §2, §3 and §6 that does
  not need the detector is re-derived by
  `python -m scripts.counterfactual.verify_spatial_review`, which also re-checks the report's
  `Score sensitivity` profile row by row and the paired interval; it prints
  `PASS: every quoted number reproduces`. Reproduce with:
  `verify_spatial_review.py` (CPU checks), `--evidence-jsonl <re-scored rows>` (frame
  evidence), `--probe-bases configs/counterfactual/bases_published.jsonl` (published-base
  probe, needs the detector).
- **One earlier number in this review series was wrong and is corrected here.** A first
  version of my own probe keyed its detection cache on `id(frame)`; iterating a decoded clip
  yields a fresh numpy view per frame, so ids collided across clips and some per-base
  components were attributed to the wrong video. It reported 6/40 presence and 2/40 gate; the
  corrected probe reports **20/40 and 3/40**. The gate numbers in §4 are unaffected — those
  came from running `pick_detectable.py` itself, with no cache layer — but the correction
  shifts the previous review's "2 of 40" to 3.
- **Not measured.** No new Official run was made (the frozen Official scores are used as-is);
  the reciprocal control of plan §9.3 and the role-swap family of §9.4 remain unbuilt; no
  controlled-composition family (§7 A) was implemented. The variant means in §3 are over the
  40 `original` clips only and are not comparable to the published CPA tables.
- **Provenance.** The row was scored at `a044ac9` and reported at `5c4a1309`; the gate and
  the frame-evidence plumbing postdate both, which is why the gate was never applied to this
  fixture and why the published report's evidence section is empty.
