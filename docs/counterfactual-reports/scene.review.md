# Review: `scene` counterfactual report

Scope: `docs/counterfactual-reports/scene.md` (code SHA `bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`)
and the instrument that produced it (`scripts/counterfactual/cpa.py`,
`scripts/counterfactual/run_dimension.py`, `scripts/counterfactual/transforms.py`,
`metrics/scene/src/scene/`).

Verdict: the headline row `scene | environment_coverage | 0.3850 | 0.9300 | +0.5450`
in `docs/counterfactual-reports/README.md`, and the README's claim that this is one of
"three monotone families [that] are genuine Repair wins with non-overlapping
intervals", **must not be quoted as a Repair win**, and the report's own two headline
numbers are mutually inconsistent. The one genuinely informative number in the family
is the **Official** one: the official Scene metric cannot recover the base video it was
given. Everything below is derived from the committed code and the published tables; no
model was re-run. The arithmetic claims are re-derived by
`output/counterfactual/_scratch/scene_report_consistency.py`.

## 1. What the report claims, and what the same tables already show

The contract at issue is not the metric's absolute scale but its response to a
controlled change: the generated clip is a **2x2 composite** whose four quadrants
come from two different videos — the *base* clip (target scene) and a *donor* clip
(wrong scene, same generator, same resolution). Level `coverage_{0,25,50,75,100}`
carries exactly `k` target quadrants (`transforms.py:317-360`), so the expected
relation is monotone increasing in `k`.

The report's own score table (`scene.md:85-96`, 25 clips per level) says:

| backend | `coverage_000` | `coverage_100` | 0 -> 100 gain | within-level std | distinct values |
|---|---:|---:|---:|---:|---:|
| official | 0.0000 | 0.3950 | +0.3950 | 0.2663-0.4049 | 1, 3, 6, 8, 6 |
| repair | 0.3592 | 0.4021 | **+0.0429** | 0.0105-0.0154 | 25 at every level |

Two consequences follow directly, without any re-run:

- **Both backends score the unmodified base video at ≈0.40.** `coverage_100` is
  the base clip: `transforms.environment_coverage` starts from the donor frames
  and overwrites `order[:level]` quadrants with the base, so at `level=4` all four
  quadrants are the base's own pixels. The repaired metric's ceiling on the
  dimension's own reference video is 0.4021, not ~1. That is not a contradiction
  by itself — the repair is a `global x regional-mean` product
  (`metrics/scene/src/scene/backends/audit.py:35-55`) and neither factor is
  calibrated to reach 1 — but it means the absolute scale carries no information
  and the report must not present it as one.
- **The repair's entire dynamic range is 0.043**, against a per-level spread of
  0.011-0.015. A CPA of 0.9300 corresponds to ~14 wrong pairs out of 200, i.e. of
  the order of one adjacent-level inversion per base. The claim "monotone in
  coverage" therefore holds *on average across 20 bases*, not per base; the report
  presents a bootstrap CI but no per-base monotonicity count, so a reader cannot
  tell the two apart.

## 2. The Official number, not the delta, is the finding

The official Scene score is a **coarse quantised statistic**, and the report's
tables are consistent with exactly that mechanism:

- `vbench/scene.py:38-63` (upstream `fd18b3d`) loads each video as
  `num_frames=16` selected by `get_frame_indices(..., sample="middle")`, captions
  all 16 frames with Tag2Text, and sets `video_results = success_frame_count /
  frame_count`. So each clip's official score can only take values `k/16`
  (16 bins); the level mean is an average of those bins. This also explains why
  the level means may sit off the `k/16` grid (e.g. `coverage_025` = 0.1150)
  while no individual clip can.
- The per-frame test is a **string test**, not a similarity score:
  `check_generate` keeps a frame iff every whitespace-separated token of the scene
  label appears in the caption (`vbench/scene.py:29-36`, mirrored by
  `metrics/scene/src/scene/backends/vbench.py:59-62`).
- `coverage_000` returns **exactly 0.0000 with std 0.0000 over 25 clips**, which
  is the correct direction for an insensitivity control: with no target quadrant
  left, Tag2Text never emits the target label. The plan's section 7.4 warning
  ("a constant or insensitive metric must not be considered successful merely
  because it is position invariant") applies to the *inverse* direction here and
  the report should say so explicitly: this backend is insensitive to the
  *presence* of the environment, not merely to its arrangement.

Against that, the decisive observation is that at `coverage_100` the official
metric detects the target scene in only ≈0.395 of frames of the **untouched base
video**. With 16 frames and an underlying rate near 0.4, the binomial standard
error is ≈0.12 per clip, which is of the same order as the whole 0.25 coverage
step; the 16-bin resolution and that noise floor, not the arrangement of
quadrants, are what leave pairwise comparisons at chance (CPA 0.3850, CI
[0.2550, 0.5050]).

**This is the result worth putting in the paper**: the official Scene score is a
frame-level lexical hit-rate whose own reference video scores 0.40, so it cannot
reliably order clips that differ only in environmental support. The repaired
metric's 0.9300 does not add to that claim.

## 3. The +0.5450 is an isomorphism artefact, and the needed ablation is missing

The repair's aggregation is `s_global x mean(s_quadrant)` over four canonical
quadrants. The transformation's level is *defined* as the number of target
quadrants. Quantity to be predicted and quantity that computes the score are the
same quantity, so a high CPA is the design's expected outcome rather than a
finding; the CPA is still a useful *sanity check*, but it cannot be the headline
when the model family and the test family are isomorphic in this way.
Three pieces of evidence in the report support treating the composite as not yet
attributable:

1. **The floor carries the score.** `coverage_000` — zero target quadrants —
   still scores 0.3592. The global factor therefore reports ≈0.36 of scene
   support on a clip that contains none of the target environment (as far as the
   composite can know), i.e. the global term is not an environment detector. The
   distinguishable signal above that floor is the 0.043 gradient.
2. **The ablation that would settle attribution exists in the code and is not in
   the report.** `metrics/scene/src/scene/cli.py` exposes `--mode global` next to
   `environment_grounded`, with a distinct `formula_version`
   (`global-image-text-score` vs `global-times-regional-mean-v1`). Scoring the
   same 125 clips in `global` mode gives the CPA of the global factor alone; run
   it and the reader learns whether the +0.5450 is produced by the regional term
   or by the change of scorer (Tag2Text lexical hit-rate -> CLIP cosine), which is
   a *confound*: the two backends differ in both the aggregation and the model.
   Without this ablation the delta is not attributable to the mechanism the paper
   names.
3. **The margin contributes nothing, and the report does not say so.** Every
   margin in `scene.md:38-45` is 0 and the tie-aware and zero-margin CPAs are
   identical at every split. That is the *good* outcome — the result is not a
   tie-bucket artefact, unlike `subject_consistency`, where the dev margin
   (0.0446) exceeded the whole position spread — but it also means the dev split
   (5 bases, about 8 % of the pairs) constrains nothing. The dev rows should be
   reported as a calibration check, not as corroboration.

## 4. Two defects in the regenerated report

### 4.1 `## CPA by contract half` is a dev+test mixture under a contract label

The section (`scene.md:22-31`) reports `pairs = 250`, `official = 0.3760`,
`repair = 0.9240` under the label `sensitivity`, while the report's primary CPA
table reports 200 test pairs at 0.3850 / 0.9300 and the dev rows at 50 pairs and
0.3400 / 0.9000. The two are the same measurement at different weights:
`250 = 200 + 50`, `0.3760 = (0.3400*50 + 0.3850*200)/250`, and likewise
`0.9240 = (0.9000*50 + 0.9300*200)/250`. The cause is visible in the runner:
`contract_split_cpa` is called with **all** rows

```python
# scripts/counterfactual/run_dimension.py:799
cpa.setdefault("contract_split", {})[backend] = contract_split_cpa(rows, backend_scores, margin)
```

whereas the primary CPA tables use `dev_rows` / `test_rows`. So the section
labelled "contract half" is neither the test split nor a contract half: it is a
dev+test mixture at a different pair count, and the two numbers cannot both be
quoted as the family's CPA. Fix: pass `test_rows`.

### 4.2 The prose around the decomposition is `temporal_relocation`'s, not this family's

`environment_coverage` declares `expected_rank = level`, with five distinct ranks
0-4 and `C(5,2) = 10` ordered pairs per base, all of them rank-gap > 0. There are
no rank-gap-0 pairs at all — confirmed by the report's own table, whose gap
counts are 80/60/40/20 and sum to 200, and by `tied_level_groups`, which returns
no group with more than one level for this family. Nevertheless the section
states:

> This family mixes an inequality half ... with an invariance half (relocated
> variants must tie)

which is copied from the `temporal_relocation` family description and is false
here, and:

> a widening dev margin raises this half without measuring anything

which is vacuous in this family — there is no tie half to widen, and the margin
is 0. The rank-gap table's numbers are correct and useful (Repair match rate
0.90/0.95/0.95/0.95, tie rate 0.0000 at every gap, versus Official 0.2750-0.7000
with tie rates 0.3000-0.6875); only the framing is wrong. The correct reading is
that this family is a pure ordered family, that **all** of its pairs are
sensitivity pairs, and that the composite CPA needs no decomposition at all —
which is a point *in the report's favour* and is currently buried under borrowed
boilerplate. Both structures are emitted from `run_dimension.py:409-427` and
`run_dimension.py:480-507` for any mixed-rank family.

## 5. What the numbers do support

- **Neither backend recognises the base video as itself.** Official 0.3950,
  repair 0.4021 on the untouched target clip. For the repaired metric this is
  consistent with an uncalibrated cosine-to-support map and must be stated as a
  scale caveat; for the official metric it is the substantive finding of section
  2.
- **The official metric is insensitive in the direction that matters.** Exactly
  0.0000 / std 0.0000 at `coverage_000`; a 16-bin hit-rate with ≈0.4 detection on
  the reference video.
- **The method implementation is clean for this family.** 125/125 clips scored on
  both backends, no incomplete shards, no failed clips, so the CPA denominators
  are not affected by coverage asymmetry, and no tie-margin inflation is in play
  (margin 0 at every split). The scene result is methodologically *safer* than
  the `subject_consistency` row, whose composite was a margin artefact.

## 6. Changes required before this row is quoted

1. **Fix the contract-half call** (`run_dimension.py:799` -> `test_rows`) or drop
   the section for single-rank families; delete the "invariance half / relocated
   variants must tie" prose for `environment_coverage`.
2. **Add the `global`-mode ablation** over the same 125 clips and report its CPA
   next to the composite. The paper's claim is about the environment term; an
   ablation is the only thing that attributes the delta to it, and it also
   separates the scorer change (Tag2Text -> CLIP) from the aggregation change.
3. **Report the official score histogram** in `k/16` bins per level (count at
   0.0000, count at 1.0000, distinct bins), plus how many clips of the unmodified
   base video reach a perfect detection rate. This is already recorded in
   `diagnostics.frame_results` / `success_frame_count` /
   `frame_count` (`metrics/scene/src/scene/metric.py:125-150`) and is what turns
   "insensitive" into a quantified diagnosis.
4. **Report per-base monotonicity**, not only the pooled CPA: the number of the
   20 test bases whose five levels are strictly increasing, whose `coverage_100`
   is below `coverage_000`, and the per-base adjacent-level difference. With a
   0.043 range this is the honest resolution of "monotone increasing".
5. **Add a permutation/null control for the CPA**: with a 0.043 range and a 0.93
   CPA, show that shuffled level labels collapse the CPA to ≈0.5. `cpa.py`
   already supports a fixed seed for bootstraps; the null needs no model run.
6. **State the claim boundary** as in the README limitations but sharper: this is
   contract fidelity on a synthetic composite. There is no natural-set human
   agreement evidence for Scene, so nothing here licenses replacing the official
   metric.

If (2) shows the global factor alone already ranks the levels, the honest
headline becomes "the repaired *scorer* detects environment coverage that the
official lexical hit-rate misses", and the regional term is a robustness
property, not the cause of the delta.

## 7. Status of the numbers in this review

- Sections 1 and 2 use only the published tables of `scene.md` and the upstream
  source pinned in `configs/upstream.toml` (`fd18b3d`, scene source hash
  `f14c4237...`); no model was run and nothing was downloaded.
- Section 4.1 is arithmetic cross-checked against the report's own three CPA
  tables; the cause is read from the committed call site.
  `output/counterfactual/_scratch/scene_report_consistency.py` (gitignored, like
  the other review probes) re-derives every arithmetic claim in sections 1, 3 and
  4 from the published tables alone and prints `all checks passed`; it uses
  neither scores nor weights.
- This workspace has no counterfactual scores, derived clips or Scene weights
  (`output/counterfactual/bases.jsonl` holds metadata only, 5 dev + 20 test
  bases), so the corrected tables and the `global`-mode ablation of section 6
  must be produced on the scoring host. `run_dimension.py --report-only` can
  rebuild the report from cached per-clip scores once items 1-4 are implemented.
- Provenance: `scene.md` names code SHA `bdfda5c`, which is an ancestor of the
  revision that archived these reports (`a044ac9`) and not the tree that owns the
  current `run_dimension.py`. The report is therefore not reproducible from the
  revision it names; this should be corrected when the reports are refreshed, as
  already noted for `subject_consistency.review.md`.
