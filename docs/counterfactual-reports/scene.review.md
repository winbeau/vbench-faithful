# Review: `scene` counterfactual report (adversarial re-review)

Scope, this revision:

- file under review: `docs/counterfactual-reports/scene.md`, report code SHA
  `5c4a13091893273510794e45872eac3c59c3ad21` ("make the contract-half, profile and
  invariance tables split-aware"), dataset build SHA `66c4a99`, frozen scores
  `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/` on `h100-server`.
- instrument under review: `scripts/counterfactual/{run_dimension,cpa,build,transforms,score,validate}.py`,
  `metrics/scene/src/scene/`, upstream checkout `fd18b3d` at
  `/root/wenbiao_zhao/VBench`.
- supersedes: the previous `scene.review.md` written against report SHA
  `bdfda5cc`. Section 1 below is the ledger of which of its findings are now
  fixed, downgraded or withdrawn.
- date of this review: 2026-09-15 (host clock, `h100-server`).
- everything here was recomputed from the frozen trees with the interpreter
  `/root/wenbiao_zhao/venvs/vbench/bin/python`. No weight was downloaded, no
  upstream checkout was modified, no GPU was used, and `data/`, `results/`,
  `splits/`, `runs/` and the frozen dataset were never written to. Probes are
  kept in `output/counterfactual/_scratch/scene_audit_probe.py` (the numeric
  re-derivation) and summarised inline here; the staging copy used for the replay
  check lives in `/tmp` on the scoring host and is disposable.

Verdict, in one paragraph. Two of the three generator defects this review raised
last round are **fixed and verified**, and the current report reproduces
byte-identically from the frozen scores. The third (the missing ablation) is
**still missing**, and it can no longer be worked around: the frozen score tree
stores only the composite, and the report's own `Frame evidence` section records
that no per-clip evidence exists. Worse, the headline claim in
`CONSOLIDATED.md` — "the official Scene metric cannot recover the base video it
was given ... Official is globally blind to which video it scored" — **does not
survive recomputation**. Official is not blind; it is *under-determined*. It
commits a sign on 82 of the 200 test pairs and is right on 77 of them (93.9%),
and on those 82 pairs it never once disagrees with the Repair backend in the
wrong direction. The +0.5450 delta is a tie-count effect, not a ranking effect,
and the scene row cannot be quoted in any form that names a Repair improvement.

## 1. Summary of this round's verdicts

| prior claim | status now | evidence |
|---|---|---|
| §4.1 the `## CPA by contract half` table mixed dev+test (`contract_split_cpa(rows)`) | **fixed** | `run_dimension.py:1225-1230`; section is no longer rendered for this family at all (`run_dimension.py:571-582`) |
| §4.2 the decomposition prose was `temporal_relocation`'s template | **fixed** | the whole mixed-contract template is now gated on a shared rank; `scene.md` carries no such text |
| §4.3 `## Score sensitivity` mixed dev+test | **fixed** | `run_dimension.py:1218-1221` + split column at `run_dimension.py:855-865` |
| §1 the report is not reproducible from the revision it names | **downgraded** | it reproduces byte-identically at HEAD; only the header SHA line differs (section 3) |
| §1 "both backends score the base video at ≈0.40, the absolute scale carries no information" | **holds** | 0.3950 / 0.4031 reproduced |
| §1 "the repair's whole dynamic range is 0.043 over a 0.011-0.015 within-level spread" | **holds** | mean per-base `coverage_100 - coverage_000` = +0.0445, one base negative |
| §2 "the official score is a 16-bin frame hit-rate" | **holds** | 0 of 125 clips off the `k/16` grid |
| §3 "the repaired win is an isomorphism artefact and the needed ablation is missing" | **holds, and is now stronger** | section 4 |
| §3 "the official metric cannot recover the base video it was given" | **downgraded to under-determination** | section 5 |
| — | **new defect** | the Official Spearman in `## Sequence-level order statistics` is averaged over 14 of 20 test bases while the table says `bases = 20` (section 7) |
| — | **new defect** | the header of *every* regenerated report states `repair variant: ordered_role_identity_assignment`, which is Spatial Relationship's variant; the scene repair's own variant, scorer and mode are not recorded (section 7.2) |

## 2. Were the two §4 generator defects actually fixed?

Yes. Checked against the code paths that produce the current `scene.md`, not
against the commit message.

**2.1 `contract_split_cpa` is now per split.** `run_dimension.py:1225-1230`:

```python
cpa.setdefault("contract_split", {})[backend] = {
    split: contract_split_cpa(
        split_rows, backend_scores, margin, args.iterations, args.seed
    )
    for split, split_rows in (("dev", dev_rows), ("test", test_rows))
}
```

The old single `contract_split_cpa(rows, ...)` call — the one that produced
`pairs = 250` and the dev+test-weighted `0.3760 / 0.9240` — is gone. The frozen
`scene__cpa.json` confirms the new shape: `contract_split.official.test.sensitivity`
= `{n_bases: 20, n_pairs: 200, cpa: 0.385}` and `.dev.sensitivity` =
`{n_bases: 5, n_pairs: 50, cpa: 0.34}`, disjoint, each with its own bootstrap CI.

**2.2 `score_profile` is now per split.** `run_dimension.py:1218-1221` builds
`cpa["profiles"][backend] = {"dev": score_profile(dev_rows, ...), "test": score_profile(test_rows, ...)}`,
and the renderer at `run_dimension.py:855-865` emits a `split` column. The
published table matches: `official | dev | coverage_025 | 5 | 0.1875 | ...` and
`official | test | coverage_025 | 20 | 0.0969 | ...`. The previous report's
"25 clips per level" rows are gone, so no reader can divide a published mean by
the wrong `n`.

**2.3 The `temporal_relocation` template no longer reaches this family.** The
guard is now explicit at `run_dimension.py:571-582`:

```python
# A family is a *contract mixture* only when it declares more than one rank
# *and* at least one rank is shared by several levels (a tie contract).  A
# pure invariance family has one rank, and a purely ordered family has one
# level per rank; neither is a mixture, so the mixed-contract template does
# not describe them.
rank_groups = tied_level_groups(rows)
dispersion_rows = [(rank, levels) for rank, levels in rank_groups.items() if len(levels) > 1]
mixed_family = len(ranks) > 1 and bool(dispersion_rows)
if mixed_family:
```

`environment_coverage` declares five distinct ranks (`expected_rank = level`), so
`dispersion_rows` is empty and neither `## CPA by contract half` nor the
rank-gap `## Contract decomposition` block renders. This is the right fix, and it
is structural rather than cosmetic: the condition that made the old text wrong is
the condition that now suppresses it. The frozen `contract_split` for this
family independently corroborates the gate — `invariance.n_pairs = 0` on both
dev and test for both backends.

## 3. The report is reproducible, and the SHA line is still ambiguous

Re-running the report path on the scoring host against the frozen trees:

```bash
python -B -m scripts.counterfactual.run_dimension --dimension scene \
  --manifest .../counterfactual-vbench/manifest.jsonl \
  --dataset-root .../counterfactual-vbench \
  --annotations-root .../vbench-1.0-human-preference \
  --upstream /root/wenbiao_zhao/VBench \
  --scores .../counterfactual-vbench/scores --reports /tmp/cf_reports_repro --report-only
```

produces a file that differs from the archived `scene.md` in exactly one line:

```
9c9
< - code SHA: `a4d0403234b765d4a97ca6ee5bbdf72d5fc6c35c`
---
> - code SHA: `5c4a13091893273510794e45872eac3c59c3ad21`
```

So the previous review's "not reproducible from the revision it names" is
**withdrawn**: the report content is determined by the frozen manifest and score
tree, and the report generator at HEAD reproduces it exactly. What remains is a
smaller provenance nit: the header line is the *checkout* revision, so it
conflates "the code that rendered this file" with "the code that scored it". The
scoring code is the manifest's `code_sha = 66c4a99`; the metric packages are not
pinned by either. Since the current run rewrites the line to whatever HEAD is,
a reader cannot tell from the header whether the numbers were recomputed.
Recommend naming both explicitly (`score code` and `report code`) in that block.

## 4. The ablation is still missing, and it is now un-recoverable from the cache

The prior review asked for the `global`-mode ablation
(`metrics/scene/src/scene/cli.py:26,95`, `formula_version = global-image-text-score`)
over the same 125 clips. It still does not exist anywhere:

- the counterfactual scoring driver hardcodes the complete variant —
  `scripts/counterfactual/score.py:341` passes `"environment_grounded"`, and the
  scene branch of that file has no `global` path;
- no report, notebook or score tree on the scoring host contains a scene row with
  `formula_version = global-image-text-score`;

And unlike last round, the ablation can no longer be *approximated* from what was
saved. The repair scores are stored one number per clip:

```json
{"base_id": "scene-v_06dd31dff92009dbc6e7", "derived_id": "...__coverage_000",
 "level": "coverage_000", "score": 0.37292016483843327, "status": "succeeded"}
```

The composite is `s_global x mean(s_quadrant)` (`metrics/scene/src/scene/backends/audit.py:35-55`),
and both factors are discarded. The report says so itself, in a section that did
not exist last round:

> ## Frame evidence
> No per-clip evidence was recorded for this run, so a low repair score cannot be
> attributed to detector drop-outs rather than to a wrong direction. Re-score with
> the current `score.py` to populate it.

That is an honest disclosure, and it is also the admission that the attribution
question cannot be answered from this run: it covers not only detector drop-outs
but the global/regional split and the scorer change (Tag2Text lexical hit-rate ->
CLIP ViT-B/32 cosine-to-support) that separate the two backends.

**Why this now decides the row.** The repair's aggregation is built from the same
quantity the transformation varies — level = number of target quadrants, score =
`global x mean(four quadrant scores)` — so a high CPA is the design's expected
output, not evidence. With the ablation missing, there is no way to show that the
environment term contributes anything the global term would not, and there is no
way to separate "the regional term works" from "CLIP cosine is a better scene
probe than a Tag2Text caption hit-rate on this fixture". Both are live, because
the two backends differ in scorer, query construction, frame handling *and*
aggregation.

Recommendation: the scene dimension may be **re-scored** but must not be
**re-argued** from the cached numbers. If the paper wants a scene row, the run has
to record, per clip, the two factors (`--mode global` and
`--mode environment_grounded` over the same 125 clips), plus the frame evidence
already stubbed in the report.

## 5. "Official cannot recover the base video": recomputed, and the wording is wrong

This was the previous review's headline and `CONSOLIDATED.md` still carries it as
the verdict for this row. Recomputed from the frozen scores, all 125 clips:

- `coverage_100` is the base clip. The manifest makes this exact: the
  transformation starts from donor frames and writes `order[:level]` target
  quadrants, so at level 4 every quadrant is the base's own pixel. The published
  clips are re-encodes, not byte copies (0/25 byte-identical to
  `source_video_sha256`), and the decode check puts the difference where
  re-encoding puts it — decoding `coverage_100` against the upstream source gives
  `max|diff| = 19-29`, `mean|diff| = 1.7-2.4` out of 255 on 4 checked bases,
  i.e. H.264/GIF quantisation noise, not content. `coverage_000` likewise matches
  its donor to `mean|diff| = 1.1-2.4`.
- Official on the base video: **mean 0.3950** over 25 bases, **median 0.1875**,
  **9 of 25 exactly 0.0000**, 16 of 25 below 0.5. On the test split alone:
  mean 0.4344, 6 of 20 exactly 0.
- Official on `coverage_000`: **0.0000 for all 25 clips**, no exceptions.
- Official scores are perfectly quantised: 0 of 125 clips off the `k/16` grid,
  histogram `0/16=91, 1/16=1, 3/16=6, 4/16=1, 6/16=1, 7/16=3, 9/16=1, 10/16=3,
  12/16=2, 13/16=5, 15/16=2, 16/16=9`. That is upstream's own `num_frames=16`
  mean-of-booleans (`vbench/scene.py:38-63`), so the resolution floor is a
  property of the shipped metric, not of this fixture.

Where the claim goes wrong is the word "blind". Decomposing the 200 test pairs at
zero margin:

| backend | decisive | tied | correct | wrong | CPA | accuracy on decisive pairs |
|---|---:|---:|---:|---:|---:|---:|
| official | 82 (41.0%) | **118** | 77 | 5 | 0.3850 | **0.9390** |
| repair | 200 (100%) | 0 | 186 | 14 | 0.9300 | 0.9300 |

and on the 82 pairs Official actually decides, the two backends never disagree in
opposite directions — both right on 77, both wrong on 0, Repair right on 5 where
Official tied (exact McNemar `p = 0.0625`, not significant). Credit exact ties and
Official reads **0.9750** against Repair's 0.9300.

Official's per-rank-gap behaviour is equally un-blind: gap 1 decided 25/80 with
22 correct, gap 2 23/60 with 21, **gap 3 20/20 perfect, gap 4 14/14 perfect**. The
metric is *more* reliable on exactly the comparisons a scene metric is for — large
environmental-support differences — and it refuses to resolve the small ones.

The correct diagnosis is therefore: **the official Scene score is
under-determined at its 16-level resolution.** It abstains on 59% of the declared
ladder, and a strict-sign CPA scores every abstention as a miss, which is what
drives 0.385. It does not "fail to recover the video"; it fails to *separate* the
rungs of a synthetic ladder whose adjacent steps move the underlying caption-level
evidence by less than one frame out of sixteen. A tie is a resolution failure and
a real defect of the metric; it is not evidence of indiscriminate scoring, and the
paper must not write it as such. Note also that the fixture's premise is
adversarial to this particular metric by construction: the ladder holds the
*scene identity* fixed and changes only the fraction of frame area, while the
official metric is a frame-level boolean over a caption model whose vocabulary is
the 86 scene nouns of VBench.

## 6. Is the scene row usable in the paper?

No, not as a Repair win, and the reason is now stronger than isomorphism alone:
the win is a tie-count artifact of the declared statistic.

1. **The delta does not reflect better ranking.** On all 82 pairs where Official
   commits, the backends agree on 77 and neither is ever wrong alone; the entire
   +0.5450 comes from the 118 pairs where Official ties and Repair does not.
2. **It is still an isomorphism.** Level = target-quadrant count, and the repair
   multiplies a global score by the mean of the four quadrant scores. The
   composite responds to the intervention because the intervention is written
   into the composite; the 0.9300 is a sanity check that the pipeline is wired
   correctly.
3. **The ablation that would break the isomorphism is missing** (section 4), and
   cannot be reconstructed from the frozen cache.
4. **The composite is near-blind in absolute terms.** Over all 125 clips the
   repair spans 0.3363–0.4317 (range 0.0955); a clip with zero target quadrants
   averages 0.3586 and the untouched base averages 0.4031. The ordering it gets
   right is a ~0.045 systematic tilt on a 0.34 floor. Official, for scale, spans
   the full 0.0000–1.0000.

What *can* be published from this dimension is the resolution finding of section
5, and only in this form: *the official Scene score is a 16-level frame hit-rate
that assigns identical scores to 59% of the declared coverage ladder and returns
exactly zero on 36% of the base videos it is given*, which is a statement about
the official metric's resolution and detection floor on this fixture. It is a
diagnostic about Official, not a claim about Repair, and it does not by itself
license a replacement.

If the scene row is kept at all, it must appear with the tie rate, the
accuracy-on-decided-pairs, and the base-video floor next to the CPA. If that is
too much caveat for a table row, drop the row and keep the diagnostic in the text.

## 7. Two new reporting defects

### 7.1 The Spearman column averages over the bases it silently drops

`## Sequence-level order statistics` reports, for Official test:

```
| official | test | 20 | 0.7480 | 0.7071 | 0/20 (0.0000) |
```

`bases = 20` is the denominator of `strict_order_rate`, but the correlation is
averaged over **14** bases. `cpa._spearman` returns `None` when every score at
that base is identical (`cpa.py:168`), `order_statistics` skips those bases
(`cpa.py:205-207`), and the returned `n_bases` is incremented *before* the skip
(`cpa.py:202`). For Official test, six bases — `scene-v_4d22ab03c24246af6b2f`,
`scene-v_509c21960fc2a9860ec7`, `scene-v_54655302b83e13405404`,
`scene-v_972d943d32dc1f7d63cd`, `scene-v_dc43ebab070495e2e88a`,
`scene-v_e72e44ea32387e075db2` — produced the same score at all five levels and
therefore contribute nothing to the mean. Dev is worse in relative terms: 3 of 5
bases used for Official, and those are the 3 bases from which the dev margin is
calibrated.

This is not a rounding issue; it is the same failure mode the report's own
insensitivity warning is written to catch, reappearing one level up. The omitted
bases are exactly the ones where the metric had no resolution, so the published
0.7480 is an average over the subset where the metric worked, presented under a
count that includes the subset where it did not. Repair is unaffected
(`n = 20 = bases`), so the column flatters Official relative to Repair and
understates the resolution problem. Fix: report `n_bases` for the correlation and
for the strict-order rate separately, and add the count of
constant-over-levels bases as its own column.

### 7.2 The header mislabels the repair variant, and the scene repair's own configuration is unrecorded

`scene.md:10` claims:

```
- repair variant: `ordered_role_identity_assignment` (detection-conditioned: false)
```

That string is Spatial Relationship's role-assignment ablation, not scene's. The
scoring driver states the scope itself at `score.py:243-244`:

```python
``repair_mode``/``detection_conditioned`` only affect the Spatial Relationship
repair, which is the one dimension whose repair backend exposes scoring
variants (plan section 9.6).  Other dimensions ignore them.
```

Three facts make the line wrong rather than merely generic:

- the renderer emits the line unconditionally for any dimension
  (`run_dimension.py:524-531`);
- the value defaults to the spatial namespace —
  `run_dimension.py:1119-1121` uses
  `os.environ.get("VBENCH_AUDIT_SPATIAL_MODE", "ordered_role_identity_assignment")`,
  and that environment variable is exported by the spatial section of
  `score_orchestrate.sh:39`;
- the scene repair has no such variants at all. Its identity is fixed by
  `score.py:335-341`: scorer from `VBENCH_AUDIT_SCENE_SCORER` (default `openclip`,
  local `ViT-B/32`), mode hardcoded `"environment_grounded"`, recorded in the
  metric CLI as `formula_version = global-times-regional-mean-v1`.

I confirmed the line is identical in all seven regenerated reports
(`scene`, `subject_consistency`, `multiplt_object`, `motion_smoothness`,
`dynamics_degree`, `human_action`, `spatial_relationship`), so this is a
generator-wide regression introduced with the current template, not a scene
problem — but it lands on this row as the only record of what "Repair" means.
It interacts badly with section 4: the one line of provenance that could have
told a reader which scorer and which aggregation produced the column is occupied
by another dimension's configuration. Fix: render the block only when the
dimension actually consumes `repair_mode`, and otherwise record the metric-level
identity (`scorer`, `pretrained`, `mode`/`audit-variant`, `formula_version`),
which `score.py` already knows at the time it writes each row.

## 8. Is 25 bases / 125 clips enough? No, and the family needs a premise gate

**Sample size.** The test split's entire inferential content for Official is 82
decisive pairs spread over 20 bases: median 4 decisive pairs per base, 6 bases with
**zero**, and all 5 of Official's wrong calls concentrated in 3 bases. The
cluster bootstrap over 20 `base_id`s is therefore resampling from an effective
~9 informative clusters, and the paired CI `[+0.425, +0.680]` is correspondingly
optimistic. Repair is better behaved (all 200 pairs decided, 14 wrong spread over
6 bases) but one base supplies 8 of its 14 violations, and its whole effect is a
0.0445 mean tilt.

**The premise.** The family asserts that a 2x2 mosaic — two unrelated scenes
occupying disjoint thirds/halves of the same frame — is ordered by target-scene
area on a scene metric. Two things must be true for that to be a property of a
metric rather than of the fixture: (a) the base video itself must be resolvable as
its labelled scene, and (b) the detector's evidence must vary monotonically with
the area replaced. Neither is checked, and (a) is false for a large minority:
9 of 25 bases score exactly 0.0 on the unmodified base video, 6 of 20 test bases
produce a constant Official profile across all five levels, and Official is at
0.0 for every base at `coverage_000`. A composite metric has no concept of
"environmental support"; it has a caption model, and the family never establishes
that the caption model can see the scene in the first place.

**This is the same class of defect as `spatial_relationship`,** and the fix
already exists as a pattern. `pick_detectable.py:15-24` gates the flip family on
the *audited geometry itself* ("the gate cannot drift from the metric",
`pick_detectable.py:54`), rejecting a base unless the declared relation holds in
enough detected frames. The scene analogue is a
`--scene-min-detection` gate: run the official scene evaluator on the **base**
video and reject any base whose detection rate falls below a pre-registered
threshold, labelling every kept base with its rate, exactly as the relation gate
labels its kept bases. A cheaper minimum, given no re-score budget: report the
per-base base-video detection rate as a column of the `Score sensitivity` table so
a reader can see which bases the ladder could never have resolved. Without either,
a scene ladder violation has two equally good explanations — the metric cannot
rank environment coverage, or the metric cannot see this scene at all — and the
family cannot distinguish them.

Two further construction facts a reader needs: the split is unbalanced across
generators (`modelscope` 8 test / 0 dev, `lavie` 4 test / 0 dev, `cogvideo` 3 dev /
4 test, `videocraft` 2 dev / 4 test), and the dev margin is calibrated on 5 dev
bases of which only 3 carry a defined Official correlation. Both margins are 0,
so nothing in this row is a margin artifact — that part of the previous review
still holds and is the row's one methodological strength.

## 9. Donor search: deterministic, replayable, provenance almost complete

Verified by re-running the committed search, not by reading it.

- `build._pick_donor` (`build.py:408-431`) filters candidate rows to the same
  dimension, generator and split with a different `prompt_id`, then draws the
  donor prompt via `stable_sample(prompt_ids, 1, f"{SEED}:donor:{base_id}")` and
  the donor clip via `stable_sample(pool, 1, f"{SEED}:donorvideo:{base_id}")`.
  Re-running it for all 25 scene bases reproduces the manifest's recorded donor
  for **25/25**, and running it twice gives identical results. One donor is
  shared by two bases (`v_6e9d20252f6927ab823d`); the other 23 bases draw 23
  distinct donors.
- Provenance is complete in every one of the 125 rows: the donor block carries
  `video_uid`, `prompt_id`, `relative_video_path` and `generator`, plus
  `quadrant_order`, `target_quadrants` and `target_fraction` per level. 0 rows are
  missing a field; 0 rows carry a `coverage_000` with target quadrants or a
  non-default quadrant order; every level's `target_quadrants` is exactly the
  prefix of `quadrant_order` implied by its rank, and every `target_fraction` is
  the matching `k/4`.
- **Full replay passes.** Staging the 125 scene clips plus the 49 upstream source
  videos (25 bases + 24 donors) into `/tmp` and running the committed verifier
  (`validate.py`, no `--skip-replay`, no `--replay-sample`) gives
  `{"status": "VALID", "replayed": 125, "mismatches": 0}` in 82 s, with
  `sha256_mismatch: 0` and `properties.violations: 0`. The transformation is
  therefore fully re-derivable from `transformation_parameters` alone, and the
  per-frame detector is not needed for any scene variant.
- **One gap, worth recording rather than fixing retrospectively:** the donor's
  *content* is pinned by `video_uid` + `relative_video_path`, not by a hash. The
  dataset's own `metadata/verification.json` has no `replay` block (it stores
  structure, file hashes, split isolation and properties), and the donor videos
  are not redistributed, so a future re-verification against a changed upstream
  package would fail only indirectly — via a replayed byte mismatch — instead of
  at the provenance check. A `sha256` in the donor block would close it; the
  published dataset cannot be changed retroactively, so this belongs in the
  builder for the next revision.

## 10. Actionable list

Ordered by what unblocks the paper.

1. **Withdraw the "globally blind" wording.** In `CONSOLIDATED.md` (scene row and
   "What can still be claimed") and `README.md`, replace "the official Scene
   metric cannot recover the base video it was given / Official is globally blind"
   with the recomputed finding: Official is *under-determined* — 118 of 200 test
   pairs tie, its CPA of 0.3850 is a tie-count effect, and on the 82 pairs it
   decides it is right 77 times and never once disagrees with Repair in the wrong
   direction.
2. **Fix the Spearman denominator.** `cpa.order_statistics` and the
   `## Sequence-level order statistics` renderer must report the number of bases
   the correlation was actually averaged over, in addition to `bases`, and a
   `constant_profile_bases` count. Today the Official test cell reads `20` for a
   mean over 14. Small change in `cpa.py:191-224` and the renderer; the frozen
   cache can be re-rendered with `--report-only`, no re-score needed.
3. **Fix the `repair variant` line generator-wide.** Gate it on the dimension
   actually consuming `repair_mode` (`score.py:243-244`), and for the other six
   record the metric-level identity instead (scorer, pretrained, audit variant,
   `formula_version`). Today all seven reports name Spatial Relationship's
   ablation. Same `--report-only` re-render.
4. **Decide the row's fate.** Either drop `scene` from Table 2 and keep the
   resolution diagnostic as text, or keep it with the tie rate,
   accuracy-on-decided and base-video floor attached. It must not stay in the
   table as `+0.5450 | yes` under a "Repair wins" reading.
5. **Add a scene premise gate before any re-score.** Follow the relation gate:
   evaluate the base video, keep bases whose scene-detection rate clears a
   pre-registered threshold, and label each kept base with its rate. Report the
   number of candidates rejected. Without this, section 8's confound stands.
6. **Attribution run, if a scene row is wanted at all.** Same 125 clips, two
   configurations — `--mode global` and `--mode environment_grounded` — recording
   `s_global`, the four regional scores, and the frame evidence the report already
   knows how to render. Report the `global`-only CPA beside the composite. If the
   global factor alone already ranks the ladder, the honest headline is about the
   *scorer*, not the aggregation, and the regional term is a robustness property.
7. **Report the tie structure in the report itself.** Add a tie-rate column (or a
   `decided / tied / accuracy-on-decided` block) to the CPA table for every
   ordered family. This is not scene-specific: a strict-sign CPA silently scores
   abstentions as errors, and for a 16-level official metric that is the dominant
   term.
8. **Provenance, next revision:** record a `sha256` for the donor clip in
   `transformation_parameters.donor`, and pin the scoring code as well as the
   report code in the header block (section 3).
9. **Keep the verified parts.** The split fixes, the byte-exact replay, and the
   zero-margin/tie-aware identity are real and should survive the rewrite
   untouched; the previous review's §1 and §3 statements about the 0.043 range,
   the 0.40 base-video scale, the 16-bin grid and the missing ablation remain
   correct as written.

## 11. Status of the numbers in this review

- Sections 1, 2, 5, 6, 7 and 8 were recomputed from
  `…/counterfactual-vbench/manifest.jsonl` and
  `…/scores/scene__{official,repair,cpa}.json{,l}` with an independent script
  (`_scratch/scene_audit_probe.py` and three throwaway probes on the host). The
  CPA, tie, rank-gap, Spearman, strict-order and bootstrap numbers were all
  re-derived without importing `run_dimension.py`, and they match the report
  except where section 7 says they do not.
- Section 3 is an actual re-run of the report path at HEAD on the scoring host;
  the diff is quoted verbatim.
- Section 9 is an actual replay validation of the 125 published scene clips from
  a `/tmp` staging copy, plus a re-run of `build._pick_donor` for all 25 bases.
- The paired delta CI in section 5 (`[+0.4249, +0.6800]`) reproduces
  `CONSOLIDATED.md`'s `[+0.425, +0.680]` under a 2000-resample `base_id` cluster
  bootstrap with seed 2026, matching `cpa.paired_bootstrap_ci`.
- Coverage is complete and clean: 125/125 clips scored on both backends, no
  incomplete shards, no failed clips, so no CPA denominator in this row is
  affected by missing data.
- Not verified here, and still stated as such in the report: model/CUDA/weight
  parity against the frozen E0 baselines. The absolute values in this row are
  first-run measurements.
- Three-way consistency was checked before this review was written: the working
  tree, the Git remote and `/root/wenbiao_zhao/vbench-audit` on `h100-server`
  held byte-identical copies of `scripts/counterfactual/*.py`,
  `configs/counterfactual/bases_published.jsonl` and
  `docs/counterfactual-reports/scene.md` (SHA-256 compared file by file), and the
  published dataset's 25 scene bases are the same 25 `base_id`s as the
  repository's frozen selection. Every command in this review ran from that
  checkout; the report generator was copied to the host with `scp` into `/tmp`
  and never written into the dataset tree.
- After this review was committed, the three copies were re-checked and are at
  one revision: local working tree, `origin/winbeau` and
  `/root/wenbiao_zhao/vbench-audit` all carry this text on top of `3a4586b`,
  which is also the revision at which `scene.md` (`1c7d4152…`),
  `run_dimension.py` (`4ae87492…`), `cpa.py` (`33d7d8f1…`),
  `score.py` (`42f28b3d…`) and `configs/counterfactual/bases_published.jsonl`
  (`29f8d439…`) were compared SHA-256 prefix by prefix across the three trees and
  found identical.
