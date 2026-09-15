# Review: `human_action` counterfactual report (adversarial re-review)

Scope: `docs/counterfactual-reports/human_action.md` at code SHA
`5c4a13091893273510794e45872eac3c59c3ad21`, regenerated 2026-09-15 06:04 local;
the tables it feeds (`README.md`, `CONSOLIDATED.md`, `SUMMARY.md`, `table2.csv`,
`table2.json`); the CPA instrument (`scripts/counterfactual/cpa.py`,
`run_dimension.py`, `transforms.py`, `score.py`, `summarize.py`); the metric
packages under `metrics/human-action/src/human_action/`; and the frozen upstream
`vbench/human_action.py` at `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`.
Every statistic in this review was recomputed on `h100-server` from the frozen
score tree `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/` with that
host's interpreter; no model, GPU, or weight was used.

**Verdict: the `human_action` family still tests nothing, and the dimension's row
is still unusable as evidence about the metric.** The construction-level finding
is unchanged and now verified on the real published clips: the Official target
label *is* the filename, and 6 of 25 clips (24%) lose their correct action for no
reason other than the filename they were given. The Repair level comparison is
still an identity. Of the three concrete defects the previous review reported,
two are **fixed and re-verified** (§2.1) and one **stands** (§3); in addition a
**new generator defect** was found, and it also corrupts `dynamics_degree.md` (§5).

| previous review claim | status after this round |
|---|---|
| §2 `Invariance statistics` renders header-only (generator regression) | **FIXED** by `5c4a130`; value recomputed on the host and matches exactly. The fix is still **not protected by a test** (new finding, §2.3). |
| §1 Official target == filename, family unfalsifiable | **STANDS**, re-verified against the current metric package and upstream, and now demonstrated on the 75 published clips rather than a synthetic table (§3). |
| §3 dispersion arithmetic (Official CV √2, relative range 3.0; Repair CV 0) | **STANDS**; recomputed per base from the frozen tree. |
| §5 coverage note (README said 48/60) | **OVERTURNED**: coverage is 150/150 pairs, and the README's 48/60 claim was already corrected. But the dispersion table's own denominator is *not* 25/25 — see §2.2, which the previous review missed. |
| §4 evidence run reproducible | **PARTLY**: the scripts still exist and still run, but they no longer run *where the clips are*, and the previous semantics probe used an invented category list. Superseded by §6. |
| — | **NEW:** the paired-CI column in `## Official vs Repair` prints the *zero-margin* interval next to the *tie-aware* delta, producing an impossible table row (§5). Systemic, affects both invariance families. |
| — | **NEW:** the `filename_wrong` level uses one single action (`shooting basketball`) for all 25 bases, so the family's only discriminating condition is confounded with that action (§7b). |

## 1. What the family is, verified on the published artefacts

25 bases (dev 5 / test 20), 3 levels, 75 clips; coverage 150/150 scored pairs on
the two backends; no failed clips. The derived files live under
`human_action/interventions/filename_invariance/`; 60 carry `.mp4` and 15 carry
`.gif` (the CogVideo sources), which is the `f54fe67` container fix.

Byte-identity is not asserted by the manifest alone — it was recomputed from the
files on disk:

| check | result |
|---|---|
| `output_sha256 == input_sha256` in the manifest | 75/75 |
| `input_sha256 == source_video_sha256` in the manifest | 75/75 |
| per base, the 3 files' recomputed sha256 are equal | 25/25 |

`scripts/counterfactual/validate.py:163-166` is the check that enforces this.
The transformation is therefore exactly what the report header claims: only the
name changes. That is what makes the result below a statement about the metric
rather than about the clips.

## 2. Q1 — the empty-table regression is fixed; two residuals remain

### 2.1 The code path now matches the report

`run_dimension.py:753-767` iterates backend × split, builds the score map with the
nested key, and appends a row whenever `mean_cv` is not `None`:

```python
# run_dimension.py:756-760
# `scores` is the nested {derived_id: {backend: score}} shape here.
backend_scores = {
    r["derived_id"]: (scores or {}).get(r["derived_id"], {}).get(backend)
    for r in split_rows
}
```

`5c4a130` replaced the stale flat `(derived_id, backend)` lookup with this block
and added the `split` column. Recomputing the same statistic from the frozen tree
reproduces the published table exactly:

| backend | split | reported bases / CV / rel. range | recomputed bases / CV / rel. range |
|---|---|---|---|
| official | dev | 4 / 1.4142 / 3.0000 | 4 / 1.4142 / 3.0000 |
| official | test | 15 / 1.4142 / 3.0000 | 15 / 1.4142 / 3.0000 |
| repair | dev | 5 / 0.0000 / 0.0000 | 5 / 0.0000 / 0.0000 |
| repair | test | 20 / 0.0000 / 0.0000 | 20 / 0.0000 / 0.0000 |

The whole `Score sensitivity` table also reproduces field for field (official
dev `0.8000/0.4000`, test `0.7500/0.4330`; repair dev `0.9629/0.0440`, test
`0.9206/0.1644`), as do all six CPA rows and the dev margin (`official 1.0`,
`repair 0.0`).

### 2.2 The dispersion denominator is not the coverage denominator

The user-facing "bases" figure is 5 dev / 20 test (`Score coverage`, header
`bases: 25`), but the dispersion table says 4 / 15 for Official. Both are correct
and neither is a bug, yet the table does not say why they differ.

`invariance_stats` skips any base whose `scale = mean(|score|)` is `<= 0`
(`run_dimension.py:280-284`), because CV is undefined when all three levels score
zero. For Official that is exactly the 6 bases where the correct action did not
make the accepted top-5, so all three levels are 0:

| split | bases in manifest | bases with a nonzero level | skipped (all-zero) |
|---|---:|---:|---|
| dev | 5 | 4 | `human_action-v_22f6e3d31f6189874751` |
| test | 20 | 15 | `…207c8b36…`, `…44600612…`, `…6fab945c…`, `…9465331e…`, `…b4e796ce…` |

On every included base the vector is [1, 0, 0], so CV = √2 and relative range = 3
exactly — which is why the published means are identical to four decimals and why
the skips are harmless to the value. But "bases 4" next to the report's own
"bases 25" invites the reading that one dev base failed to score. It did not: it
scored zero at all three levels. The table should carry the skip count, e.g.
`bases 4 (1 all-zero base excluded)`.

This also corrects an inference available from the previous report revision: the
correct-level success count is 4/5 dev and 15/20 test, i.e. **19/25 = 76%**, and
the CPA values follow from it arithmetically. With `T` bases scoring zero at the
correct level, `CPA_zero(T) = (25 + 2T)/75`, giving `T = 1` and `T = 5`, hence
`0.4667` (dev) and `0.5000` (test) — both reproduced exactly.

### 2.3 The fix has no regression test

`tests/test_counterfactual_cpa.py:198-213` still asserts only
`assertIn("## Invariance statistics", report)`. That assertion passed while the
table was header-only, so it does not protect the fix; reverting the lookup to the
flat shape would recreate the empty table with a green suite. The one-line
strengthening is to use the fixture that already supplies nested scores and assert
a data row (for example `| official | test | 4 |`), not just the header.

## 3. Q2 — the Official target is still the filename, now on real clips

The coupling is unchanged in both places that matter:

- upstream `vbench/human_action.py:78` builds `video_label_ls` from the basename,
  and `:109` is `acc = cor_num / cnt`, i.e. the dataset score is the accuracy of
  *filename label vs. detection*;
- the adapter carries the same parser at
  `metrics/human-action/src/human_action/backends/vbench.py:28-29`, uses it at
  `:71`, and `official_decision` (`:32-49`) accepts a clip iff that parsed string
  is in the rounded top-5 with `p >= 0.85`;
- `scripts/counterfactual/score.py:292` reduces the result to
  `float(bool(result.matched))` for the Official backend;
- the prompt is read and discarded upstream, and `score.py:78-86` only feeds
  `target_action` from the manifest prompt to the **Repair** backend.

Replaying the parser over all 75 published filenames with the real
`kinetics_400_categories.txt`:

| level | distinct parsed targets | parses to a K400 class | note |
|---|---:|---:|---|
| `filename_correct` | 25 | 25/25 | parser recovers the intended action on 25/25 bases |
| `filename_wrong` | **1** | 25/25 | every base says `shooting basketball` |
| `filename_neutral` | 1 | **0/25** | `neutral` is not a Kinetics-400 class |

So the three levels hand one byte-identical clip three different labels: its own
action, a fixed unrelated action, and a token that is not a class at all. Holding
the probabilities fixed, the decision flips with the label alone
(`squat 0.99` / `shooting basketball 0.98` in one vector):

```text
target='squat'                matched=True   accepted=('squat', 'shooting basketball')
target='shooting basketball'  matched=True   accepted=('squat', 'shooting basketball')
target='neutral'              matched=False  accepted=('squat', 'shooting basketball')
```

and on the real data the outcome is 19/25, 0/25, 0/25.

**This is a measurement, not a bug in the fixture.** It is the single most
quotable result the dimension produces: *if you rename the same video, the
Official Human Action score falls from 0.76 to 0.00 while the pixels, duration
and container are bit-identical.* It is also why the family cannot test
invariance: the metric is defined to respond to this input, so "the score must
not move" is unsatisfiable for Official rather than violated by it. The
`filename_neutral` level additionally shows that an unrecognised label is scored
0 through the `target in accepted` comparison rather than raising, which is why
this run has no failed clips where the pre-`f54fe67` revision had 12.

## 4. Q3 — can the family test anything, and should the paper keep it?

**It cannot test the invariance contract it declares, for either backend, and the
Repair column adds nothing.**

- Official: the counterfactual moves the label the metric is defined to compare
  against, so CPA 1.0000 is a fixture property. The zero-margin CPA is exactly
  the tie rate of one level: `(25 + 2T)/75` with `T` the number of bases whose
  correct level also scored 0. The published 0.4667 / 0.5000 and the split
  `T = 1 / 5` reproduce it, and the bootstrap grid can only take values that are
  multiples of 1/60, which the published `[0.4000, 0.6333]` respects.
- Repair: `parse_action_query` takes the target from
  `metadata["dimension_metadata"]["human_action"]["target_action"]`
  (`audit.py:16-42`, source `metadata_target_action`), which this family never
  moves. Three identical videos and one identical query give three bit-identical
  scores, so CV 0 is an identity. Verified in the probe: all three levels return
  `target='squat'`, `p_target=0.9900`, rank 1.

What the run *does* establish, and what it does not:

| claim | supported? |
|---|---|
| Official Human Action is driven by the filename label: 6/25 correct-action clips score 0, and all 25 wrong-name clips score 0 | **yes**, directly measured |
| A non-K400 label scores 0 instead of raising (why coverage is 100%) | **yes** |
| The Repair reads its target from metadata and ignores the filename | **yes** (interface property) |
| The Repair's score is stable / invariant under this transformation | **trivially, because nothing it consumes changed** — no evidence value |
| The Repair detects the action, or discriminates a wrong action | **no** — never tested; the family never varies the Repair query |

**Recommendation: keep it in the paper, but as a negative result and family-design
counterexample, not as a CPA row.** Three reasons to keep: (i) the 0.76 → 0.00
rename result is a concrete, reproducible statement about the official metric that
costs one paragraph; (ii) it is the cleanest illustration of the review's general
point that a metamorphic family is only valid if the transformation leaves the
metric's *query interface* alone, which is the same failure that makes `scene` and
`spatial_relationship` unusable; (iii) deleting it would remove the evidence for
that point while leaving the `degenerate` row in `CONSOLIDATED.md` unexplained.
The row itself must not be read as 1.0000 vs 1.0000: either drop the CPA columns
for this family or print them with the degenerate label attached, as the
`README.md` already does.

## 5. New defect: the paired-CI column is the wrong statistic

The `## Official vs Repair (test, tie-aware)` table prints, for
`human_action`, `repair - official = +0.0000` with a paired 95% CI of
`[+0.3667, +0.6000]` — an interval that cannot contain its own point estimate.
The cause is at `run_dimension.py:895-899`:

```python
paired_zero = cpa.get("paired", {}).get("zero_margin", {})
paired_tie  = cpa.get("paired", {}).get("tie_aware", {})
ci_low  = paired_zero.get("ci_low")      # <-- zero-margin interval
ci_high = paired_zero.get("ci_high")
```

`paired_ci` is the **zero-margin** interval, while the row's delta and both
marginal columns are **tie-aware**. The frozen `human_action__cpa.json` carries
both: `zero_margin {delta 0.5, ci [0.3667, 0.6]}` and
`tie_aware {delta 0.0, ci [0.0, 0.0]}`. An independent cluster bootstrap written
from the documented procedure reproduced both pairs exactly, so the data is right
and only the renderer conflates them.

This is systemic. For the four ordered families the two statistics coincide and
the table is accidentally correct; for both invariance families it breaks
visibly — `dynamics_degree.md` shows delta `−0.0611` with CI `[+0.0000, +0.0000]`.
The correct value is already published correctly elsewhere:
`summarize.py:91-94` reads `paired["tie_aware"]`, which is why `table2.csv` and
`CONSOLIDATED.md` carry `−0.0611 [−0.150, +0.017]`. The per-dimension Markdown is
the only artefact with the wrong number. The fix is one line (select the interval
by the statistic being printed) plus regenerating all seven reports.

## 6. Q5 — status of the old evidence run

The two scratch probes still exist locally
(`output/counterfactual/_scratch/ha_review_probe.py`,
`ha_review_semantics.py`, both gitignored) and the invariance-rendering probe
still reproduces the old behaviour because it exercises `render_report` directly.
The semantics probe, however, was written against a **synthetic** category list
that happened to make `neutral` look like a Kinetics-400 class, and it ran on the
laptop, where the clips are not present. It is superseded: the re-run in this
review used the real `kinetics_400_categories.txt`, the real `human_action`
package at `5c4a130`, and the 75 real files under
`/root/wenbiao_zhao/datasets/counterfactual-vbench/`, executed on `h100-server`
via the host repo `/root/wenbiao_zhao/vbench-audit` (same SHA `a4d0403`). Two
old-review details are corrected by that run: `neutral` is *not* in K400 (so the
neutral level is a label-miss, not a class-miss), and the `filename_wrong` action
is not merely "a different K400 action" — it is the same one everywhere (§7b
below).

The per-clip `Frame evidence` section of the report is accurate: the frozen
`human_action__repair.jsonl` rows carry no `evidence` key at all (0/75), because
the run predates the evidence-persisting `score.py`. Re-scoring would populate
that section but cannot change what the family measures, so it is not worth GPU
time on its own.

## 7. Q4 — what a falsifiable redesign costs

The family fails because the two backends do not share a query interface:
Official's target is the filename, Repair's is the manifest prompt. One
transformation cannot be neutral for both. The redesigns below are ordered by
cost; only (a) makes the dimension say something the current set does not.

**(a) Split the two contracts — the family becomes mixed-rank.** Keep
`filename_correct` / `filename_wrong` / `filename_neutral` exactly as built, and
add three levels with the **same byte-identical video under the same filename**
while the manifest target moves: one carrying a semantically unrelated but valid
K400 action, one carrying an action that is wrong *for this clip*, one carrying
the correct action as a control. Give them `expected_rank = 0` and the existing
three `expected_rank = 1`. Then:

- the Repair is finally falsifiable: it must score `correct > mismatched`, and it
  must tie when the metadata target is unchanged across identical bytes. Its
  current CV 0 survives only as the invariance half, where it is meaningful;
- the Official backend still cannot satisfy either half, which the report can
  state analytically from the parser rather than presenting as a CPA;
- the machinery already exists: `contract_split_cpa`
  (`run_dimension.py:301-333`) assigns `expected == 0` pairs to the invariance half
  and the rest to sensitivity, and `Contract decomposition` renders rank-gap-0 vs
  rank-gap>0 pairs separately. Only `_build_filename_family`
  (`build.py:166-210`), `filename_invariance` (`transforms.py:201-232`),
  `EXPECTED_LEVELS` (`validate.py:64`) and the per-level metadata handoff in
  `score.py:78-86` need to change;
- cost: 3 new levels × 25 bases = 75 new clips (byte copies, seconds to build),
  one re-score of the 150 existing + 75 new clip-backend pairs on 1 GPU, and the
  usual report/table refresh. No new weights, no new detector, no external data.
  This is the cheapest change that turns the dimension from vacuous into
  informative, and it is the option I would take.

**(b) Per-base wrong actions instead of one fixed action.** Whether or not (a) is
adopted, the `filename_wrong` action should be redrawn per base. Today the
"wrong" level is `shooting basketball` for all 25 bases, so the wrong-level
statistic (Official 0.00 and anything the Repair would produce there) is
completely confounded with that one action: it cannot distinguish "the metric
rejects mismatched labels" from "the metric has a blind spot for shooting
basketball". The published dataset carries this as built; changing it means a new
dataset revision (75 regenerated copies + manifest) and a re-score, but the fix
itself is a two-line change in `_filename_stem` / `choose_wrong_action`. Any new
level added under (a) must draw per-base actions.

**(c) Move the filename and the metadata together.** Without (a) this changes
nothing: Official reads only the filename, Repair only the metadata, so moving
both moves each backend's input in lockstep and the Repair half stays an
identity. Listed here only to be dismissed.

**(d) Delete the family.** Not recommended: it discards the 0.76 → 0.00 rename
measurement and the only worked example of an interface-coupled family, and it
would not save anything — the clips exist, the scores exist, and the paper needs
the counterexample more than it needs the row removed.

Whichever option is taken, the two table defects in §2.2 and §5 should be fixed
in the same pass, because both are one-line renderer changes and both are
currently misleading in the published Markdown.

## 8. Executable action list

Ordered; items 1-4 need no GPU, items 5-6 do.

1. **Regenerate the seven per-dimension reports after fixing the paired-CI
   selection** (`run_dimension.py:895-899`): choose the interval matching the
   statistic in the row. Verify each report's `| metric | … | paired 95% CI |`
   row now contains its own delta, and that `dynamics_degree` prints
   `[−0.150, +0.017]` rather than `[+0.0000, +0.0000]`. No re-score is needed:
   `run_dimension.py --report-only` rebuilds from the frozen score files (the
   existing `/root/wenbiao_zhao/run_dim.sh <dim>` wrapper just needs
   `--report-only` appended, and `scripts/counterfactual/regen_dynamics_degree_report.py`
   is the precedent for a score-only regeneration path).
2. **Label the dispersion denominator** (`run_dimension.py:750-767`): report
   `bases N (M all-zero excluded)` so `4 / 15` cannot be read as a coverage
   failure. This is the same block `invariance_stats` feeds at
   `run_dimension.py:280-284`. Expected on this dataset: official dev `4 (1)`,
   official test `15 (5)`, repair `5 (0)` / `20 (0)`.
3. **Add the row assertion** to
   `tests/test_counterfactual_cpa.py::test_same_rank_family_keeps_the_invariance_block`
   so the empty-table regression cannot return (assert a data row, not the
   header).
4. **Refresh the derived tables** (`SUMMARY.md`, `table2.csv`, `table2.json`,
   `CONSOLIDATED.md`, `README.md`) only if item 1 changes any number; the
   `human_action` row itself (`1.0000 / 1.0000 / +0.0000`, CI `[0.000, 0.000]`,
   `degenerate`) is correct and already comes from `summarize.py`.
5. **Redesign the family as a mixed-rank contract** (option 7a), with per-base
   wrong actions (option 7b): implement the new levels and metadata handoff, then
   re-score on 1 GPU. Declare in the report which half the Repair satisfies —
   a Repair that does *not* drop on a mismatched metadata target is a real finding,
   not a failure of the run.
6. **If item 5 is deferred**, drop the CPA columns for this dimension from the
   per-dimension report (or stamp them `degenerate`) and quote only the rename
   result `19/25 → 0/25` in the paper text.

## Provenance of the numbers in this review

| claim | source |
|---|---|
| coverage 150/150, per-level means, dispersion CV/range, all CPA rows, dev margins | recomputed from `scores/human_action__{official,repair}.jsonl` + `manifest.jsonl` on `h100-server`, replicating `cpa.py` / `invariance_stats` |
| skip lists and skipped-base counts | same recomputation, printing the `scale <= 0` bases |
| paired deltas and CIs (both margins) | `scores/human_action__cpa.json` plus an independent cluster bootstrap written from the documented procedure |
| filename parsing, K400 membership, decision flip | replayed on `h100-server` against `kinetics_400_categories.txt` and the 75 real files, with `metrics/human-action` at `5c4a130` |
| byte-identity of the 75 derived files | sha256 recomputed over the files on disk; `validate.py:163-166` is the in-repo check |
| dataset reproducibility of the bases | the 25 dataset `human_action` base ids are identical to `configs/counterfactual/bases_published.jsonl` (25/25) |
| three-way consistency | local repo, host repo `/root/wenbiao_zhao/vbench-audit`, and `<dataset>/reports/` all carry `human_action.md` md5 `8beb491336dbfd68f63a2ee1ad49853e` at SHA `a4d0403` |

Nothing here needed a model forward pass, a weight download, or an upstream
modification, and no file under `data/`, `results/`, `splits/` or `runs/` was
touched.
