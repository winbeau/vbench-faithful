# human_action — origin vs ours (paper material)

**Thesis.** VBench 1.0's Human Action score is the **per-video accuracy of a
filename-derived label against a Kinetics-400 top-5 call above a hard 0.85
threshold**. The ground truth is not read from the prompt, from metadata, or from
the annotation file: it is parsed out of the video's own file name
(`human_action.py:78`). The nuisance factor this document names is therefore
**file name**: the score is not a function of the video, it is a function of the
video *plus the string it was saved under*. On byte-identical copies of the same
clip the official metric reads 19/25 when the name is right and 0/25 when it is
wrong or neutral — a 76 % → 0 % collapse attributable to the file name alone.
This is not "instability under renaming": because the file name *is* the query
label, the invariance contract is **unsatisfiable for the official metric by
construction**, so our `filename_invariance` family cannot adjudicate it and the
Repair's CV = 0 on that family is an identity rather than a measurement.

Source of record: `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
`vbench/human_action.py`, 122 lines,
`source_sha256 = 4cee71a1e09e0b2f7e1a8126300c9250be19e3866f49004c8b69b4a5a15e8945`
(`configs/upstream.toml:29-33`), read on the locked scoring checkout. All
`vbench/` line numbers below are that revision. Ours:
`metrics/human-action/src/human_action/` and `scripts/counterfactual/`; frozen
scores `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/`. Every number
in §4 was recomputed on `h100-server` from that tree with
`scripts/counterfactual/verify_human_action_review.py` (CPU only, no GPU, no
re-scoring; the run prints `PASS: every quoted number reproduces`).

---

## 1. Origin — the complete reduction, video → scalar

**Label.** The ground truth is a string operation on the path, not a data lookup
(`:78`):

```text
label(v) = basename(v).lower().split('-')[0].split("person is ")[-1].split('_')[0]   :78
```

Everything after the first hyphen, and everything after the first underscore, is
discarded; the canonical name `a person is <action>-<id>_<n>.mp4` therefore yields
`<action>`. `compute_human_action` loads the prompt list and **throws the prompt
away** (`:115`, `video_list, _ = load_dimension_info(...)`), so no annotation
reaches the comparison.

**Model call.** Per video (`:80-88`):

```text
x        = load_video(v, T, num_frames=16)           :80   # 16 frames, middle-sampled
p        = sigmoid( UMT-ViT-L/16( x ) )  ∈ [0,1]^400 :84   # Kinetics-400, per-class sigmoid
r, i     = topk(p, 5)                                :85   # top-5 only
r        = [round(r_j, 4) for j in 1..5]             :88
```

The backbone is UMT ViT-L/16 fine-tuned on K400 (`:57-71`, weights `:41-56`); our
adapter reproduces the constructor, transform and `num_frames=16`
(`metrics/human-action/src/human_action/models.py:9,9-11`,
`backends/vbench.py:42-48`).

**Threshold and decision.** Only top-5 entries whose rounded probability clears a
hard 0.85 survive (`:89-92`), and the clip scores 1 iff the *label string* is one
of them (`:93-100`):

```text
cat_ls = [ cat_dict[i_j] : r_j >= 0.85, j = 1..5 ]   :89-92
flag(v) = ( label(v) ∈ cat_ls )                       :93-100
```

**Aggregation.** The dataset score is the unweighted mean of that boolean — a
per-video binary accuracy — and under DDP the mean of per-video counts (`:109`,
`:120`):

```text
S(D) = (1/|D|) * sum_{v ∈ D} 1[ label(v) ∈ cat_ls(v) ]      :109, :120
```

Our adapter mirrors this exactly — `official_target_from_filename`
(`backends/vbench.py:28-29`), `official_decision` with
`OFFICIAL_TOP_K`/`OFFICIAL_THRESHOLD` (`:32-49`), and the scoring driver reduces
the result to a float boolean (`scripts/counterfactual/score.py:292`).

## 2. Origin — what breaks, and why the file name is the nuisance factor

### P1 (code fact, `:78`) — the file name is the ground truth

`label(v)` is computed from `v`'s path, and the prompt returned by
`load_dimension_info` is discarded at `:115`. A *code fact*, not an inference:
the metric's target for a clip is fully determined by how the file is named.
Two consequences:

- **Non-invariance.** For a rename `f` that changes only the string, the model
  probabilities are untouched (same decoded frames, `:80-88`) while
  `label(f(v)) ≠ label(v)`, hence in general `flag(f(v)) ≠ flag(v)` and
  `S(f(D)) ≠ S(D)`. The score is not a function of the video.
- **Which renames matter.** `:78`'s parser is sensitive to *irrelevant* name
  structure too: a hyphen or an underscore inserted before the action truncates
  the label, and a case change is absorbed by `.lower()`. Names that are
  semantically identical to a human are not identical to this parser.

### P2 (code fact, `:91`) — the 0.85 gate makes the score piecewise constant

`flag` reads the *rounded* top-5 values through a step function, so a clip whose
true class sits at probability `0.8499` scores 0 and at `0.8501` scores 1. The
gate also discards information the model did provide: a clip answered correctly
at rank 1 with `p = 0.84` is scored identically to a clip where the class is
absent from the top-5. The metric is therefore discontinuous in the model output
and monotone in nothing.

### Formalisation and the key distinction

Write the metric as `S(D) = mean_v g(label(v), cat_ls(v))`, with `g` the indicator
of `:93-100`. The prediction is `S(f(D)) ≠ S(D)` for some rename `f`, and it is
**confirmed** (§4). But that same fact makes the *invariance* contract
unsatisfiable rather than merely violated: an invariance test asks whether the
metric is blind to a nuisance factor, while here the metric's *definition*
consumes that factor as its label. If the transformation acts on exactly the
input the metric uses as ground truth, no implementation of "Human Action as
specified" can pass, and the failure carries no information about implementation
quality — a *family-design* defect, not a metric defect. The distinction governs
how the result may be quoted.

### Testable predictions

1. Renaming a clip from its correct action to a different valid K400 action
   should drop the official score to near 0 — **observed: 0/25** (§4).
2. Renaming to a non-K400 token should also score 0, because `label ∉ cat_ls` by
   construction rather than by exception — **observed: 0/25, no failed clips**.
3. A correct class at rank 5 with `p ≥ 0.85` scores 1 and at `0.8499` scores 0
   (P2); this needs a probability-level probe rather than the counterfactual
   tree, and is the cheapest open experiment for this dimension.

## 3. Ours — the family, and what it can and cannot show

**Construction.** `filename_invariance` (`scripts/counterfactual/transforms.py:201-232`)
emits three variants of one base, all `copy_bytes: True` and all
`expected_rank = 1` (`:212`, `:220`, `:228`) — a **same-rank invariance family**
whose declared contract is "only the file name changes, so the score must not
move". `_build_filename_family` (`scripts/counterfactual/build.py:166-210`) copies
the bytes and asserts `sha256(out) == sha256(source)`; `_filename_stem`
(`build.py:145-162`) makes the three names resolve, under the official parser, to
the base's own action, a different action, and no action. The builder's docstring
says why at `build.py:148-152` ("the filename *is* the experimental variable
here"), and the expected relation is `0` at every level (`build.py:206`).
Recomputed on the frozen tree: 75/75 derived files hash equal to their source,
25/25 bases carry three byte-identical files, containers 60 `.mp4` + 15 `.gif`.
The wrong action is the *same* string (`shooting basketball`) on all 25 bases —
`choose_wrong_action` (`scripts/counterfactual/select_bases.py:112-117`) draws one
stable sample from the action vocabulary — so this level is confounded with that
single action and should be redrawn per base in any re-run.

**What "reading the target from metadata" means.** Our backend takes the target as
an explicit argument rather than parsing it: `parse_action_query`
(`backends/audit.py:16-42`) reads `dimension_metadata.human_action.target_action`
(`:29`, `target_source = "metadata_target_action"` at `:30`), falls back to an
exact K400 prompt only if absent, and rejects a target outside the 400 classes.
The driver fills that field from the *manifest prompt* (`score.py:78-86`), so the
Repair's query comes from the dataset, not from the storage path.

**The input divergence, stated plainly.** Inside this family the two backends do
not see the same task: Official's target is `label(file name)`, Repair's is the
manifest prompt. The family moves the former and holds the latter fixed, so it is
a *counterfactual for Official* and a *no-op for Repair* — verified on the real
clips, where the three names parse to the base's action, to `shooting basketball`,
and to `neutral` (not a K400 class), while Repair returns the same `target =
'squat'` for all three.

**Why Repair's CV = 0 is an identity.** With identical bytes and an identical
query, every input to Repair is identical, so its three scores are bit-identical
by determinism, not by robustness. `run_dimension.py:280-284` computes the
within-base dispersion `std/mean(|score|)`; for Repair it is exactly 0 on 20/20
test and 5/5 dev bases. That establishes *reproducibility* and nothing else, and
it must not be quoted as "Repair is filename-invariant": since no level moves the
Repair query, the run provides no evidence that Repair *responds* to a query at
all — the missing sensitivity half.

## 4. Experiment

All numbers are from the frozen tree (25 bases / 75 clips; dev 5 / test 20),
recomputed by `scripts/counterfactual/verify_human_action_review.py` against
`docs/counterfactual-reports/human_action.md` (code SHA `5c4a1309`).

**The row.** `CONSOLIDATED.md:47` / `README.md:30`: official `1.0000`, repair
`1.0000`, delta `+0.0000`, paired CI `[0.000, 0.000]`, verdict `degenerate`. Both
CPAs are degenerate for the reason in §3: the dev margin saturates (official
`1.0`, repair `0.0`) and every pair expects a tie, so the statistic can only
report the tie rate.

**Invariance statistics** (`human_action.md:45-50`, verified per base):

| backend | split | bases | mean within-base CV | mean relative range |
|---|---|---:|---:|---:|
| official | dev | 4 / 5 | 1.4142 | 3.0000 |
| official | test | 15 / 20 | 1.4142 | 3.0000 |
| repair | dev | 5 / 5 | 0.0000 | 0.0000 |
| repair | test | 20 / 20 | 0.0000 | 0.0000 |

Reading, and the base-count caveat. The official numbers are not a distributional
measurement but an exact consequence of the binary score: every included base has
the vector `[1, 0, 0]`, so `mean|x| = 1/3`, `std = √2/3`, `CV = √2`, and
`range/mean = 3` — identical on all 19 included bases, which is why the dev and
test means coincide to four decimals. The denominators `4/5` and `15/20` are
**not** coverage failures: `invariance_stats` drops any base whose three levels
are all zero (`run_dimension.py:280-284`), which for official is exactly the 6
bases whose correct action missed the top-5 (1 dev + 5 test). Coverage itself is
complete: 75/75 clips on both backends.

**What each number supports or refutes.**

| number (source) | supports / refutes |
|---|---|
| official correct-name mean `0.7500` test, `0.8000` dev, 19/25 pooled (`human_action.md:60-65`) | P1: the correct name is detected on only 76 % of clips — the metric's *positive* rate |
| official wrong/neutral means `0.0000` (`:61-65`) | **P1 confirmed**: on byte-identical bytes, renaming alone takes the score to 0 |
| parser census: correct 25 distinct / 25 in K400; wrong 1 distinct (`shooting basketball`) 25/25 in K400; neutral 1 distinct, 0/25 in K400 | the three levels are a correct label, a *different valid* label, and a non-label — the 0.00 is not an artefact of malformed names |
| official CV `1.4142` vs repair CV `0.0000` (`:47-50`) | quantifies the divergence, but see the identity caveat — not a Repair win |
| repair level means identical to 4 d.p., `distinct = 5/20` per level (`:66-71`) | §3: identical input ⇒ identical output; the 20 distinct *values* are 20 clips, not 20 filename conditions |
| repair `p_target` = the correct action for all three names | §3 input divergence: this family never moves Repair's query |
| paired Δ `+0.5000` `[+0.3667, +0.6000]` zero-margin, `+0.0000` `[0.000, 0.000]` tie-aware (`cpa.json`) | the only non-degenerate contrast is the margin artefact, not a metric difference |

**Can this dimension still test anything? What the run does show.** It cannot
test its declared contract for either backend — unsatisfiable for official (§2),
an identity for Repair (§3). It does produce one solid negative result about the
official metric: *renaming a clip with identical bytes costs it 76 accuracy
points*, and a label outside K400 is scored 0 rather than raising (which is why
coverage here is 100 %, versus the 12/60 failures the pre-`f54fe67` build produced
by naming GIF bytes `.mp4`).

**Keep or delete.** **Keep, as a family-design counterexample and an
unfalsifiable negative result — not as a Repair win, and not as a CPA row.** The
rename result is concrete, reproducible and costs one paragraph; the family is
also the clearest worked example of the rule that a metamorphic family is only
valid when the transformation leaves the metric's *query interface* alone — the
same failure mode that invalidates `scene` and `spatial_relationship`, so the
paper needs it as the generalisable lesson; and deleting it would remove the
evidence while leaving the `degenerate` row unexplained. Two conditions: the CPA
columns must not be quoted, and any repair-side claim must come from a redesign
that moves the Repair query too (a mixed-rank family with a metadata-mismatch
level and per-base wrong actions).

## 5. Consistency with this dimension's review

`docs/counterfactual-reports/human_action.review.md` (commit `049bc87`) reaches
the same conclusions from the same evidence; this document is its paper-facing
condensation. Both record the §2 empty-table regression as **fixed and
re-verified**: `run_dimension.py:756-760` now reads the nested
`{derived_id: {backend: score}}` shape (fix landed in `5c4a130`, whose message
names the old flat lookup as the cause), and the four rows print `4/5`, `15/20`,
`5/5`, `20/20` with CV `1.4142` / `0.0000`. Both agree the fix lacks a regression
test (`tests/test_counterfactual_cpa.py:198-213` asserts only the section header)
and that the dispersion table should label the bases it drops. The review's §3
(the file name is the official label) and §4 (the family tests nothing; keep as a
counterexample) stand; its §5 paired-CI defect — the per-dimension Markdown
prints the zero-margin interval beside the tie-aware delta — remains open, which
is why §4 above quotes both margins.
