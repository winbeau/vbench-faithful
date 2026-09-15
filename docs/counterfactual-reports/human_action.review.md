# Review: `human_action` counterfactual report

Scope: `docs/counterfactual-reports/human_action.md` (code SHA
`f54fe67c53b44b39d5874df28632317b69477b52`), the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `scripts/counterfactual/run_dimension.py`,
`scripts/counterfactual/build.py`), the two backends
(`metrics/human-action/src/human_action/backends/`) and the frozen upstream
`vbench/human_action.py` at the pinned SHA `fd18b3d0`.

Verdict: the row `human_action | filename_invariance | 1.0000 | 1.0000 | +0.0000`
in `docs/counterfactual-reports/README.md` is **not** a measurement of the
dimension, and the `Repair is *exactly* filename-invariant (CV 0.0000)` reading
next to it is **vacuous**. For the Official backend the filename *is* the query
label, so the family's invariance expectation is unsatisfiable by construction
and the Official column is a property of the fixture, not of metric stability.
For the Repair backend the target comes from metadata, so all three levels share
one query *and* one byte-identical video: CV 0 is an identity, not a result.
Separately, the report's own `Invariance statistics` table is **empty** because
of a generator regression; the correct numbers (Official CV √2, Repair CV 0) are
derivable from the report's level table and are what `table2.csv` and
`SUMMARY.md` already publish. Unlike `dynamics_degree`, this family's headline
CPA is not *inverted* — it is simply out of scope.

## 1. The Official score is a function of the filename

`official_target_from_filename` (upstream lines 78-79 of
`vbench/human_action.py`, reproduced at
`backends/vbench.py:28-29`) builds the target as

```text
basename.lower().split('-')[0].split('person is ')[-1].split('_')[0]
```

and `official_decision` accepts a clip iff that parsed string is in the rounded
top-5 with `p >= 0.85`. The prompt from the annotation JSON is loaded and
discarded (`docs/upstream-mapping.md`, "Human Action mapping"). At dataset level
the metric is `cor_num / cnt`: the accuracy of *filename label vs. detection*.

The builder knows this and says so — `build.py:144-153` explains that "the
filename *is* the experimental variable here", because naming every variant after
its `base_id` "would make all three parse to the same junk target and the family
would carry no signal". But the family is then published as
`"Invariance: only the filename changes."` (`transforms.py:201-232`,
`publish.py:28-32`, the report header). Those two statements cannot both hold:
a byte-identical copy whose *label* changes is exactly the counterfactual the
Official metric is defined to respond to spot-check. The level table proves the
response is not subtle:

| level | Official mean (25) | reading |
|---|---:|---|
| `filename_correct` | 0.7600 | 19/25 clips have the filename action in the accepted top-5 |
| `filename_wrong` | 0.0000 | 25/25 miss, on the same pixels |
| `filename_neutral` | 0.0000 | 25/25 miss, on the same pixels |

`output/counterfactual/_scratch/ha_review_semantics.py` replays the parser and
the decision rule on the shipped category table. The three levels parse to a
correct K400 action, a different K400 action, and a non-K400 token respectively;
`neutral` and `ha-000__filename_correct` are not in the 400 categories, so those
clips fall through to `target in accepted == False` and score 0 rather than
raising. With one fixed probability vector (ironing 0.99, juggling balls 0.98)
the Official decision flips with the filename alone: target `ironing` →
`matched=True`, target `juggling balls` → `matched=True`, target `washing
dishes` → `matched=False` — the pixels are identical in all three cases.

**Consequence.** `filename_invariance` cannot be a named contract of the
Official backend. Both CPA columns are meaningless as a comparison:

- Official `zero-margin` CPA is exactly the tie rate of the
  [correct, neutral, wrong] triple. The two non-correct levels are identically
  zero, so two of the three pairs per base always tie and the third ties iff the
  correct level is also zero; with `T` bases of the 25 in which it is,
  `CPA = (25 + 2T) / 75`. The published pooled `0.4933` gives `T = 16`, and the
  split CPA `0.5000` (test, 60 pairs) and `0.4667` (dev, 15 pairs) give the same
  `T = 10 / 20` and `3 / 5`. The bootstrap can only return multiples of
  `1/60 = 1.7%` / `1/15 = 6.7%`, so the published `[0.4000, 0.6333]` is exactly
  that grid. The statistic carries the tie rate of one level, not metric
  stability.
- Repair `zero-margin` CPA 1.0000 is an identity: the Repair target is
  `metadata["dimension_metadata"]["human_action"]["target_action"]`
  (`audit.py:16-42`), i.e. the annotation prompt, which the family never moves.

The `## Score sensitivity` block already contains both facts. The report prints
its insensitivity warning only when a backend returns **one** distinct value at
every level; here the Repair returns 25 distinct values that happen to be
identical *within* each base, so the warning does not fire and the table reads as
a pass. It is a pass for a reason that has nothing to do with the metric.

## 2. `Invariance statistics` is empty — a generator regression

The report prints the section header and the column header, then no rows. The
number it should show is in `table2.csv` and `SUMMARY.md` (`official_cv
1.4142135623730954`, `repair_cv 0.0`) and was present in an earlier revision of
this same report: at commit `8dd2b48` the table read

```text
| official | dev  | 3  | 1.4142 | 3.0000 |
| official | test | 14 | 1.4142 | 3.0000 |
| repair   | dev  | 4  | 0.0000 | 0.0000 |
| repair   | test | 16 | 0.0000 | 0.0000 |
```

Root cause, confirmed by replay: commit `bda28f2` changed `main()` to pass
`nested_scores` (keyed `derived_id -> backend -> score`) into `render_report` and
updated the *contract-decomposition* lookup to the nested shape
(`run_dimension.py:531-534`), but left the *invariance* lookup one commit older:

```python
# run_dimension.py:471 (same shape at HEAD and in the working tree)
backend_scores = {r["derived_id"]: (scores or {}).get((r["derived_id"], backend)) ...}
```

`scores` is now nested, so every `scores.get((derived_id, backend))` returns
`None`, i.e. the same value it returns for a genuinely unscored clip;
`invariance_stats` then finds `scale == 0` for every base, returns
`mean_cv = None`, and `render_report` silently `continue`s past every row. The
block was written correctly at `6a823a3` (which passed the tuple-keyed `scores`)
and broke when the call site changed shape without the lookup. The
contract-decomposition lookup degenerates more gracefully (it only skips bases
with fewer than two scored levels), which is why mixed-rank reports such as
`subject_consistency.md` still show 36 populated rows while both same-rank
families show none.

`output/counterfactual/_scratch/ha_review_probe.py` reproduces the empty block
from synthetic rows and shows that the same rows render four populated rows once
the scores are nested. The bug is not specific to this dimension:
`dynamics_degree.md` prints the same empty table, so the `dynamics_degree.review`
section 2.4 reading of "the published dispersion" is currently reading values
from `table2.csv`, not from the report.

The regression survives the test suite because
`tests/test_counterfactual_cpa.py::test_same_rank_family_keeps_the_invariance_block`
asserts only `assertIn("## Invariance statistics", report)` — a header-only
contract. A `| official |` row assertion would have caught it.

## 3. What is actually true, and what the family could measure instead

The dispersion numbers themselves are correct and are still worth publishing.
From the report's own level table the arithmetic is exact, not estimated. Every
base has the vector [1, 0, 0] under Official, so per base

```text
mean |x| = 1/3,  std = sqrt(2)/3  =>  CV  = sqrt(2) = 1.4142
range / mean = 1 / (1/3)          =>  relative range = 3.0000
```

for all 25 bases, which is exactly the published `1.4142 / 3.0000`. Under Repair
the three levels print the same mean, sd, min and max for every level, and
`table2.csv` records the within-base CV as exactly `0.0`; a CV of zero on a
non-constant vector means the three level scores are bit-identical, not merely
close. That is a genuine reproducibility property of the Repair pipeline —
byte-identical input and identical query give identical output — but it is a
property of the fixture, and the family supplies no evidence that the Repair
scalar would move if the *query* changed. Section 4 shows it does, in principle;
the counterfactual set just never asks.

Three confined changes would make the dimension informative:

1. **State the Official column as a fixture property, not a score.** The expected
   Official outcome for a byte-identical family is analytic: 1 iff
   `official_target_from_filename(name)` is in the accepted top-5. Reporting it
   next to, not instead of, the CPA is honest; quoting a delta against the Repair
   is not.
2. **Add the sensitivity half this family lacks.** The `## CPA by contract half`
   block already has a `sensitivity` branch (`run_dimension.py:246-276`) that is
   empty here because all three levels share one rank. A fourth level that keeps
   the correct prompt/content and sets the metadata target to the *wrong* action
   measures the Repair's query response — the mirror of the current invariance
   half. The probe in section 4 shows the instrument would separate it.
3. **Fix the lookup and guard it with a row assertion.** One-line fix
   (`(scores or {}).get(r["derived_id"], {}).get(backend)`) plus a test that the
   same-rank block emits `n_bases > 0`.

## 4. Evidence run for this review

Everything below ran in this workspace; nothing needed GPU or weights.

| probe | what it shows |
|---|---|
| `_scratch/ha_review_probe.py` | the invariance block renders zero rows with `nested_scores` and four rows with tuple-keyed scores, from identical rows |
| `_scratch/ha_review_semantics.py` | the three filename levels parse to a correct, a different, and a non-K400 label; one fixed probability vector is accepted or rejected purely by filename |
| same probe, Repair | `class_evidence` + `aggregate_temporal_evidence` give different scalars for `playing cricket` / `ironing` / `juggling balls` / `washing dishes` on one fixed vector (target probability 0.99 / 0.93 / 0.80 / 0.00), and `parse_action_query` returns the metadata target for all three filename spellings |

Repair-side observation from the same probe, worth carrying into any re-run: the
Repair scalar is the duration-weighted mean of the target probability over up to
four temporal windows (`audit.py:70-137`), whereas the Official score is one
16-frame middle sample thresholded at 0.85. The two are different estimators of
different quantities, so the Repair row's mean of `0.9291` must not be read as a
detection rate comparable to Official's `0.7600`. The report's own
`scalarization` field says the mapping is uncalibrated; the Markdown never
surfaces it.

## 5. Status of the numbers in this review

- Sections 1 and 4 are verified on this machine against the pinned upstream
  `human_action.py`, the shipped `kinetics_400_categories.txt`, and the metric
  packages; no model forward pass was run.
- Section 2 is verified by replaying `render_report` / `invariance_stats` on
  synthetic rows and by reading the commit history; the specific figures
  `1.4142 / 3.0000 / 0.0000` are recomputed analytically from the report's own
  per-level distributions.
- The report's 75/75 coverage cannot be re-verified here: the derived clips,
  manifest and score files live on the scoring host and are not reachable from
  this workspace. Note that `docs/counterfactual-reports/README.md` still says
  "48/60 test clips scored on both backends", which contradicts both this report
  (`none` incomplete shards, 60/60 test, 75/75 total) and `table2.csv`
  (60/60). One of the two must be corrected when the reports are regenerated.
- No fix is included here. Applying the one-line lookup fix and the missing row
  assertion touches `scripts/counterfactual/run_dimension.py` and
  `tests/test_counterfactual_cpa.py`; the two sibling reports for the same-rank
  families would need regenerating afterwards.
