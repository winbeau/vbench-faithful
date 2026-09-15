# Review: `subject_consistency` counterfactual report

Scope: `docs/counterfactual-reports/subject_consistency.md` (code SHA
`66c4a99dba05aceaebe80276ffbffc607c3d2e40`) and the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `scripts/counterfactual/run_dimension.py`,
`scripts/counterfactual/summarize.py`).

Verdict: the headline comparison in that report is **not** a measurement of the
dimension's contract, so the row `subject_consistency | temporal_relocation |
0.5917 | 0.8500 | +0.2583` in `docs/counterfactual-reports/README.md` must not be
quoted as a Repair win. The report's own level table already contains the
contradiction; nothing below depends on re-running the model.

## 1. Why this family is not an invariance family

`temporal_relocation` (`transforms.py:70-129`) has four levels and two distinct
expected ranks:

| level | `expected_rank` | contract |
|---|---:|---|
| `clean` | 1 | clean > each corrupted clip |
| `corrupt_start` | 0 | the three positions must tie |
| `corrupt_middle` | 0 | ” |
| `corrupt_end` | 0 | ” |

So the family is a conjunction of two different contracts, and
`family_pairs` (`cpa.py:58-72`) expands it into two disjoint halves exactly as
section 5.2 prescribes:

- 3 *sensitivity* pairs per base with expected relation `+1`;
- 3 *invariance* pairs per base with expected relation `0`.

## 2. The reported CPA is half sensitivity, half the invariant half

Per-base pair census, reproduced from the rank rule: 20 test bases × 6 pairs =
120 pairs = **60 sensitivity + 60 invariance**. Reading the published table:

| backend | dev margin | reported test CPA | correct pairs | implied split |
|---|---:|---:|---:|---|
| official | 0.01389 | 0.5917 | 71 / 120 | ≤ 11 / 60 invariance pairs correct |
| repair | 0.04457 | 0.8500 | 102 / 120 | ≤ 42 / 60 invariance pairs correct |

Both backends score the sensitivity half nearly perfectly (their mean clean
gap, 0.101 / 0.075, is many standard errors wide), so the reported number is
dominated by the *easy* half and by how far the dev-calibrated margin happens
to stretch over each backend's position differences. A position-invariant
backend ties all 60 invariance pairs; the report cannot distinguish that from
the observed behaviour because the margin is allowed to absorb it.

`output/counterfactual/_scratch/subject_consistency_cpa_decomposition.py`
demonstrates the failure mode directly with the repository's own `cpa.py`
functions: on synthetic scores where the only thing that changes is the
position bias, the reported tie-aware CPA *rises monotonically* as the
invariance violation grows (0.34 at a perfectly invariant position gap → 0.80 at
a grossly position-dependent one). The instrument is inverted for this family.

## 3. The level table contradicts the "positions tie" contract

From the report itself (test split):

| backend | clean | start | middle | end | clean − worst corrupt | position spread | order |
|---|---:|---:|---:|---:|---:|---:|---|
| official | 0.9022 | 0.7738 | 0.8423 | 0.8533 | +0.0489 | 0.0795 | start < middle < end |
| repair | 0.9135 | 0.8461 | 0.8297 | 0.8387 | +0.0674 | 0.0164 | middle < end < start |

- Sensitivity holds (clean is above every corrupted level for both backends), so
  plan 7.4's `clean > subject-corrupted` requirement is satisfied.
- Position invariance does **not** hold: 0.0795 and 0.0164 mean spreads are both
  larger than the per-level standard error (≈0.07/√25 ≈ 0.014).
- The two backends disagree on the *direction* of the effect: the Official metric
  penalises the corruption most at the start (consistent with its first-frame
  anchor plus adjacent pairs), the repair penalises it least at the start. At
  least one of them violates any consistent position contract, and the report's
  "+0.2583" hides exactly this.
- Neither plan 7.3 statistic — maximum position gap and within-family score
  variance — appears anywhere in the report, and no `Invariance statistics`
  block is emitted for this family because `run_dimension.py:291` only emits it
  when the whole family has a single rank. The invariant *sub-contract* is
  therefore never measured, while its pairs silently dilute the headline.

Also worth flagging: each backend's dev margin (0.0139, 0.0446) is calibrated on
5 dev bases / 30 pairs, and the repair margin is larger than its entire mean
position spread (0.0164) — the tie half of its CPA is a margin artefact by
construction.

## 4. Why the repair did not remove the position dependence

The repair replaces the Official fixed-first-frame anchor with symmetric
all-pairs aggregation (`metrics/subject-consistency/IMPLEMENTATION_REPORT.md`):

```text
Official:  S = 0.5 * mean(adjacent pairs) + 0.5 * mean(first frame vs all frames)
Repair:    S = 0.5 * mean(adjacent pairs) + 0.5 * mean(all unordered pairs)
```

Removing the anchor is a real improvement — it is the term that gives frame 0 a
privileged role — and the position spread does drop from 0.0795 to 0.0164. But
it does not reach zero, and the residual runs the wrong way. Three distinct
causes are visible in the code and the published means.

### 4.1 The `0.5 * local` term is itself position-dependent (structural)

The corruption window is `w = round(0.25 T)` frames wide at every position, so
`T = 16, w = 4` with starts `{0, 6, 12}`. For an uncorrupted pair the score
contributions are `+1`; for a pair touching the window, `+a` with `a < 1`. The
window therefore owns this fraction of the score:

```text
0.5 * (boundary + (w - 1)) / (T - 1) + 0.5 * w * (w - 1) / C(T, 2)
```

| position | crossing adjacent pairs | same-window adjacent pairs | corrupted-pair share of score |
|---|---:|---:|---:|
| start | 1 | 3 | 0.3583 |
| middle | 2 | 3 | 0.3917 |
| end | 1 | 3 | 0.3583 |

The all-pairs term is exactly position-neutral here (66 clean–clean, 48 mixed,
6 window–window pairs at every position), but the local term is not: a window in
the interior has two boundaries and an endpoint window only one. The predicted
ordering is therefore `middle` lowest, `start`/`end` higher. At `a = 0.9` this
boundary term alone buys only `0.0033` of spread, about 20 % of the observed
`0.0164` — real, unavoidable while `0.5 * local` is retained, but second order.
It also predicts the *opposite* sign to the observed `middle < end < start`, so
it cannot be the whole story.

### 4.2 The intervention is not temporally uniform (construction confound)

`transforms._corrupt_region` blurs and hue-rotates whatever GRiT box is tracked
**on each frame** (`transforms.py:157-171`, boxes from `build._track_boxes`), so
the corrupted region's area and content differ between the three windows even
though the blur radius, hue shift and frame count are fixed. A clip whose
subject is small or distant in the opening frames — the usual case, and the
regime where the video starts — receives a geometrically smaller and visually
weaker corruption than the same clip's middle window.

The three levels are therefore **not** "identical corruption, different
position": position is confounded with corruption severity. The `δ_C` term in
`S_rep = 1 - (1 - a) * [0.5 W_local(C) + 0.5 W_global(C)]` is position-dependent
through the window's box geometry and content, and that is the only mechanism
left that can produce the observed `start`-highest ordering.

`output/counterfactual/_scratch/subject_consistency_position_probe.py` tests
this on the scoring host: it reads the recorded `window_boxes` from the
manifest, prints the per-position median corrupted area and the per-base
max/min area ratio, and correlates area against `clean - corrupt_position`. The
probe was validated on synthetic manifests where the answer is known (equal
boxes → ratio 1.000, no position effect; deliberately scaled boxes with an
area-proportional score → ratio 4.04 and `r = +1.000` at every position).

### 4.3 The measurement could not have caught it

Even after 4.1 and 4.2, the report's headline cannot show the failure because
the mixed-rank composite CPA is half sensitivity pairs and half a dev-margin
tie fraction (section 2). The repair's margin (0.0446) is larger than its entire
position spread (0.0164), so widening the tie bucket is rewarded and the
remaining violation is invisible.

## 5. Remedies, in priority order

1. **Make the intervention temporally uniform (fixes 4.2).** Apply the
   corruption to a fixed spatial region for all three positions — e.g. the
   per-clip median tracked box — instead of the per-frame box, and record/report
   the per-base corrupted-area CV so the reader can see that the intervention
   held constant. Without this, a nonzero position gap has two possible readings
   (metric artefact vs. weaker corruption) and the contract cannot be audited.
2. **Align the aggregation with the contract (fixes 4.1).** If the target is
   position invariance, `0.5 * local` is the wrong component: only the local term
   makes position matter for a same-size window. Either drop the local term, or
   define it so that the corrupted frames' influence is equalised across
   positions (e.g. both-directions neighbour pairs with endpoint replication).
   The symmetric all-pairs term alone is exactly position-neutral for a fixed
   window.
3. **Measure the contract, not the mixture (fixes 4.3).** Report sensitivity and
   declared-equal dispersion separately, as in section 5 of the plan-7.3 tables.

Until (2) is done, the honest statement is "the repair reduces but does not
remove temporal-position sensitivity", not "position-invariant".

## 6. What the report should say instead

Three contract-level tables, in this order:

1. **Sensitivity (mixed-rank or ordered families)** — `clean > corrupt_*`:
   fraction of the 60 sensitivity pairs matched, with the cluster bootstrap CI.
   Both backends pass this.
2. **Position invariance (the actual target)** — for each base, the three
   corrupted scores: per-base max−min (plan 7.3), mean/median spread, mean
   within-base CV, and the fraction of bases whose spread exceeds a
   pre-registered tolerance. Ideal: spread ≈ 0. Current values: 0.0795 (official)
   vs 0.0164 (repair), with opposite orderings.
3. **Composite CPA, clearly labelled as a weighted mixture** — report the
   sensitivity share and the invariance tie fraction next to it, or drop the
   composite from the headline table for mixed-rank families and list them
   under the same "dispersion, not CPA" rule that already applies to
   `dynamics_degree` and `human_action`.

The composite may stay in an appendix; it must not be the number in Table 2.

## 7. Status of the numbers in this review

- Everything in sections 2 and 3 is derived from the published report and the
  committed CPA code; no model run was needed.
- This workspace has no counterfactual scores, derived clips or DINO weights for
  the 25 real subject-consistency bases (`output/counterfactual/bases.jsonl`
  holds only the 5 dev + 20 test base metadata), so the corrected tables cannot
  be regenerated here. `run_dimension.py --report-only` can rebuild them from
  the cached per-clip scores on the scoring host once the report emits the
  statistics above.
- The archived report's `code SHA` predates the committed runner: `cpa.py` first
  appears in `7ffa6fa`, after `66c4a99`. The report files are not reproducible
  from the committed revision they name; this should be corrected when they are
  refreshed.
- `output/counterfactual/_scratch/subject_consistency_contract_decomposition.py`
  implements the corrected tables for a scoring host that still has the clips
  and scores; `subject_consistency_position_probe.py` separates cause 4.1 from
  cause 4.2 there.
