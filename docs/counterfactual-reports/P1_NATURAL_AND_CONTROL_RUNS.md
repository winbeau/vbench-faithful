# P1: natural-preference and control runs

Results of the P1 experiment list from `EXPERIMENT_STATUS_AND_ISSUES.md`
(P2 human validation and P3 optional experiments are out of scope this round).
P0 — the generator fixes — is complete and recorded in `CONSOLIDATED.md`.

All runs use the frozen E0 natural preference set (1 440 annotated videos per
dimension, prompt-disjoint dev/test split) and the `base_id`/`pair` clustering of
`scripts/evaluate_pairwise_statistics.py`. Natural-set numbers are pair accuracy
on human labels; they are a *different* question from the counterfactual CPA in
`CONSOLIDATED.md` and must never be merged with it.

## P1.1 Dynamic Degree v2 — natural preference set

**Question**: does the shipped v2 repair (`d/dt**alpha`, alpha = 0.5), the one
that satisfies the FPS-invariance contract, also preserve human-preference
agreement?

Scoring: `scripts/run_official_dataset_compare_dimension.py --backend repair` on
the 1 440 dynamic-degree natural videos, 5 shards over GPUs 1–5, 21:12–21:32
(2026-09-15). `repair_results.jsonl` = 1 440/1 440 `succeeded_scalar`.
Statistics: `scripts/evaluate_pairwise_statistics.py`, 2 000 bootstraps, seed 2026.

| backend | dev acc | test acc (zero margin) | test tie-aware acc | Kendall tau-b | coverage | pairs |
|---|---:|---:|---:|---:|---:|---:|
| Official (frozen E0) | 0.6034 | 0.6845 | **0.6845** [0.6597, 0.7093] | 0.4612 | 1.0 | 2 160 |
| **Repair v2** | 0.3471 | 0.3287 | **0.5690** [0.5426, 0.5953] | 0.2206 | 1.0 | 2 160 |

Paired difference on the same 1 290 test pairs, each backend keeping its own
dev-calibrated margin (`scripts/evaluate_paired_backend_delta.py`, 2 000
resamples, seed 2026):

| dimension | Official | Repair v2 | Delta (paired) | 95% CI | n test |
|---|---:|---:|---:|---|---:|
| dynamic_degree | 0.6845 | 0.5690 | **−0.1155** | [−0.1496, −0.0806] | 1 290 |

**Result: the v2 repair is significantly worse than Official on the natural
preference set** (−0.115 tie-aware accuracy; the paired interval excludes zero).

Read together with the counterfactual result in `CONSOLIDATED.md`
(Official `p = +0.491`, v2 `p = +0.019`), this is the paper's cleanest
audit-versus-replacement contrast:

- v2 **fixes the measurement contract** — the score stops depending on the
  sampling interval;
- v2 **does not improve, and slightly degrades, agreement with human
  preference** on unmodified VBench generations.

That is exactly the "better counterfactual result, worse natural result" cell of
the plan's decision table, whose allowed claim is *useful audit / refinement, not
replacement*. It also reproduces the earlier v1 finding (Official .684 / old
Repair .590) with the current shipped repair, as the status document required.

Note the zero-margin column: the repair's 0.329 is far below its tie-aware 0.569
because a continuous score almost never ties, while a fifth of the human labels
are ties — the dev-calibrated margin recovers most of that. Official's
zero-margin and tie-aware columns coincide because its natural score is already
the Boolean moving decision.

### Operational note

Every shard of a sharded natural run writes the same `<out>/<dim>/predictions.csv`,
so only the last shard's rows survive; `repair_results.jsonl` is appended by all
shards and is complete. `finalize_natural.py` now rebuilds `predictions.csv` from
it and emits the Official `results.csv` needed by the pair statistics. The first
P1.1 attempt reported coverage 0.2 (432/2 160 pairs) purely from this overwrite.

## P1.2 Motion Smoothness — natural preference set

Both backends had to be rerun because no frozen Official natural scores exist for
this dimension (`results/e0/raw_official_scores/` has only dynamics,
human_action, spatial_relationship, subject_consistency).

Scoring: `run_natural.sh motion_smoothness official|repair`, 5 shards over GPUs
1–5, 2026-09-15 21:33–22:08, `finalize_natural.py` merging the shard-overwritten
`predictions.csv` from `repair_results.jsonl`. Coverage: 1 440/1 440 `official`
and 1 440/1 440 `audit` rows, all `succeeded_scalar`. Statistics:
`scripts/evaluate_pairwise_statistics.py`, 2 000 bootstraps, seed 2026.

| backend | dev acc | test acc (zero margin) | test tie-aware acc | Kendall tau-b | coverage | pairs |
|---|---:|---:|---:|---:|---:|---:|
| Official | 0.5506 | 0.5682 | **0.6364** [0.6101, 0.6636] | 0.4137 | 1.0 | 2 160 |
| **Repair (direction-aware)** | 0.3356 | 0.3388 | **0.3248** [0.2992, 0.3512] | 0.0686 | 1.0 | 2 160 |

Paired difference on the same 1 290 test pairs (each backend keeps its own
dev-calibrated margin):

| dimension | Official | Repair | Delta (paired) | 95% CI | n test |
|---|---:|---:|---:|---|---:|
| motion_smoothness | 0.6364 | 0.3248 | **−0.3116** | [−0.3473, −0.2760] | 1 290 |

**Result: the shipped direction-aware repair is not merely worse than Official on
the natural preference set — it is significantly *below chance* (0.3248 against
0.5).** The marginal interval [0.2992, 0.3512] excludes 0.5 and the paired
interval excludes zero, so this is not a precision problem.

Three diagnostics, because a below-chance number invites a sign-bug hypothesis:

1. **Not a plumbing artefact.** The evaluated `predictions.csv` matches the
   `audit` rows of `repair_results.jsonl` value-for-value on all 1 440 videos, and
   the Official path reproduces the status document's `.636` exactly.
2. **The two backends do not agree at all, per video.** Spearman between the
   Official AMT score and the Repair RAFT score on the same 1 440 videos is
   **−0.1032** (Pearson 0.0198) — the repair score is essentially unrelated to the
   official one, and if anything mildly inverted. Official scores occupy
   0.7513–0.9959 (median 0.9649, 1 440 distinct values); Repair scores occupy
   0.3396–0.9552 (median 0.4943, 1 440 distinct), so both are continuous and
   neither is saturating.
3. **The repair is nevertheless direction-correct on injected corruption.** Its
   counterfactual row is parity-to-positive (`temporal_jerk` 0.8300 → 0.8800), so
   the score does rank a corrupted clip below its clean source. The estimator is
   therefore sensitive to the corruption it was built around while remaining
   anti-correlated with human preference on unmodified generations.

Read together with the counterfactual row, this is the strongest instance of the
plan's "audit, not replacement" cell: the shipped motion repair buys counterfactual
contract compliance at the cost of a large, significant loss of human agreement —
a bigger loss than the dynamics v2 repair shows.

Two scope caveats:

- The pre-existing status-document numbers (Official .636 / Repair .395) came from
  an **older estimator**; the rerun above is the comparable measurement, and the
  old Repair .395 was already below chance, so the direction of this finding
  predates the direction-aware change.
- These numbers describe the **committed** repair at `ccbd89b`. The worktree
  currently holds uncommitted motion-smoothness changes (raw-direction default,
  top-k temporal aggregation, 0.5/0.5 magnitude/direction weights) that would
  change this number and are deliberately **not** reflected here.

## P1.4 Multiple Objects — corrected ordered/control statistics

Done. What changed relative to the earlier report:

1. **The ordered ladder is now occlusion-only.** The never-co-present
   `conjunction_control` used to sit at the same rank as `occlusion_100`, so the
   composite CPA counted "control == full occlusion" as a severity tie. All
   rank-based statistics (CPA, rank-gap decomposition, contract halves, order
   statistics, dev margin calibration) now run on the ladder alone; the report
   header names the excluded control level, and the control is still reported by
   its own §11.4 predicate.
2. **Weak-target construction check (new).** Per-base median area of the
   suppressed target as a fraction of the frame, test split:

   | statistic | value |
   |---|---:|
   | bases | 20 |
   | median | **0.1575** |
   | IQR | 0.0350 – 0.3446 |
   | range | **0.0066 – 0.4101** |

   The premise is only sound if suppressing B is visible, and here the box spans
   a **62× range**: the tightest base suppresses 0.66% of the frame (essentially
   nothing) while the widest covers 41%. The severity levels are therefore not
   comparable across bases as constructed, which is a fixture finding of the
   same kind as the Spatial premise failure — the ladder needs a per-base
   minimum-area filter or an area-normalised occlusion.

3. Detection confidence/profile, paired CI, Spearman and strict-order rate were
   already present from the review round and are unchanged.
4. Construction rejection count: the detector-eligibility pass
   (`scripts/counterfactual/pick_detectable.py`) scanned 49 ranked candidates and
   kept 25 for this dimension (24 rejected for undetectable targets;
   `subject_consistency` scanned 25 for 25 kept, so the rejection is specific to
   the two-object dimensions). That number was previously only printed; it is now
   written to a summary file via `pick_detectable.py --summary` and recorded in
   `configs/counterfactual/README.md`.

## P1.3 Dynamic Degree — independent FPS validation of alpha = 0.5

**Question**: 0.5 was taken from a bootstrap CI over the same 40 bases the
counterfactual family uses, so it has never been validated on data that played no
part in choosing it.

Holdout: `build_dynamic_validation.py` ranks every eligible `dynamics_degree`
candidate with the selector's own stable hash, removes the 40 in-use bases, and
takes the next 30. Both the base UIDs and the prompts are disjoint from the
counterfactual set (asserted, not assumed), and both come from the frozen E0
prompt split. The dynamics pool is prompt-limited, so 30 of 32 unused candidates
is the largest honest holdout.

Scoring: Official plus the repair at three fixed exponents — 0 (raw
displacement), 0.5 (shipped), 1 (archived ballistic) — over the 120 holdout
clips. Ratio is the per-base `fps2 / fps8` score ratio; the contract is
invariance, so the target is 1.0.

| method | bases | median ratio | IQR | within ±20% |
|---|---:|---:|---|---:|
| Official | 30 | 1.8474 | [1.253, 3.052] | 0.133 |
| repair alpha = 0 | 30 | 1.9352 | [1.273, 3.037] | 0.133 |
| **repair alpha = 0.5 (shipped)** | 30 | **0.9676** | [0.637, 1.518] | **0.200** |
| repair alpha = 1 (ballistic) | 30 | 0.4838 | [0.318, 0.759] | 0.167 |

**The holdout validates 0.5 at the aggregate level.** It is the only setting
whose median ratio sits near 1.0 (0.968); `alpha = 0` reproduces Official almost
exactly (1.935 vs 1.847 — no time normalisation at all, so the same low-frame-rate
inflation), and `alpha = 1` is the mirror violation (0.484). The ordering of the
three exponents on unseen prompts is exactly the ordering the contract predicts,
which is what the earlier in-sample CI could not establish.

**The per-base dispersion is not fixed, and the holdout says so.** Even at
`alpha = 0.5` only 20% of bases land within ±20% of 1.0 and the IQR still spans
0.64–1.52. So the honest claim is: *0.5 corrects the aggregate sampling-interval
dependence on independent data; individual clips remain dispersed*, matching the
in-sample caveat rather than contradicting it.

### Operational note (natural runs)

Two fixes were needed for a dimension with no frozen official scores, both
landed in the server-side `run_natural.sh` / `finalize_natural.py` helpers:

1. `results/` is frozen and holds no `motion_smoothness` scores, so the repair
   joins against the `official_scores/` tree this run produced itself.
2. The generated official `results.csv` needs the E0 `status == "success"`
   convention rather than the comparator's `succeeded_scalar`, because
   `evaluate_pairwise_statistics.py` reads the official path with the former.

## P1.5 Subject Consistency — equal-area temporal relocation (confirmatory run)

**Question**: `temporal_relocation` places one corruption at the start, middle
and end of a clip and declares the three variants same-rank, so only *when* the
corruption applies may move the score. The original construction tracked the
subject per frame, which made the three placements differ in **area and content
as well as position**, so the review's position finding (Official position
invariance 0.2500 against Repair 0.9167) could not be attributed to position.
The confirmatory construction broadcasts a single per-base **median box** to
every frame (`VBENCH_AUDIT_SUBJECT_BOX=median`, the default since `61a65bc`), so
the three placements differ only in *when*.

Both constructions were rebuilt from the same 25 candidates in the same
`bases.jsonl`; two bases fail GRiT subject tracking (`truck`, `bird`) in either
construction, leaving **23 bases × 4 levels** scored by both backends
(`scripts/counterfactual/score.py`; DINO for Official and Repair). Statistics:
`scripts/counterfactual/subject_position_profile.py`, 10 000 bootstraps,
seed 2026.

### Sensitivity: clean versus corrupt

| construction | backend | clean > start | clean > middle | clean > end |
|---|---|---:|---:|---:|
| equal-area | Official | 1.000 | 1.000 | 1.000 |
| equal-area | Repair | 1.000 | 1.000 | 1.000 |
| tracked | Official | 1.000 | 1.000 | 1.000 |
| tracked | Repair | 1.000 | 1.000 | 1.000 |

Saturated: on all 23 bases both backends rank `clean` above every corrupted
position, in both constructions. The corruption is strong enough that this half
of the contract is not the informative one.

### Position profile

Per base over the three corrupted scores; `relative range` is
`(max − min) / mean`, so it is scale-free across bases.

| construction | backend | max − min median (IQR) | CV median | relative range median (IQR) | within ±5% | ±10% | ±20% |
|---|---|---:|---:|---|---:|---:|---:|
| equal-area | Official | 0.0886 (0.0301–0.1224) | 0.0461 | 0.1017 (0.0357–0.1497) | 0.261 | 0.435 | 0.826 |
| equal-area | Repair | 0.0183 (0.0083–0.0296) | 0.0097 | 0.0215 (0.0089–0.0387) | 0.783 | 0.957 | 1.000 |
| tracked | Official | 0.1043 (0.0477–0.1365) | 0.0566 | 0.1299 (0.0564–0.1745) | 0.217 | 0.348 | 0.826 |
| tracked | Repair | 0.0239 (0.0096–0.0406) | 0.0124 | 0.0296 (0.0109–0.0526) | 0.696 | 0.913 | 1.000 |

### Paired comparison

Paired bootstrap of the median relative range, equal-area − tracked, over the
same 23 bases:

| backend | equal-area | tracked | delta | 95% CI | significant |
|---|---:|---:|---:|---|---|
| Official | 0.1017 | 0.1299 | −0.0282 | [−0.0586, +0.0135] | no |
| Repair | 0.0215 | 0.0296 | −0.0081 | [−0.0213, +0.0132] | no |

### Result

**The equal-area box reduces the position spread but, at 23 bases, does not
establish the reduction.** The median relative range falls by 22% (Official) and
27% (Repair), and the tighter tolerance is where the gain shows: Official's
±10% consistency rises 0.348 → 0.435 and Repair's rises 0.913 → 0.957 (Repair's
±5% rises 0.696 → 0.783). The ±20% rate is unchanged for Official (0.826) and
already saturated for Repair (1.000). Both paired intervals include zero, so the
honest claim is *the confound is reduced, not removed*.

Two further observations that do hold at this sample size:

- **Even with equal-area boxes the Official backend is not position-stable.**
  Its relative range is still 0.1017 and only 26% of bases agree across the three
  placements to within ±5%. Making the three regions identical removed the
  *construction* asymmetry; it did not make the Official estimator
  position-invariant, so part of the original gap was an estimator property
  rather than a fixture artefact.
- **The Repair backend is about five times tighter than Official** in both
  constructions (0.0215 against 0.1017 with equal-area) and reaches 1.000
  consistency within ±20%. Its position invariance is therefore real, not an
  artefact of the box.

The published headline for this family (Official position invariance 0.2500 →
Repair 0.9167) was computed on the **tracked** construction over a *different*
base set — see the provenance note below — so it is not directly comparable with
this same-base A/B; the same-base tracked arm here is the comparable control.

## Provenance note: `bases.jsonl` does not reproduce the published detector-dependent subsets

`output/` is gitignored, so `output/counterfactual/bases.jsonl` is not versioned,
and the published dataset's base sets are not all reconstructible from it.
Comparing the base ids present in the published dataset against the base ids in
the current `bases.jsonl`, per dimension:

| dimension | published | `bases.jsonl` | overlap | match |
|---|---:|---:|---:|:--:|
| dynamics_degree | 40 | 40 | 40 | yes |
| human_action | 25 | 25 | 25 | yes |
| motion_smoothness | 25 | 25 | 25 | yes |
| scene | 25 | 25 | 25 | yes |
| spatial_relationship | 40 | 40 | 40 | yes |
| **multiplt_object** | 25 | 25 | **6** | **no** |
| **subject_consistency** | 25 | 25 | **9** | **no** |

The two mismatching dimensions are exactly the detector-dependent ones, and the
cause is *not* a selector-code change: running `select_bases.py` today and at
`0e4d189` — the commit immediately before `66c4a99` — both reproduce the current
`bases.jsonl` exactly. What `66c4a99` added alongside the ranked pool is
`pick_detectable.py`, which re-selects precisely the detector dimensions by
walking a ranked candidate pool and keeping the first `budget` candidates GRiT
can actually ground. The published subsets are the **oversampled** ones, so the
reconstruction path is `select_bases.py` **then** `pick_detectable.py`, while the
version of `bases.jsonl` now on disk is the plain `select_bases.py` output.

Consequence: the published dataset and the `CONSOLIDATED.md` table remain
internally consistent — every reported score was computed on a published clip —
but `build.py` against the current `bases.jsonl` reproduces only 5 of 7
dimensions.

**Resolved.** Running the two-step pipeline — `select_bases.py`, then
`pick_detectable.py` with no `--dimension` so all three detector-dependent
dimensions are re-selected — reproduces the published clip inventory on **all 7
dimensions, 205/205 bases**, while leaving the other five byte-identical. Because
`output/` is gitignored, the selection of record is now versioned at
`configs/counterfactual/bases_published.jsonl`, and the eligibility counts are
written next to it (`configs/counterfactual/README.md`): `subject_consistency`
scanned 25 ranked candidates for 25 kept (**0 rejected**), `multiplt_object`
scanned 49 for 25 kept (**24 rejected**). That last number is the P1.4 Multiple
Objects fixture finding, which until now existed only in a terminal scrollback;
`pick_detectable.py` gained a `--summary` output so it is durable.
