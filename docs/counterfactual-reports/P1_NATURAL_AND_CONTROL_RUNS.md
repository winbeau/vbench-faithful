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

**Result: the v2 repair is significantly worse than Official on the natural
preference set** (−0.115 tie-aware accuracy; the intervals do not overlap).

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

**Status: running.** Both backends are needed because no frozen Official natural
scores exist for this dimension (`results/e0/raw_official_scores/` has only
dynamics, human_action, spatial_relationship, subject_consistency).

Scoring started 2026-09-15 21:33 on GPUs 1–5 via `run_natural.sh
motion_smoothness official|repair`, which now calls `finalize_natural.py` so the
sharded `predictions.csv` overwrite cannot recur.

**Observed rate is the problem**: the Official backend wraps upstream VBench
`MotionSmoothness.motion_score` (AMT interpolation), which costs ~72 s per
1 440-video shard-slot on an H100 already saturated at 80–100% utilisation and
51 GB. That projects to roughly 29 h for the Official pass alone, plus the RAFT
repair pass. The job is durable in tmux `nat2`; the numbers go here when it lands.

Two caveats that hold regardless of when it finishes:

- The pre-existing natural numbers in the status document (Official .636 /
  Repair .395) come from an **older estimator**, so they cannot be compared with
  the direction-aware shipped repair without this rerun.
- The natural-set question is separate from the counterfactual one. The shipped
  repair already moved the counterfactual row from −0.105 to +0.050 (parity); the
  natural run decides whether that came with any human-agreement cost.

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
   (`scripts/counterfactual/pick_detectable.py`) scanned 49 candidate prompts and
   kept 25 for this dimension (24 rejected for undetectable targets). That number
   was printed during construction but is **not durably recorded** — re-running
   `pick_detectable.py` is currently the only way to reproduce it, and it should
   write a summary file next time.

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

## P1.2 Motion Smoothness — natural preference set (continued)

The Official pass finished: **tie-aware pair accuracy 0.6364** [0.6101, 0.6636],
coverage 1.0, 2 160 pairs — reproducing the status document's `.636`. The repair
pass was relaunched at 21:57 after two operational fixes: `results/` is frozen so
the repair joins against the official scores this run produced itself, and the
generated official `results.csv` needed the E0 `status == "success"` convention
rather than the comparator's `succeeded_scalar`.
