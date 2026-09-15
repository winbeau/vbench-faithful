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
