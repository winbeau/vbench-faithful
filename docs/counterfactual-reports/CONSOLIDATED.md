# VBench-CF: consolidated status

Everything established so far about the counterfactual (metamorphic) audit of
VBench 1.0: what was built, what was scored, and what each result is actually
worth after independent review.

- Dataset: `counterfactual-vbench`, 205 bases / 815 derived clips, 7 dimensions,
  published at `xjuIcthub/counterfactual-vbench`.
- Scoring: Official VBench 1.0 vs the repaired metric, 6 physical GPUs in
  parallel, one dimension at a time, on `h100-server`.
- Coverage: **100%** on both backends for all 7 dimensions.
- Code revision for the scoring run: `a044ac9`, with two later re-scores noted in
  place: `dynamics_degree` (v2 repair, `d/dt**p`) and `motion_smoothness`
  (direction-aware estimator, `feeb770`).
- `Overall Consistency` is out of scope: the plan requires human-authored prompt
  conditions (MD section 12.2) and no annotation round was run.

## Raw result

**This section is maintained at the latest revision and is the canonical table.**
Update it in place whenever a dimension is re-scored; do not add parallel copies.
Current snapshot: `dynamics_degree` scored with the shipped v2 repair (`d/dt**p`);
`motion_smoothness` re-scored with the direction-aware estimator (`feeb770`);
`multiplt_object` re-scored with per-half CIs and the §11.4 conjunction-control
predicate (`687d19a` + the scoring-code commit it names); all other dimensions at
scoring code `a044ac9`; coverage 100% on both backends for all seven dimensions.

CPA over all ordered level pairs, 95% cluster bootstrap over `base_id`. The last
two columns are the interval that actually decides a row: the **paired** delta
resamples each `base_id` cluster once and re-scores *both* backends on it, so it
is not the difference of two marginal intervals (which overlap even when the
paired interval excludes zero). `scripts/counterfactual/summarize.py` generates
these from the single frozen score tree; `table2.csv` carries the same numbers.

| dimension | family | type | bases | clips | Official | Repair | delta | paired delta 95% CI | significant |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| scene | environment_coverage | ordered | 25 | 125 | 0.3850 | 0.9300 | +0.5450 | [+0.425, +0.680] | yes |
| subject_consistency | temporal_relocation | mixed | 25 | 100 | 0.5917 | 0.8500 | +0.2583 | [+0.158, +0.350] | yes |
| multiplt_object (see note) | weakest_object_visibility | mixed | 25 | 150 | 0.5967 | 0.7633 | +0.1666 | [+0.090, +0.250] | yes (composite) |
| motion_smoothness | temporal_jerk | ordered | 25 | 125 | 0.8300 | 0.8800 | +0.0500 | [-0.015, +0.110] | parity |
| dynamics_degree (see note) | fps_resampling | invariance | 40 | 160 | 0.8333 | 0.7722 | -0.0611 | [-0.150, +0.017] | parity |
| human_action | filename_invariance | invariance | 25 | 75 | 1.0000 | 1.0000 | +0.0000 | [0.000, 0.000] | degenerate |
| spatial_relationship | directional_flip | ordered | 40 | 80 | 0.3667 | 0.0667 | -0.3000 | [-0.467, -0.133] | yes (Repair worse) |

**Read `significant` as a statement about the CPA statistic only, never about the
contract.** Two rows are significant yet unusable (`scene`,
`spatial_relationship` - their family premise fails); one is parity yet is the
real fix (`dynamics_degree` v2 satisfies the contract while CPA cannot see it);
and one is parity because a defect was successfully repaired
(`motion_smoothness` moved from -0.105 to +0.050 once `direction_change_t`
entered the discontinuity).

**`dynamics_degree` note —** the Repair column is the **v2 shipped repair**
(`d/dt**p`, lag-calibrated). The archived v1 (`d/dt`) scored CPA 0.8444 /
Δ +0.0111 and is kept at
`scores/archive/dynamics_degree__repair_v1_archived.jsonl`. **v2 is the variant
that satisfies the contract yet reports the *lower* CPA** — for this invariance
family CPA is non-diagnostic, so read the exponent, not the CPA.

**`multiplt_object` note —** the composite is a mixture, not a Repair win or loss. Split
by contract half (`multiplt_object.md`): sensitivity 0.5371 → 0.8171 (paired
Δ **+0.2800**, 95% CI [+0.1886, +0.3771] — a genuine win) and the tie-based
invariance pair 0.6000 → 0.0000 (paired Δ −0.6000, [−0.8000, −0.4000]). That
invariance number is **not a criterion any continuous estimator can satisfy**:
it asks a continuous score for an *exact tie* between two different corruption
geometries (a same-frame occlusion versus a clip in which no frame holds both
targets), and the Official 0.60 is saturation to sixteenths rather than
fidelity — the same criterion reads 0.90 vs 0.70 when taken as "control not
higher". The plan's §11.4 predicate is the test that applies, and both backends
pass it: control ≤ full occlusion in 18/25 (Repair) and 22/25 (Official) bases,
control means 0.2009/0.2436 (Repair) against a clean level of 0.5181. Diagnosis:
neither the SoftMin/`beta` configuration (swept β ∈ {1,3,10,30,100} and
`hard_min`, plus candidate thresholds 0.0–0.5: no setting produces an exact tie)
nor the aggregation reading temporal union as co-presence, but the detector's
confidence floor — at full suppression the weaker target is still answered at
mean confidence 0.3510 and the official 0.5 gate still accepts it in 6–15 of 16
frames, so no confidence-level rule can separate "present" from "suppressed".
See `multiplt_object.review.md` §3.

**`motion_smoothness` note —** this row is the **direction-aware estimator**
(`feeb770`); the archived v1 row (`a044ac9`) scored 0.8300 / 0.7250
(Δ −0.1050). The v1 paired delta CI was [−0.200, −0.020], so that deficit was
significant; the shipped row's paired delta is **+0.0500 with 95% CI
[−0.015, +0.110]**, which crosses zero. The shipped repair is therefore *at
parity* with Official on this family, not a demonstrated win. The paired
interval is carried in `table2.csv` (`delta_ci_low`/`delta_ci_high`).

**This table must not be read on its own.** Seven independent reviews
(`<dimension>.review.md`) found that for most rows the composite CPA is either
non-diagnostic or measures something other than the contract.

## What each row is actually worth

| dimension | review verdict | usable claim |
|---|---|---|
| `scene` | **Not a Repair win.** The two headline numbers are mutually inconsistent, and the only informative number is the Official one: the official Scene metric cannot recover the base video it was given. | Official is globally blind to which video it scored |
| `spatial_relationship` | **Not "Repair is worse".** Both numbers are artefacts of an unverified premise: the repair has no evidence on 79.4% of frames, and for most bases `original > flip` is false on this source. The Official number is exactly what upstream must produce, because that code never reads the sign of the geometry. | the family's premise fails; the fixture needs redesign |
| `multiplt_object` | Sensitivity half is a real Repair win (+0.2800 paired, CI [+0.1886, +0.3771]); the **tie-based invariance half is 0.00 vs Official 0.60 and is not satisfiable by a continuous estimator**. The plan's §11.4 "score the never-co-present control as incomplete" predicate passes on both backends, but not on the exact-tie criterion the composite uses. | win on ordering; the temporal-conjunction contract must be read as a level predicate, not a tie |
| `dynamics_degree` | Composite CPA is non-diagnostic and its sign is not trustworthy. Signed exponent (target 0): Official `+0.491`, archived v1 repair `−0.511` — the same violation mirrored — and the shipped v2 repair `+0.019`. v2 satisfies the contract yet scores the **lower** CPA. | v2 fixes the exponent; CPA cannot see it |
| `human_action` | **Unfalsifiable by construction.** The filename *is* the Official target label, so the invariance expectation is unsatisfiable for Official; the Repair target comes from metadata, so all three levels share one query and one byte-identical video and CV 0 is an identity. | the family tests nothing; needs redesign |
| `motion_smoothness` | The review found a **structural estimator defect**: it never scored direction change, so a hold-and-jump (`jerk_2`) tied or inverted a local reversal (`jerk_3`), which the contract requires to be strict. That is fixed (`feeb770`) and the dimension re-scored: the repair now beats Official on CPA (0.8800 vs 0.8300), on strict order (10/20 vs 7/20) and on Spearman (0.815 vs 0.765). The paired delta CI crosses zero ([−0.015, +0.110]), so this is parity, not a win. | the v1 deficit is fixed; the shipped repair matches Official on the declared ladder |
| `subject_consistency` | Pooled CPA is dominated by the easy sensitivity half. Split by contract half on test, the trade is explicit: sensitivity 0.9333 (Official) vs 0.7833 (Repair) — Official is *better* at what is easy — but **position invariance 0.2500, 95% CI [0.150, 0.350] (15/60 pairs, well below chance) — vs Repair 0.9167 [0.817, 1.000] (55/60)**. Official is systematically *anti*-invariant (its fixed first-frame anchor makes the same corruption score differently by position); Repair nearly removes it. | the invariance half is the win, and it must be quoted on its own |

## `dynamics_degree`: archived v1 vs shipped v2 (measured exponent)

The contract is `score independent of sampling interval`, i.e. a log-log slope
`p = 0` against inter-frame interval. Computed from the archived scores and the
`v2/` re-measurement on the scoring host:

| backend | fitted p (test, target 0) | within-base CV | tie-aware CPA | test level profile (fps8 → fps2) |
|---|---:|---:|---:|---|
| Official | **+0.4908** | 0.3177 | 0.8333 | 16.51 → 19.72 → 24.25 → 32.87 |
| Repair v1 (`d/dt`, archived) | **−0.5107** | 0.2360 | 0.8444 | 0.2632 → 0.2375 → 0.1936 → 0.1311 |
| Repair v2 `fixed1` (`d/dt**1`, control) | −0.5107 | 0.2360 | — | identical to v1 — the fix is behaviour-preserving at exponent 1 |
| **Repair v2 `fixed05` (`d/dt**0.5`)** | **+0.0187** | **0.1822** | **0.7722** | 0.0878 → 0.0910 → 0.0932 → 0.0906 |

`fixed1` reproducing v1 exactly is the control that the v2 plumbing changes
nothing else. `fixed05` is the shipped repair and removes the frame-rate
dependence: `p` moves from −0.511 to **+0.019** against a target of 0, and the
level profile flattens.

Two caveats carried by the result itself:

1. The 0.5 exponent is a **default** taken from a bootstrap CI over the same 40
   bases (`exponent_source =
   diffusive_sqrt_lag_default_bootstrap_ci_0p39_0p65_not_independently_calibrated`),
   not an independently calibrated constant.
2. It is an **aggregate-level** fix only: per-base `fps2/fps8` median is still
   1.284, so individual clips remain dispersed.

**v2 satisfies the contract yet scores a *lower* CPA than the broken v1**
(0.7722 vs 0.8444, Δ = −0.0611 against Official 0.8333). That is the clearest
single demonstration that a tie-margin CPA is anti-correlated with the contract
in this family, and the reason the exponent rather than the CPA is the headline
here.

## What can still be claimed

- **No dimension currently supports an unqualified "the repair is better" claim.**
  `scene`, `human_action` and `spatial_relationship` fail on family design;
  `multiplt_object` consists of a genuine sensitivity win plus an invariance
  half that no continuous estimator can satisfy (§‡ above), so it can be claimed
  only half by half and only with the §11.4 predicate as the invariance test;
  `dynamics_degree` is an invariance family where CPA cannot adjudicate — its
  shipped v2 repair does satisfy the contract (`p=+0.019`), but it can only be
  claimed on the exponent, never on CPA;
  `motion_smoothness` was a real deficit in the shipped estimator; the
  direction-aware fix removes it, but only to parity (the paired delta CI
  crosses zero).
- The most defensible positive result is **`subject_consistency` reported split by
  contract half**, where the invariance half improves from ≤11/60 to ≤42/60.
- The most defensible diagnostic result is **Official `dynamics_degree` is
  frame-rate dependent** (`p = +0.49`, ~2x inflation from 8 to 2 fps), which is a
  property of the shipped official metric independent of any repair.
- `scene` currently supports a *negative* diagnostic finding about Official
  (it cannot recover the base video it was given), not a repair win.

## Limitations

- **First-run measurements from the current metric packages.** Model, CUDA and
  weight parity against the frozen E0 baselines has **not** been verified, so
  absolute values are not reproduced official numbers.
- Failed clips are recorded as `failed` and excluded from the CPA denominator —
  never scored as zero — and coverage is reported per dimension.
- A same-rank (invariance) family's CPA must never be quoted without its
  dispersion statistics or a signed exponent.
- Reviews were carried out on the derived reports and the committed code; where a
  claim needed a model run it is marked as such in the individual review.
- `motion_smoothness`'s `magnitude_weight`/`direction_weight` (0.7/0.3) are
  engineering defaults fixed before the re-score, not tuned on this dimension's
  clips; the ordering fix is verified on canonical trajectories, and the real
  clips only supply the reported scores.

## Provenance

| item | value |
|---|---|
| dataset | `xjuIcthub/counterfactual-vbench` (1199 files) |
| dataset root on host | `/root/wenbiao_zhao/datasets/counterfactual-vbench/` |
| scores | `<dataset>/scores/`, one file per dimension and backend |
| reports | `<dataset>/reports/`, mirrored in this directory |
| machine-readable table | `table2.csv`, `table2.json`, `SUMMARY.md` |
| scoring code | `scripts/counterfactual/` at `a044ac9` |
