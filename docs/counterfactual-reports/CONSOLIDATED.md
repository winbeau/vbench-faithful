# VBench-CF: consolidated status

Everything established so far about the counterfactual (metamorphic) audit of
VBench 1.0: what was built, what was scored, and what each result is actually
worth after independent review.

- Dataset: `counterfactual-vbench`, 205 bases / 815 derived clips, 7 dimensions,
  published at `xjuIcthub/counterfactual-vbench`.
- Scoring: Official VBench 1.0 vs the repaired metric, 6 physical GPUs in
  parallel, one dimension at a time, on `h100-server`.
- Coverage: **100%** on both backends for all 7 dimensions.
- Code revision for the scoring run: `a044ac9` (later revisions are noted where
  they change a number).
- `Overall Consistency` is out of scope: the plan requires human-authored prompt
  conditions (MD section 12.2) and no annotation round was run.

## Raw result

**This section is maintained at the latest revision and is the canonical table.**
Update it in place whenever a dimension is re-scored; do not add parallel copies.
Current snapshot: scoring code `a044ac9`, coverage 100% on both backends for all
seven dimensions, plus the `dynamics_degree` v2 re-measurement marked below.

CPA over all ordered level pairs, 95% cluster bootstrap over `base_id`.

| dimension | family | type | bases | clips | Official | Repair | delta | Repair 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---|
| scene | environment_coverage | ordered | 25 | 125 | 0.3850 | 0.9300 | +0.5450 | [0.840, 0.985] |
| subject_consistency | temporal_relocation | mixed | 25 | 100 | 0.5917 | 0.8500 | +0.2583 | [0.758, 0.933] |
| multiplt_object | weakest_object_visibility | mixed | 25 | 150 | 0.5967 | 0.7633 | +0.1666 | [0.677, 0.830] |
| dynamics_degree | fps_resampling | invariance | 40 | 160 | 0.8333 | 0.7722 | −0.0611 | [0.667, 0.867] |
| human_action | filename_invariance | invariance | 25 | 75 | 1.0000 | 1.0000 | +0.0000 | — |
| motion_smoothness | temporal_jerk | ordered | 25 | 125 | 0.8300 | 0.7250 | −0.1050 | [0.650, 0.790] |
| spatial_relationship | directional_flip | ordered | 40 | 80 | 0.3667 | 0.0667 | −0.3000 | [0.000, 0.167] |

† The `dynamics_degree` Repair column is the **v2 shipped repair**
(`d/dt**p`, lag-calibrated). The archived v1 (`d/dt`) scored CPA 0.8444 /
Δ +0.0111 and is kept at
`scores/archive/dynamics_degree__repair_v1_archived.jsonl`. **v2 is the variant
that satisfies the contract yet reports the *lower* CPA** — for this invariance
family CPA is non-diagnostic, so read the exponent, not the CPA.

**This table must not be read on its own.** Seven independent reviews
(`<dimension>.review.md`) found that for most rows the composite CPA is either
non-diagnostic or measures something other than the contract.

## What each row is actually worth

| dimension | review verdict | usable claim |
|---|---|---|
| `scene` | **Not a Repair win.** The two headline numbers are mutually inconsistent, and the only informative number is the Official one: the official Scene metric cannot recover the base video it was given. | Official is globally blind to which video it scored |
| `spatial_relationship` | **Not "Repair is worse".** Both numbers are artefacts of an unverified premise: the repair has no evidence on 79.4% of frames, and for most bases `original > flip` is false on this source. The Official number is exactly what upstream must produce, because that code never reads the sign of the geometry. | the family's premise fails; the fixture needs redesign |
| `multiplt_object` | Sensitivity half is a real Repair win; the **invariance half is a Repair loss** (0.00 vs Official 0.60). The composite nets a win against a loss. | win on ordering, loss on temporal conjunction |
| `dynamics_degree` | Composite CPA is non-diagnostic and its sign is not trustworthy. Signed exponent (target 0): Official `+0.491`, archived v1 repair `−0.511` — the same violation mirrored — and the shipped v2 repair `+0.019`. v2 satisfies the contract yet scores the **lower** CPA. | v2 fixes the exponent; CPA cannot see it |
| `human_action` | **Unfalsifiable by construction.** The filename *is* the Official target label, so the invariance expectation is unsatisfiable for Official; the Repair target comes from metadata, so all three levels share one query and one byte-identical video and CV 0 is an identity. | the family tests nothing; needs redesign |
| `motion_smoothness` | The deficit is real on the declared ladder, but the report misdescribes it (pure ordered family reported as mixed), the deficit is partly structural in the Audit estimator (it ties or inverts `jerk_2` vs `jerk_3`, which the contract requires to be strict), and the missing paired interval means −0.1050 is quoted at unsupported precision. | real deficit, under-evidenced |
| `subject_consistency` | Pooled CPA is dominated by the easy sensitivity half; the invariance half is ≤11/60 (Official) vs ≤42/60 (Repair). | must be reported split by contract half |

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
  `multiplt_object` is a win on one contract half and a loss on the other;
  `dynamics_degree` is an invariance family where CPA cannot adjudicate — its
  shipped v2 repair does satisfy the contract (`p=+0.019`), but it can only be
  claimed on the exponent, never on CPA;
  `motion_smoothness` is a real deficit.
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

## Provenance

| item | value |
|---|---|
| dataset | `xjuIcthub/counterfactual-vbench` (1199 files) |
| dataset root on host | `/root/wenbiao_zhao/datasets/counterfactual-vbench/` |
| scores | `<dataset>/scores/`, one file per dimension and backend |
| reports | `<dataset>/reports/`, mirrored in this directory |
| machine-readable table | `table2.csv`, `table2.json`, `SUMMARY.md` |
| scoring code | `scripts/counterfactual/` at `a044ac9` |
