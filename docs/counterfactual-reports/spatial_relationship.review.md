# Review: `spatial_relationship` counterfactual report

Scope: the archived `docs/counterfactual-reports/spatial_relationship.md` (code SHA
`bdfda5cc6e31725cddb6f45ce194ff1333f7c05d`), the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `run_dimension.py`, `score.py`), the spatial repair
(`metrics/spatial-relationship/src/spatial_relationship/`), and the base selection that
built the family (`scripts/counterfactual/select_bases.py`, `pick_detectable.py`,
`build.py`, `transforms.py`).

Verdict: the row `spatial_relationship | directional_flip | 0.3667 | 0.0667 | -0.3000` in
`docs/counterfactual-reports/README.md` must not be read as "the repaired metric is worse
than Official". Both numbers are artefacts of a family whose premise was never verified:
the repair has **no evidence at all** on 79.4% of frames, and on this source the
`original > flip` expectation is false for most bases. The Official number is not a noisy
measurement of a working metric either — it is exactly what the upstream code must
produce, because that code never reads the sign of the geometry. Sections 2 and 3 separate
what is provable from source code from what was measured on the scoring host.

All numbers below were produced on `h100-server` (`zhengchen-ubuntu-8xh100-05`, 7×H100,
torch 2.5.1+cu121, `/root/wenbiao_zhao/venvs/vbench`) against the built
`counterfactual-vbench` dataset with the repository's own code at `a044ac9`, read-only.
The two diagnostic entry points are archived at
`output/counterfactual/_scratch/diag_spatial.py` and
`output/counterfactual/_scratch/diag_spatial_labels.py`.

## 1. The published table already contains the contradiction

The archived report's `Contract decomposition` block publishes, for the test split at rank
gap 1 (30 pairs), a match rate of `0.0667` and a tie rate of `0.8333` for the repair and
`0.3667`/`0.3667` for Official. Expanding those rates over the 30 pairs:

| backend | correct | tied 0:0 | inverted |
|---|---:|---:|---:|
| official | 11 | 11 | 8 |
| repair | 2 | **25** | 3 |

`match rate 0.0667 / tie rate 0.8333` is not "the repair gets the direction wrong". It is
"the repair produces the same score — zero — on both levels of 25 of 30 pairs". A tie is
scored as an error by `cpa_at_margin` (`cpa.py:74-81`), so the headline `-0.3000` measures
the repair's silence, not its sign. The same silence is visible in the archived
`Score sensitivity` table:

| backend | level | n | mean | distinct |
|---|---|---:|---:|---:|
| official | `original` | 40 | 0.3104 | 22 |
| official | `horizontal_flip` | 28 | 0.3108 | 17 |
| official | `vertical_flip` | 12 | 0.3319 | 12 |
| repair | `original` | 40 | 0.0448 | 6 |
| repair | `horizontal_flip` | 28 | 0.0312 | 4 |
| repair | `vertical_flip` | 12 | 0.0521 | 3 |

70 of the 80 scored clips are exactly `0.0`. The report never says what those zeros are,
and the scoring driver could not have said: `score.py::_first` kept only the scalar.

## 2. The Official metric is direction-blind by construction

This needs no statistics. Upstream `VBench/vbench/spatial_relationship.py::get_position_score`
is:

```python
if locality in 'on the right of' or locality in 'on the left of':
    if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
        score = 1
```

`x_distance` is computed and then only its absolute value is used. The `on the top of` /
`on the bottom of` branch does the same with `abs(y_distance)`. The sign of the arrangement
never enters the score, so mirroring the video cannot lower it in any frame.

Reproduced locally against the ported geometry (`metrics/spatial-relationship/src/spatial_relationship/relation.py`),
on boxes where the subject is unambiguously left of the object:

| relation | level | `official_position_score` | `ordered_position_score` |
|---|---|---:|---:|
| `on the left of` | original | 1.000 | 1.000 |
| `on the left of` | horizontal mirror | **1.000** | 0.000 |
| `on the right of` | original | 1.000 | 0.000 |
| `on the right of` | horizontal mirror | **1.000** | 1.000 |

The measured level means (0.3104 / 0.3108 / 0.3319) are the empirical shadow of that code
path: Official scores `original` and its mirror identically on average, and its
`0.3667` is re-encoding and detector noise around a quantity that is invariant by design.
`11 correct / 11 tied / 8 inverted` is that noise, not a directional signal.

Consequence for the paper: the Official finding here is a **code-level** finding and should
be argued from `get_position_score`, with the level means as confirmation. Quoting
`0.3667 ≈ chance` as if the metric had been given a fair chance understates it: the metric
cannot represent the distinction the probe tests, at any sample size.

## 3. The repair's zero is a detector/binding floor, not a wrong sign

The missing evidence was recovered by re-running the repair over the 40 `original` clips
with the repository's own `score_predictions` and counting `frame_reason` (640 frames):

| `frame_reason` | frames | share |
|---|---:|---:|
| `missing_object` | 278 | 43.4% |
| `missing_subject` | 230 | 35.9% |
| `direction_mismatch` | 72 | 11.2% |
| `axis_mismatch` | 22 | 3.4% |
| `relation_satisfied` | 20 | 3.1% |
| `relation_satisfied_with_iou_penalty` | 18 | 2.8% |

**508 of 640 frames (79.4%) never reach the geometry check** — the detector did not return
a box for one of the two required labels. Only 132 frames (20.6%) are resolved at all, and
38 of those (28.8% of resolved, 5.9% of all frames) satisfy the relation.

This also settles the role-assignment question. `ORDERED_ROLE` — maximising signed
geometry over every `(subject, object)` pair instead of binding the highest-confidence
instance per role — does recover cases the identity-first contract discards, but only for a
handful of clips: it exceeds the identity-first score on 4 of the 40 originals. Over those
40 clips the means move

| variant | mean clip score |
|---|---:|
| `ordered_role_identity_assignment` (end-to-end) | 0.0448 |
| `ordered_role_identity_assignment` + detection-conditioned | 0.0854 |
| `ORDERED_ROLE` (end-to-end) | 0.0921 |
| `ORDERED_ROLE` + detection-conditioned | 0.1396 |

Assignment strictness is a real but secondary effect; the floor is the detector.

## 4. The objects are genuinely absent, not merely renamed

If the 79% were a vocabulary mismatch between VBench annotations and GRiT's emitted
categories, the fix would be label normalisation. Dumping every label GRiT emitted for
those 40 clips shows otherwise: **58 of the 80 target nouns (72.5%) do appear at least
once**, and where they do not, the generation contains a different object:

| prompt target | labels GRiT emitted |
|---|---|
| `frisbee` | `umbrella x56`, `tennis racket x18` |
| `cow` | `elephant x91`, `person x5` |
| `banana` (partner of `broccoli`) | `broccoli x109` only |
| `toaster` | `laptop`, `chair`, `book`, `keyboard`, `microwave x10` |
| `vase` | `scissors x16`, `toothbrush x16` |

The remaining misses are presence *within* a clip rather than naming: `remote` is emitted
once in 16 frames, `fork` twice, `cup` twice. Per-clip native co-detection over the 40
`original` clips:

| frames with both targets natively detected (of 16) | clips |
|---|---:|
| 0 | **23** |
| 1–4 | 6 |
| 5–8 | 4 |
| 9–15 | 2 |
| 16 | 5 |

**23 of 40 bases (57.5%) have no measurable frame at all**, and only 5 bases are fully
resolvable. Of those five, two satisfy the ordered relation in all 16 frames
(`bicycle on the left of car`, `giraffe on the right of bird`), one does so only under the
multi-instance `ORDERED_ROLE` search (`bed on the left of tv`: 0.9922 ordered, 0.0000
identity-first), and two violate it in every frame (`banana on the top of apple` 1/16,
`clock on the left of vase` 0/16).

So the premise `original > flip` is not merely unverified — for this source it is mostly
false. The generator frequently does not place the named objects at all, and when it does,
the arrangement is as likely to be reversed as correct.

## 5. Plan section 9.2 was never carried out

`docs/plans/2026-09-14-experiment-plan-8d.md` section 9.2 requires, before the family may be
used: both objects visibly present, the original relation confirmed by a human validity
check, and sufficient detector coverage for an end-to-end comparison. Section 9.3
additionally requires a reciprocal control (flip the video *and* the prompt relation, expect
a tie) and section 9.6 requires reporting both the detection-conditioned and the end-to-end
settings.

None of that shipped:

- `build.py::_row` stamps every manifest row `manual_validity_status: "pending"` and nothing
  ever resolves it. There is no eligibility check for the directional premise anywhere in
  the build path — `transforms.directional_flip` is a pure frame transform and needs no
  detector.
- `pick_detectable.py` guarded only `subject_consistency` and `multiplt_object`; Spatial
  Relationship was not in `DETECTOR_DIMENSIONS`, so its 40 bases were selected without any
  box-presence requirement, let alone a directional one.
- The built family has three levels (`original`, `horizontal_flip`, `vertical_flip`) and no
  reciprocal level, so the one plan 9.3 statistic that is well-posed without a validity
  check was never measured.
- Only the end-to-end setting was scored.

The `directional_flip` premise is a property of the *source clip*, and the source clips were
never asked to have it.

## 6. What changed as a result

The repair-side knobs that section 3's evidence points at are now implemented, and the
dataset side now refuses to ship an unverified family.

Metric (`metrics/spatial-relationship/`):

- `diagnostics.py::aggregate_video` gained `condition_on_detection`, which averages only over
  frames where both roles were detected, so a detector drop-out leaves the denominator
  instead of being scored as a relation violation. `detected_frame_count` /
  `scored_frame_count` / `aggregation_method` record the change.
- `AblationMode.ORDERED_ROLE` is reachable through `audit.py`, `metric.py` and the CLI
  (`--audit-mode`, `--detection-conditioned`); `formula_version` names the contract that
  produced a run.
- `scripts/counterfactual/score.py` and `run_dimension.py` expose the same knobs
  (`--repair-mode`, `--repair-detection-conditioned`, or `VBENCH_AUDIT_SPATIAL_MODE` /
  `VBENCH_AUDIT_SPATIAL_DETECTION_CONDITIONED`), record them on every scored row and in the
  report header, and refuse to append a second configuration into one shard file.

Dataset (`scripts/counterfactual/pick_detectable.py`):

- Spatial Relationship is now a detector-scanned dimension with a **directional gate**: at
  least `--min-co-detected-frames` (default 0.25) of frames must natively detect both
  targets, and at least `--relation-min-frames` (default 0.75) of those must satisfy the
  ordered relation. Every kept base records `relation_oracle`, `relation_frame_rate`,
  `relation_frames_scored` and `relation_frames_total`, which `build.py` copies into each
  manifest row. A failure prints the observed rate distribution so the thresholds are tuned
  against data rather than guessed. `--relation-oracle human --relation-validity <file>`
  substitutes the plan 9.2 human check.
- This gate uses the audited geometry on the shared GRiT boxes. It is therefore **not an
  independent oracle**: a repair that consumes the same boxes is being asked to rank a
  verified arrangement above its mirror, not to discover the arrangement. Any claim built
  on a gated family must say so.

Evidence persistence (`score.py::frame_evidence`, `run_dimension.py::render_evidence_section`):

- Every scored row now carries a compact `frame_reason` histogram plus the detection counts,
  and the generated report renders them next to the CPA. This is what section 3 had to be
  reconstructed by hand to obtain.

Applied to the existing 40 bases, the gate admits **2 of 40** at the default thresholds —
which is the correct behaviour and the quantitative statement of this review: the budget of
10 dev + 30 test bases is not reachable from this source.

## 7. What the report should say instead

1. Replace the `directional_flip` verdict with the code-level finding from section 2, and
   state that the counterfactual family could not be built on VBench-1.0 human-preference
   clips: 23/40 bases have no frame in which both named objects are detected, and among the
   frames that are resolved only 28.8% satisfy the prompt's relation (section 4).
2. Report the repair's `0.0667` with its tie rate — `25/30 pairs tied at 0:0` — and never as
   a direction error. On the evidence available, the repair has not been shown to get the
   direction wrong once.
3. Do not quote `0.3667` as a passing/failing measurement of Official. Quote
   `get_position_score`, with the level means as confirmation.
4. Drop this dimension from Table 2, or carry it as an explicit "not constructible from this
   source" row. The same caveat applies to any other family whose expectation is a property
   of the source clip rather than of the transformation.

## 8. What would produce working data

The probe is sound; the source is not. Three routes, in descending order of strength:

- **Controlled family.** Compose clips from real object crops taken from VBench frames,
  pasted at coordinates chosen by construction, then mirror. Ground truth is exact by
  construction, the base count is unlimited, and GRiT-detectability can be self-checked
  before scoring. Official still fails by section 2; the repair is then tested against a
  premise that is true rather than assumed. This is the only route that removes both the
  circularity and the budget problem.
- **Signed-agreement family on the existing clips.** Derive each base's `expected_rank` from
  the measured arrangement of its original level instead of asserting it. Every measurable
  base then contributes a signed prediction. It stays circular for the repair (same
  detector) and still only covers 17/40 bases, so it is a comparative statement about
  Official, not a clean repair win.
- **Per-video tracking in the repair.** Loosen the per-frame conjunction by carrying boxes
  across frames (`track_target` already exists for the build path). This lifts detection
  coverage toward 100% with a small change, but it changes the metric's contract and plan
  section 9.2's scope note currently excludes temporal tracking. If taken, it must be
  reported as a new scored variant, not as the frozen repair.

## 9. Status of the numbers in this review

- Measured on `h100-server` against `counterfactual-vbench` with repository code at `a044ac9`,
  read-only: sections 1, 3, 4 and the gate projection in section 6. The scripts are scratch
  (untracked) at `output/counterfactual/_scratch/diag_spatial.py` and
  `.../diag_spatial_labels.py`; they were copied to `/tmp` on the host and run as
  `CUDA_VISIBLE_DEVICES=2 python -u /tmp/diag_spatial.py original` and
  `... /tmp/diag_spatial_labels.py`. Neither writes to the dataset or the repository, and
  `git status` on the host stayed clean.
- Proven from source, not re-measured: section 2's upstream code path, and the local
  geometry reproduction (model-free).
- Not measured: any re-scored Official or repair run under the new variants; the reciprocal
  control; the rebuild of the family with the gate. The variant means in section 3 are over
  the 40 `original` clips only, so they are not comparable to the published CPA tables.
- The archived scores were produced before the fixes in section 6 and are not affected by
  them; the report at
  `docs/counterfactual-reports/spatial_relationship.md` still shows the old numbers.
