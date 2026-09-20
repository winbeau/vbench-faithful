# Eight-Dimension VBench Audit: Experimental Protocol

> **Scope update (2026-09-15):** the audit scope is now eleven dimensions — the
> seven in-flight dimensions below (all except Overall Consistency) plus four
> candidates: `background_consistency`, `temporal_style`, `object_class`,
> `color`. `overall_consistency` is retired from this round (its estimator is
> the same one `temporal_style` uses). See
> [`2026-09-15-dimension-scope-11d.md`](2026-09-15-dimension-scope-11d.md);
> the protocol below still governs the seven retained dimensions.

## 1. Purpose and claim boundary

This document specifies the experiments required for the eight-dimension
VBench audit. It is an execution protocol, not a record of completed results.
Every number produced later must be traceable to a frozen manifest, code
revision, model checkpoint, and output directory.

The target paper claim is:

> Contract-aware refinements improve the response of VBench metrics to
> controlled, dimension-relevant changes while retaining comparable behavior
> on natural VBench generations.

Two kinds of evidence are therefore required:

1. **Contract fidelity:** Official and repaired metrics are evaluated on
   controlled counterfactuals with a known expected score relation.
2. **Natural-set compatibility:** Official and repaired metrics are evaluated
   on unmodified VBench human-preference videos.

Passing only the first criterion supports a diagnostic or contract-fidelity
claim. Claiming that a repaired metric can replace Official VBench additionally
requires acceptable natural-set human agreement, score coverage, and failure
behavior.

The eight audited dimensions are:

1. Dynamic Degree;
2. Subject Consistency;
3. Human Action;
4. Spatial Relationship;
5. Scene;
6. Multiple Objects;
7. Overall Consistency;
8. Motion Smoothness.

Dynamic Degree and Spatial Relationship are the two headline empirical cases.
The other six dimensions must each have at least one real-model
counterfactual experiment, but only their primary result needs to appear in
the four-page paper.

---

## 2. Terminology

### 2.1 VBench Natural Set

The Natural Set is the unmodified public VBench human-preference video set. It
contains videos from four generators and official pairwise human labels. The
repository currently refers to the corresponding baseline analysis as `E0`.
`E0` is an internal experiment identifier, not the name of an official VBench
dataset, and should not be used as a dataset name in the paper.

Each annotated group contains four generator outputs and six unordered model
pairs. Human labels are:

- `1`: video/model A is preferred;
- `0`: video/model B is preferred;
- `0.5`: tie.

The frozen split is prompt-disjoint, with nominal 40% development and 60%
test partitions. The current four-dimension counts are:

| Dimension | Dev pairs | Test total pairs | Test scored pairs | Current Official coverage |
|---|---:|---:|---:|---:|
| Dynamic Degree | 870 | 1,290 | 1,290 | 100% |
| Subject Consistency | 870 | 1,290 | 1,290 | 100% |
| Human Action | 1,200 | 1,800 | 1,800 | 100% |
| Spatial Relationship | 1,290 | 1,950 | 1,470 | 77.8% overall; 75.4% test |

Spatial coverage is lower because Official VBench 1.0 does not score the
`inside of` prompts present in the public preference set. Unsupported samples
must remain abstentions; they must not be assigned zero.

The pairwise metadata also contains annotations for Scene, Multiple Objects,
Overall Consistency, and Motion Smoothness. Their current test-pair inventories
are 1,560, 1,470, 1,680, and 1,290 respectively, but their complete
Official/Ours score caches have not yet been produced.

### 2.2 VBench-CF

`VBench-CF` is the working name for the counterfactual test set derived from
public VBench videos. It is not a separate source dataset. Each
counterfactual family contains:

- one unmodified VBench base clip;
- one frozen prompt and dimension contract;
- one controlled transformation;
- two or more transformation levels;
- an expected pairwise or ordered score relation.

For example, a Spatial family can contain an original clip and its horizontal
flip. With the original left/right prompt fixed, the expected relation is that
the original score is higher.

### 2.3 Independent unit

The independent statistical unit is the base clip or prompt, not a derived
video and not an individual pair. All variants from the same base receive the
same `base_id` and remain in the same split and bootstrap cluster.

---

## 3. Data acquisition and provenance

### 3.1 Required source material

Obtain the public VBench human-preference package containing:

- official per-dimension JSON annotations;
- the referenced four-generator videos;
- prompt and auxiliary dimension metadata;
- model/generator identity;
- human pairwise labels.

Do not treat the prompt suite alone as a video dataset. If the video package is
not available, the Natural Set and VBench-derived counterfactual experiments
cannot be completed from CSV metadata alone.

### 3.2 Provenance record

For every source video record:

```text
source_dataset
source_archive_or_url
source_annotation_sha256
source_video_sha256
dimension
prompt_id
prompt
generator
group_id
video_uid
relative_video_path
license_or_usage_note
```

For every derived video additionally record:

```text
base_id
derived_id
split
intervention_family
intervention_level
expected_relation
transformation_parameters
transformation_code_sha
output_video_sha256
manual_validity_status
```

### 3.3 No metric-based selection

Base eligibility must be decided from metadata, programmatic construction, or
human inspection. Do not select bases because Official or Ours gives a desired
score. This would bias the comparison.

### 3.4 Split policy

Reuse the frozen E0 prompt split whenever a base comes from the preference set.
All variants of a base remain in its original split. Development data may be
used for:

- tie-margin selection;
- scalarization or threshold selection;
- transformation debugging;
- hyperparameter selection.

Test data must not be inspected for parameter selection. If additional prompts
are introduced, create a deterministic prompt-disjoint split before running
any final metric.

---

## 4. Recommended sample budget

The target below balances eight-dimensional breadth with feasible compute.

| Dimension | Dev bases | Test bases | Expected variants per base |
|---|---:|---:|---:|
| Dynamic Degree | 10 | 30 | 8--12 |
| Spatial Relationship | 10 | 30 | 3--5 |
| Subject Consistency | 5 | 20 | 4--5 |
| Human Action | 5 | 20 | 5--7 |
| Scene | 5 | 20 | 5--6 |
| Multiple Objects | 5 | 20 | 5--6 |
| Overall Consistency | 5 | 20 | 4--5 |
| Motion Smoothness | 5 | 20 | 5--7 |

This yields 50 development bases, 180 test bases, and approximately 800--1,000
derived videos. The manuscript must report both base count and derived-video
count.

If compute is insufficient, reduce the number of transformation levels before
reducing the number of independent test bases.

---

## 5. Common scoring protocol

### 5.1 Compared systems

Every main result compares:

1. the locked Official VBench 1.0 implementation;
2. the repaired/Audit implementation;
3. dimension-specific mechanism ablations where required.

Official and Audit should share the original backbone, checkpoint, input
frames, and metadata whenever the repair concerns target binding or scoring.
Any unavoidable representation change must be identified explicitly.

### 5.2 Counterfactual Pair Accuracy

For a pair `(base, counterfactual)`, define

```text
delta_score = score_counterfactual - score_base
expected_relation in {-1, 0, +1}
```

Using a tie margin `delta` selected on development prompts:

```text
prediction = +1 if delta_score >  delta
prediction = -1 if delta_score < -delta
prediction =  0 otherwise
```

Counterfactual Pair Accuracy (CPA) is the proportion of predictions matching
the expected relation. Report zero-margin and dev-calibrated tie-aware CPA.

For ordered levels, expand each family into all ordered pairs for CPA, but keep
the same `base_id` cluster. Also report a sequence-level statistic such as
Spearman correlation or strict-order rate.

### 5.3 Macro averaging

Do not allow a family with more derived variants to dominate the overall
result. Compute:

1. metrics within each base;
2. mean across bases within each intervention family;
3. macro mean across intervention families within a dimension;
4. optional macro mean across dimensions.

The primary paper result is per-dimension CPA, not only one eight-dimension
average.

### 5.4 Uncertainty

Use 2,000 bootstrap repetitions clustered by `base_id` for VBench-CF and by
`prompt_id` for the Natural Set. Report percentile 95% confidence intervals for:

- each method;
- `Ours - Official` paired difference;
- non-inferiority differences on natural videos.

Derived clips and the six pairs within a four-model group must never be
resampled as independent observations.

### 5.5 Missing scores

Record every score as one of:

- succeeded;
- unsupported/abstained;
- model failure;
- invalid counterfactual;
- missing input.

Report coverage separately. Never convert failure or unsupported cases to
zero. For a method comparison, report both shared-support results and each
method's absolute coverage.

---

## 6. Dynamic Degree protocol

### 6.1 Contract

Dynamic Degree should respond to task-relevant motion magnitude and temporal
support while distinguishing subject motion from camera-induced apparent
motion.

### 6.2 Base eligibility

Select VBench Dynamic Degree clips with:

- a visible primary subject;
- no hard scene cut;
- sufficient duration for resampling;
- a prompt that identifies subject motion;
- decodable and monotonic frame timestamps.

Record the original FPS, duration, resolution, prompt, and dominant motion
type. Human inspection must be performed without seeing Official/Audit scores.

### 6.3 Primary family: camera injection

For every base, produce:

- original;
- low global motion;
- medium global motion;
- high global motion.

Apply a deterministic partial-affine transform to the complete frame sequence.
The first experiment should use transformations supported by the repair:

- horizontal/vertical translation;
- small rotation;
- uniform zoom.

Use the same transform schedule for all methods. Keep compression settings,
duration, FPS, and subject content fixed. Avoid parallax in the primary family;
reserve it for robustness analysis.

Evaluate each video under:

- a SUBJECT motion target;
- a CAMERA motion target.

Expected behavior:

- SUBJECT evidence remains stable under added global camera motion;
- CAMERA evidence increases with injected camera strength.

Primary statistics:

- SUBJECT invariance CPA;
- CAMERA monotonic CPA;
- camera/subject interaction effect;
- residual and camera intensity curves.

### 6.4 Auxiliary family: FPS invariance

Resample the same decoded trajectory to 8, 12, 16, 24, and 30 FPS while
preserving video duration. Use timestamp-aware encoding and verify the output
frame count and duration.

Expected behavior: task-relevant intensity should remain approximately stable.

Report:

- within-base coefficient of variation;
- relative range;
- all-pairs invariance CPA.

### 6.5 Auxiliary family: motion coverage

Identify an active segment from the source clip. Construct fixed-duration
versions in which that motion occupies 25%, 50%, 75%, and 100% of the timeline.
Fill inactive intervals using repeated boundary frames from the same clip. Keep
the active segment's playback rate unchanged.

Expected behavior: temporal coverage increases monotonically. Inspect the
derived clips to reject visible cut artifacts.

Report coverage MAE, Spearman correlation, strict ordering, and pairwise CPA.

### 6.6 Optional controlled speed family

Natural video time warping confounds speed, duration, and frame repetition.
Therefore use a small procedural supplement for exact speed tests:

- fixed background and subject appearance;
- fixed duration and camera;
- common path direction;
- five known image-diagonal-per-second speeds.

Treat this as a sanity check rather than the source-dataset headline result.

### 6.7 Ablations

Avoid ambiguous labels such as `Time-only` unless their full computation is
specified. Prefer leave-one-component-out variants:

- `Official`;
- `Full`;
- `Full w/o time normalization`: displacement rather than speed, no duration
  weighting;
- `Full w/o source separation`: apparent flow is used for every prompt;
- `Full w/o persistence`: intensity is retained but temporal coverage omitted;
- `Full w/o prompt routing`: all channels are computed but apparent flow is
  always selected.

The primary Dynamic scalar for Natural Set comparison must be frozen before
test evaluation. The simplest initial candidate is routed duration-weighted
intensity. Coverage remains a separate diagnostic. If a combined scalar is
selected, choose it on development data only and report all candidates.

---

## 7. Subject Consistency protocol

### 7.1 Contract

The same subject inconsistency should receive a comparable penalty regardless
of when it occurs, while genuine subject corruption must still lower the score.

### 7.2 Base eligibility

Select VBench Subject Consistency clips with:

- one clearly identifiable primary subject;
- no scene cut;
- limited existing identity drift;
- sufficient duration for three non-overlapping corruption locations.

### 7.3 Primary family: temporal relocation

Create one fixed subject-region corruption using the same mask, intensity, and
duration. Place it at:

- the beginning;
- the middle;
- the end.

Candidate corruptions include hue/texture replacement, localized blur, or a
fixed identity-like patch. The dataset-construction mask may come from a
detector or segmenter even though the evaluated metric still uses whole-frame
DINO features. Manually reject transformations that substantially modify the
background.

Expected behavior: the three corrupted versions receive comparable scores.

Report:

- maximum position gap;
- within-family score variance;
- position-invariance CPA.

### 7.4 Sensitivity controls

For each base also retain:

- clean video;
- subject-corrupted video;
- background-only corruption where feasible.

The repaired metric must satisfy:

```text
clean score > subject-corrupted score
```

while reducing the temporal-position gap. A constant or insensitive metric
must not be considered successful merely because it is position invariant.

### 7.5 Compared variants

- Official adjacent + first-frame anchor;
- symmetric local + all-pairs aggregation;
- fixed ROI with Official aggregation;
- fixed ROI with symmetric aggregation.

Keep fixed ROI as a negative ablation unless it simultaneously reduces
background sensitivity and preserves positive subject-corruption sensitivity.

---

## 8. Human Action protocol

### 8.1 Contract

The score must use the prompt/metadata action target, remain invariant to file
storage names, retain the target-class evidence, and distinguish persistent
from localized action evidence where claimed.

### 8.2 Base eligibility

Select VBench Human Action videos for which:

- the prompt maps exactly to one Kinetics-400 category;
- the action is visible on inspection;
- the video decodes identically after file copying/renaming;
- the action is not determined solely from background context where possible.

### 8.3 Primary family: filename invariance

Create byte-identical or losslessly copied inputs with:

- correct action filename;
- wrong Kinetics action filename;
- neutral filename.

Use identical prompt and explicit `target_action` metadata for Audit.

Expected behavior: all Audit scores are exactly equal; Official may change
because its target is filename-derived.

Report:

- filename-invariance CPA;
- maximum absolute score gap;
- target strings selected by each backend.

### 8.4 Target-discrimination family

Keep video bytes fixed and evaluate:

- correct target;
- semantically related but incorrect Kinetics target;
- unrelated Kinetics target.

Expected ordering:

```text
correct > related incorrect > unrelated
```

The related label must be selected before inspecting scores.

### 8.5 Optional duration family

Construct fixed-length videos containing 25%, 50%, 75%, and 100% of an action
segment, padded with neutral frames from the same actor and scene. Use only if
the clips remain visually natural under manual inspection.

### 8.6 Compared variants

- Official filename target + rounded Top-5/threshold Boolean;
- explicit target with Official Boolean decision;
- explicit target with continuous full-clip probability;
- explicit target with continuous temporal windows.

Freeze one public scalar definition before final evaluation. The current code
uses duration-weighted mean target probability, while an older implementation
report says `score=null`; resolve this inconsistency before running results.

Do not claim action-completion understanding unless the optional duration,
preparation-only, and incomplete-action tests are completed.

---

## 9. Spatial Relationship protocol

### 9.1 Contract

The score must evaluate an ordered query `(subject=A, relation=R, object=B)`
with signed geometry. Swapping roles or reversing direction must change the
result when the relation changes.

### 9.2 Base eligibility

Select VBench clips with:

- relation in left/right/top/bottom;
- both objects visibly present;
- original relation confirmed by a human validity check;
- sufficient detector coverage for an end-to-end comparison.

Exclude `inside of` from Official comparisons because the locked evaluator
does not support it.

### 9.3 Primary family: directional flip

- Horizontal flip for left/right prompts;
- vertical flip for top/bottom prompts.

With the original prompt fixed, the transformed score should decrease.

Create a reciprocal control by changing the prompt relation together with the
flip. The reciprocal score should remain close to the original.

Report directional CPA, signed score gap, reciprocal MAE, and coverage.

### 9.4 Role-swap family

Keep video bytes fixed and change query metadata from `(A,R,B)` to `(B,R,A)`.
For non-reciprocal pairs, the original should score higher.

### 9.5 Optional multi-instance distractor family

Add an extra A or B instance that creates a geometrically favorable but
semantically incorrect pair. Use source objects from other VBench frames or a
transparent overlay. Manually validate that the intended pair remains clear.

### 9.6 Compared variants

- Official role pooling + unsigned geometry;
- signed-only;
- role-preserving with Official unsigned geometry;
- signed + role-preserving;
- identity-first Full.

Report two settings:

1. detection-conditioned: both A and B are correctly detected;
2. end-to-end: detector failures retained.

Existing HFlip and role-swap artifacts may be reported only after their source
inputs and outputs are copied into the reproducible workspace or the
experiments are rerun with the frozen code.

---

## 10. Scene protocol

### 10.1 Contract

Scene evidence should describe the distributed environment rather than a
single local object or lexical cue.

### 10.2 Base eligibility

Select VBench Scene clips with:

- an unambiguous target scene;
- no scene cut;
- broad environmental support;
- a second VBench clip from an incompatible scene for controlled mixing.

The target and donor clips should have compatible resolution, duration, and
camera behavior after preprocessing.

### 10.3 Primary family: environment coverage

Divide corresponding frames into a 2x2 grid. Replace quadrants of the wrong
scene with target-scene quadrants to obtain target environmental support of:

```text
0%, 25%, 50%, 75%, 100%
```

Use a fixed quadrant ordering per base, then repeat with a second deterministic
ordering on development data to check layout sensitivity.

Expected behavior: target-scene score increases with distributed support.

Report monotonic CPA, Spearman correlation, and score range.

### 10.4 Local-cue control

Insert one small target-associated patch into an otherwise wrong environment.
The target scene should remain unsupported globally. Audit should not produce
a high score from this isolated cue.

### 10.5 Compared variants

- Official Tag2Text caption + substring check;
- continuous global image-text score;
- global + 2x2 regional environment score.

Freeze the concrete image-text encoder and checkpoint before running the final
experiment. Report that changing from Tag2Text to the selected encoder is a
representation change, not only an aggregation change.

---

## 11. Multiple Objects protocol

### 11.1 Contract

All target objects must be simultaneously present in a frame. The score should
respond continuously to evidence for the weakest required target without
turning temporal union into co-presence.

### 11.2 Base eligibility

Select VBench Multiple Objects clips with:

- exactly two named target categories;
- both targets visibly present in at least part of the video;
- usable boxes or masks for the weaker target;
- no same-class multiplicity requirement.

### 11.3 Primary family: weakest-object visibility

Keep target A unchanged and progressively weaken target B through deterministic
occlusion, blur, or alpha blending at 0%, 25%, 50%, 75%, and 100% severity.
Prefer masks; if boxes are used, manually reject edits that destroy unrelated
content.

Expected behavior: completeness decreases monotonically as B becomes less
visible.

Report monotonic CPA, weakest-target evidence, threshold crossings, and
Official tie rate.

### 11.4 Temporal-conjunction control

Create a fixed-length video in which only A is visible in the first half and
only B in the second half. No frame contains both. Both Official and Audit
should score this as incomplete. This prevents the repair from silently
changing same-frame conjunction into temporal union.

### 11.5 Compared variants

- Official threshold + hard AND;
- captured continuous confidence + hard minimum;
- captured continuous confidence + SoftMin;
- one development-only sensitivity check for SoftMin beta.

The real GRiT/Detectron2 runtime is required. Model-free unit tests are not an
empirical result.

---

## 12. Overall Consistency protocol

### 12.1 Contract

A global prompt match should not allow a strongly matched condition to fully
compensate for a failed critical condition.

### 12.2 Base eligibility and condition annotation

Select compositional VBench prompts with two to four explicit conditions, for
example:

```text
A red car drives on a snowy road.
```

Annotate proposition-preserving conditions such as:

```text
red car
car driving
snowy road
```

Use explicit human-authored conditions for the main experiment; do not depend
on an unvalidated automatic parser.

Select bases judged to satisfy the original prompt sufficiently for a
condition-substitution experiment.

### 12.3 Primary family: condition substitution

Keep video bytes fixed and create prompt/condition variants in which exactly
one condition is replaced:

- attribute substitution;
- action substitution;
- scene substitution;
- object substitution where applicable.

Expected behavior: the original condition set scores higher than each
single-condition violation.

Report violation CPA, score drop, and whether the substituted condition is
identified as the weakest condition.

### 12.4 Violation-count family

On a smaller subset, replace zero, one, and two conditions. Expected score
decreases monotonically with violation count.

### 12.5 Compared variants

- Official full-prompt global ViCLIP;
- condition mean only;
- condition minimum only;
- condition mean + minimum;
- global + condition Full.

Tune mixture weights on development prompts only. A small human validity check
is required because natural VBench videos may not perfectly satisfy their
original prompts.

---

## 13. Motion Smoothness protocol

### 13.1 Contract

Given that motion exists, smoothness should reflect continuity of the motion
trajectory rather than motion magnitude, FPS, or generic reconstruction
difficulty.

### 13.2 Base eligibility

Select VBench Motion Smoothness clips with:

- visible nonzero motion;
- no scene cut;
- no severe pre-existing freeze or decoding error;
- sufficiently smooth original trajectory for perturbation.

### 13.3 Primary family: temporal jerk severity

Keep the decoded frames fixed but progressively perturb their temporal order:

- level 0: original;
- level 1: duplicate one frame locally;
- level 2: duplicate and skip two frames;
- level 3: reverse a short local segment;
- level 4: introduce multiple local reversals/skips.

Use the same normalized temporal location across bases. Verify that encoding
does not introduce unrelated visual artifacts.

Expected behavior: smoothness decreases monotonically with perturbation
severity.

Report monotonic CPA, Spearman correlation, mean discontinuity, and tail
discontinuity.

### 13.4 Controls

1. **Uniform FPS resampling:** preserve the underlying trajectory; score should
   remain relatively stable after timestamp normalization.
2. **Motion-magnitude control:** uniformly speed up a smooth sequence without
   adding trajectory discontinuity; smoothness should not decrease solely
   because magnitude increases.
3. **Single-tail event:** add one severe jerk to a long sequence to test whether
   mean+tail aggregation detects localized failure.

### 13.5 Compared variants

- Official AMT interpolation score;
- flow acceleration only;
- acceleration + direction change;
- temporal mean only;
- mean + tail Full.

Real AMT and flow-model execution is required for the final result.

---

## 14. Minimal human validation

Large-scale human preference collection is not required for the first
submission. Human evaluation is used only to validate that the constructed
counterfactuals have the intended semantic relation.

### 14.1 Minimum sample

Randomly sample:

- 20 pairs from Dynamic;
- 20 pairs from Spatial;
- 10 pairs from each of the remaining six dimensions.

Total: 100 pairs. Use three annotators per pair, for 300 judgments.

Do not use these judgments to tune model parameters. Sample after the
counterfactual generator is frozen.

### 14.2 Questions

Use one dimension-specific question per pair:

- Dynamic: ignoring camera motion, is subject motion stronger in A, stronger in
  B, or the same?
- Subject: is the degree of subject inconsistency greater in A, greater in B,
  or the same?
- Human Action: which video better performs the specified action, or are they
  the same?
- Spatial: which video better satisfies the stated ordered relation?
- Scene: which video better depicts the target environment?
- Multiple Objects: which video more completely contains both targets at the
  same time?
- Overall: which video satisfies more of the stated prompt conditions?
- Motion Smoothness: which video has smoother motion?

Allow four responses:

```text
A
B
same/indistinguishable
invalid/unclear
```

### 14.3 Validity criteria

Before metric comparison, report:

- percentage of majority labels matching the predetermined relation;
- invalid/unclear rate;
- Fleiss' kappa or Krippendorff's alpha;
- results for Dynamic/Spatial separately from the supporting dimensions.

Target criteria:

- at least 80% majority agreement with the intended relation;
- less than 10% invalid/unclear pairs.

If a transformation family fails these criteria, redesign or exclude the
family using development data. Do not relabel test examples after seeing
metric scores.

Objective metadata transformations such as filename changes do not need a
large human sample, but their videos should still be checked for decoding and
copying errors.

---

## 15. Natural Set evaluation

### 15.1 Metrics

For every dimension with complete Official and Audit scores, report:

- zero-margin test pair accuracy;
- dev-calibrated tie-aware test accuracy;
- Kendall tau-b;
- coverage;
- model-level Pearson with `n=4`, marked descriptive.

### 15.2 Success criterion

Counterfactual superiority is the primary criterion:

```text
lower 95% CI of (CPA_Ours - CPA_Official) > 0
```

For a replacement claim, also predefine a natural-set non-inferiority margin:

```text
lower 95% CI of (PairAcc_Ours - PairAcc_Official) > -delta_NI
```

A candidate margin of two to three percentage points may be considered, but it
must be justified and frozen before test evaluation.

Interpret outcomes as follows:

| Counterfactual result | Natural pair result | Allowed claim |
|---|---|---|
| better | non-inferior/better | contract-faithful replacement candidate |
| better | significantly worse | useful audit/refinement, not replacement |
| not better | non-inferior | no evidence that the repair solves its target |
| worse | worse | reject or redesign the repair |

Do not treat a change in four-model Pearson alone as decisive. Pair-level
agreement and its confidence interval are more informative.

---

## 16. Efficiency, robustness, and failure reporting

For every Official/Audit run record:

- wall-clock time per video;
- peak GPU memory if available;
- number of model passes;
- success, failure, and abstention counts;
- detector/decoder/flow/affine/parser failure causes;
- fallback use;
- effective frame count and duration.

Required robustness checks:

- one development-only parameter sensitivity curve for each new threshold or
  mixture weight used in the final score;
- random-seed repeat for stochastic preprocessing, if any;
- byte/hash confirmation for filename invariance;
- video duration/FPS/frame-count validation after temporal transforms;
- detector-conditioned and end-to-end Spatial results.

Do not hide failed samples by averaging only successful scores without
reporting coverage.

---

## 17. Execution stages

### Stage 0: freeze definitions

- Freeze Official upstream SHA and each Audit code SHA.
- Resolve the Human Action scalar/report inconsistency.
- Freeze the Dynamic scalar used for Natural Set ranking.
- Freeze Scene encoder/checkpoint.
- Freeze Overall semantic-condition annotations and mixture weights.
- Record every model checkpoint hash.

### Stage 1: acquire and audit source data

- Download/locate VBench preference videos.
- Verify every manifest path and video hash.
- Reproduce E0 Official scores where runtime permits.
- Record missing and unsupported cases.

### Stage 2: create development counterfactuals

- Implement transformations as deterministic scripts.
- Generate only development prompts first.
- Inspect outputs and adjust transformations without viewing test scores.
- Freeze transformation parameters and expected relations.

### Stage 3: pilot inference and parity

- Run a small Official/Audit parity sample.
- Verify shared backbone/checkpoint and frame inputs.
- Confirm output schemas, score directions, and failure accounting.

### Stage 4: generate and score the test set

- Generate all test derivatives once from the frozen manifest.
- Run Official, ablations, and Full.
- Do not change thresholds after inspecting test results.

### Stage 5: human validity sample

- Randomly draw the frozen 100-pair sample.
- Collect 300 judgments.
- Report invalid pairs and agreement before selecting examples for the paper.

### Stage 6: statistics and paper artifacts

- Compute per-family and per-dimension CPA.
- Run cluster bootstrap.
- Run Natural Set comparisons.
- Export final tables, plots, manifests, and run metadata.

---

## 18. Four-page paper output

The experiment may be broad, but the paper presentation must be compact.

### Figure 1: teaser and pipeline

Show:

- Spatial flip: semantics change while Official remains unchanged;
- Human Action rename: pixels remain unchanged while Official changes;
- measurement contract -> counterfactual -> diagnosis -> refinement.

### Table 1: eight-dimension audit and primary intervention

One row per dimension:

```text
dimension | collapsed contract | primary transformation | expected response
```

### Table 2: main quantitative result

One row per dimension:

```text
dimension | Natural Pair Acc. Official/Ours |
CF CPA Official/Ours | paired difference and 95% CI | coverage
```

If Natural Audit scores are unavailable for a dimension, show `--` and do not
imply completion.

### Table 3: compact mechanism ablation

Use two panels only:

- Dynamic: Official, Full w/o time, Full w/o source, Full w/o persistence,
  Full;
- Spatial: Official, signed-only, role-only, Full.

Subject and Human Action diagnostics should be summarized in one or two
sentences. Do not preserve separate large tables for every dimension.

### Space priority

If the final draft exceeds four pages, cut in this order:

1. optional second figure;
2. detailed Related Work;
3. secondary method formulas;
4. optional counterfactual families;
5. parameter-sensitivity prose.

Do not cut sample counts, confidence intervals, baselines, coverage, or the
main Dynamic/Spatial ablations.

---

## 19. Completion checklist

### Data

- [ ] Official VBench preference videos are locally available and hashed.
- [ ] Eight dimension manifests are complete.
- [ ] Prompt/base-disjoint splits are frozen.
- [ ] Base eligibility is independent of metric scores.
- [ ] Derived video transformations are deterministic and recorded.

### Models

- [ ] Official parity is checked for each available backend.
- [ ] Checkpoint hashes and preprocessing are recorded.
- [ ] Human Action scalar definition is frozen.
- [ ] Dynamic Natural Set scalar definition is frozen.
- [ ] Scene and Overall model/condition choices are frozen.

### Experiments

- [ ] Eight primary counterfactual families are scored.
- [ ] Dynamic and Spatial ablations are complete.
- [ ] Natural Set Official/Ours results are complete where claimed.
- [ ] Failure and coverage tables are complete.
- [ ] Cluster bootstrap confidence intervals are generated.
- [ ] Minimal human validity study is complete.

### Claims

- [ ] Every headline claim maps to a completed result.
- [ ] No model-free unit test is presented as empirical improvement.
- [ ] Historical results are either rerun or explicitly labeled historical.
- [ ] Human alignment is claimed only if human-labeled comparisons support it.
- [ ] Replacement claims satisfy the Natural Set compatibility criterion.

