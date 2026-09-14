# Multiple Objects Implementation Report

## 1. Official upstream mapping

The immutable reference is `/home/msy625/vbench1`, remote
`https://github.com/msy625/VBench.git`, branch `master`, commit
`13dee903cc97e2633ed6e8f50dea61bc90717935`. The mapped sources are:

- `vbench/multiple_objects.py`: metadata, frame checking, video/dataset aggregation, distributed gather;
- `vbench/utils.py`: 16-frame uniform middle sampling with last-frame padding;
- `vbench/third_party/grit_model.py`: GRiT `DenseCaptioning` ObjectDet wrapper;
- `vbench/third_party/grit_src/image_dense_captions.py`: model configuration and threshold `0.5`;
- `grit/modeling/roi_heads/grit_roi_heads.py`: strict `scores > score_thresh`, NMS, and final score construction.

The local `VBench1.0_paper.pdf` is a 284-byte 404 HTML response, not a PDF.
The paper-level definition was therefore checked against the official
[CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf),
not the invalid local file.

## 2. Official pipeline and Evaluation Contract

Official input comes from `auxiliary_info.object`. Upstream requires exactly
two targets and unpacks `object_info.split(" and ")`; matching is exact and
case-sensitive. For each video it loads 16 middle-sampled frames, downsizes
only when the shorter side exceeds 768 (to 720), runs the official GRiT
ObjectDet model/weight, and marks a frame successful only when both target
labels occur in that frame. The video score is successful frames divided by
sampled frames. Dataset and distributed aggregation use global successful-frame
and frame-count totals.

Evaluation Contract: every sampled frame is checked for simultaneous complete
target-set co-presence, and the video score is that frame-completeness signal's
temporal coverage.

## 3. Core Failure Mechanism

**Mechanism Hypothesis — Hard-Conjunction Uncertainty Amplification.** GRiT
uncertainty is discretized by a fixed threshold and then combined by a hard
AND. A near-threshold target can therefore cause an abrupt frame-level failure.
This is detector uncertainty interacting with the evaluator decision; it is
not a pure aggregation defect under perfect-oracle detections.

## 4. Actual repair implementation

For each target `k` in sampled frame `t`, the Audit backend uses the maximum
captured official ROI threshold score among exact-label matches:

```text
e[t,k] = max matched detection confidence; 0 if no match
c_t = -(1 / beta) * log((1 / K) * sum_k exp(-beta * e[t,k]))
S_video = mean_t(c_t)
```

SoftMin is evaluated with a max-shifted log-sum-exp calculation. Inputs must be
finite and in `[0,1]`; `beta` must be finite and positive; empty target sets and
zero-frame videos are rejected rather than silently assigned a score. The
result is bounded to `[0,1]` only after input validation. Duplicate detections
for one label use the maximum confidence. No arithmetic target mean is used.

The repair creates the same upstream GRiT ObjectDet configuration but lowers
its repair-only ROI candidate threshold before model construction. A source-
locked hook copies the selected instance's threshold-input score before the
text decoder overwrites `Instances.scores`. It does not change tensor values or
the official backend. For reliable diagnostics, each sampled frame also
receives a separate pass with the official ROI threshold `0.5`; its returned
labels drive the displayed official decision.

Defaults are `repair_candidate_threshold=0.0`, `softmin_beta=10.0`, and
`aggregation_mode="softmin"`. `hard_min` is an explicit diagnostic/ablation.
These are design defaults, not optimized experimental results.

## Confidence Provenance

**Verdict: same-score continuous relaxation.** The official threshold input is
`scores` in `GRiTROIHeadsAndTextDecoder._forward_box`: the mean foreground/
background class probabilities over the cascade stages (`R x (K+1)` per image),
combined with proposal objectness as `sqrt(class_probability *
proposal_score)` because `MULT_PROPOSAL_SCORE=True`. After removal of the
background column it has shape `R x K` (`K=1` in this configuration).

`fast_rcnn_inference_single_image_GRiT` applies `scores > score_thresh` before
NMS and top-k. It then writes surviving values to `result.scores`. The adapter
copies those exact values into `official_threshold_scores` before ObjectDet
text decoding. The repaired per-target evidence comes from this copied field,
not from the final `predictions["instances"].scores`.

After threshold/NMS/top-k, GRiT generates labels and overwrites
`pred_instance.scores` with
`sqrt(official_threshold_score * exp(description_logprob))`. That overwritten
value is retained only as `final_instance_confidence` diagnostics. Official
Multiple Objects membership uses `instances.det_obj.data`, produced by the
second ObjectDet decoder path; the adapter uses the same labels. The hook also
checks that the primary and ObjectDet selections have identical boxes and
threshold scores before allowing index-aligned label/score/bbox extraction.

`repair_candidate_threshold=0.0` is not an evaluator-side collection filter.
It changes the repair inference candidate population at the ROI filter before
NMS/top-k so candidates with `0 < score <= 0.5` can be decoded and matched.
It does not alter the score formula, backbone, or weights. It cannot recover
CenterNet proposals removed by `INFERENCE_TH=0.0001`, proposal pre/post-NMS
top-k, ROI NMS/top-k, non-finite filtering, or scores `<=0.0`.

## 5. Preserved components and rejected hypotheses

Preserved: official upstream identity gate, GRiT backbone/weight, ObjectDet
task, sampled frames, resize rule, exact target-label matching, same-frame set
semantics, temporal mean, official backend behavior, and global frame-weighted
dataset/multi-GPU aggregation.

Explicitly rejected: temporal union, sequential completion, tracking,
segmentation, count/multiplicity repair, smoothing, temporal attention,
dominant-object compensation, detector replacement, and arithmetic-mean target
aggregation. Perfect-oracle conjunction remains consistent with the intended
contract.

## 6. Diagnostics

The computation path records target objects, sampled-frame index, aligned
matched label/box/threshold-score detections, final instance confidence,
maximum per-target official score, weakest
target/confidence, labels from the official-threshold pass, official binary
decision and its source, repaired frame score, official/candidate thresholds,
threshold margin, aggregation mode, beta, frame score sum/count, and final
video score. Threshold margin is computed from the captured official threshold
score; final instance confidence is never substituted into that field.

## 7. Confirmed implementation problems and fixes

- **Parity/Correctness:** the previous repair initialized GRiT at official
  threshold `0.5`, so sub-threshold evidence was already lost. It now configures
  the repair-only threshold before model construction.
- **Correctness/Diagnostics:** the previous code re-thresholded final
  description-adjusted `instances.scores` with `>= 0.5`. Upstream filters a
  different pre-description score with strict `> 0.5`. The repair no longer
  double-thresholds final scores; official diagnostics use a real separate
  official-threshold pass.
- **Confidence provenance:** the previous repair used final
  description-adjusted `instances.scores` and `pred_object_descriptions`, while
  official membership uses `det_obj` and thresholds the earlier ROI score. The
  adapter now captures the exact threshold score before overwrite and matches
  it with `det_obj`, with runtime selection/index-alignment assertions.
- **Correctness:** previous matching was case-insensitive while upstream uses
  exact string membership. Matching is now exact and case-sensitive.
- **Numerical:** invalid/out-of-range evidence, invalid beta, empty SoftMin,
  and zero-frame videos could be silently converted or scored. They now fail
  explicitly before output clipping.
- **Interface:** the previous CLI bypassed audit-core GPU parsing, silently
  fell back to CPU, and could not handle comma-separated GPUs. It now uses the
  shared parser/CUDA contract, supports deterministic round-robin shards, and
  includes `python -m multiple_objects` entry-point support.
- **Aggregation:** the previous dataset summary averaged video scores. Worker
  outputs now retain numerator/count and combine global frame totals, matching
  official single-/multi-GPU weighting, including empty-shard avoidance.
- **Engineering:** an unused identity diagnostics shim was removed.

## 8. Tests

Model-free tests cover complete/missing targets, distractor invariance,
duplicate-label maximum, SoftMin monotonicity, missing-object behavior,
symmetry, large-beta convergence, single-target identity, strict
`0.49/0.50/0.51` boundary semantics, sequential-not-simultaneous regression,
empty detections, malformed metadata, invalid confidence/range/beta, zero
frames, official/repair isolation, exact diagnostic-label routing, GRiT
label/box/score index alignment, frame-total shard aggregation, CLI GPU parsing,
upstream locks/source invariants, and module entry point.

## 9. Limitations and future validation

The exact threshold score is available only for candidates surviving the
repair candidate threshold, NMS, and top-k. Earlier CenterNet/proposal
filtering and later NMS/top-k removals remain irreversible. Lowering the ROI
candidate threshold changes repair inference population and compute cost; it
must not be described as recovering all raw detector evidence.

Only code facts and model-free contracts are validated here. A real matching
CUDA/Detectron2/GRiT run is still required to confirm runtime parity and memory
cost of the two-pass diagnostic path. Counterfactual data and human evaluation
are required to test the mechanism hypothesis, choose `beta`, and measure any
correlation or benchmark effect. No superiority or empirical improvement is
claimed.
