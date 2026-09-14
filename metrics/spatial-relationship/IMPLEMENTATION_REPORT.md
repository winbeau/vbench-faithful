# Spatial Relationship implementation report

## Intended semantic

The metric evaluates an ordered query `(subject=A, relation=R, object=B)`. Audit execution keeps this order through semantic parsing, identity evidence, independent subject/object assignment, ordered geometry `R(A, B)`, frame scoring, and video aggregation.

## Upstream identity

- Path: `/home/msy625/vbench1`
- Remote: `https://github.com/msy625/VBench.git`
- Branch: `master`
- Commit: `13dee903cc97e2633ed6e8f50dea61bc90717935`
- Dirty status: clean at implementation and parity-test time
- Git submodules: none reported

This checkout is a unified VBench-series repository. Only its root `vbench/` implementation is used; the directory name is not treated as a verified VBench1.0 tag.

## Original execution path

```text
compute_spatial_relationship
-> DenseCaptioning.initialize_model_det
-> load_dimension_info
-> load_video(num_frames=16, sample="middle")
-> optional resize (shorter side > 768 -> 720)
-> GRiT / Detectron2 ObjectDet at threshold 0.5
-> get_dect_from_grit
-> check_generate
-> get_position_score
-> mean frame score per video
-> mean over all sampled frames
```

## Confirmed code facts

The locked metadata has 84 entries and explicit `object_a`, `object_b`, and `relationship` fields. Relations comprise left, right, top, and bottom variants.

Official `check_generate()` pools A-labelled and B-labelled boxes into one list, then enumerates list-index pairs and selects the maximum score. The structured A/B roles are not preserved in the geometric candidate representation.

Official horizontal relations use `abs(dx) > abs(dy)` without a left/right sign check. Vertical relations use `abs(dy) > abs(dx)` without a top/bottom sign check. The IoU threshold is 0.1; a dominant-axis pair scores 1 below the threshold and `0.1 / IoU` at or above it.

Detectron2 instances contain scores internally, but the official `dense_pred_to_caption_tuple()` and `get_dect_from_grit()` path discards them. The same wrapper exposes raw predictions through `run_det_tensor()`, so audit retains the original `Instances.scores` when present and records `confidence=null` only when absent. This does not change the detector, threshold, or model inference.

The repository file `VBench1.0_paper.pdf` is invalid 404 HTML. The official arXiv rendering states that Spatial Relationship checks whether object relationships follow the prompt, covers left/right and top/bottom, and uses rule-based evaluation. It does not document the implementation's role pooling or exact IoU penalty.

## Confirmed experiment facts

Read-only recomputation from the existing experiment JSON artifacts under `/home/msy625/projects/server-backup/VBench-experiments/runs/exp02/spatial_relationship/` gives:

| Experiment | Official / ablation original | Counterfactual | Interpretation |
|---|---:|---:|---|
| HFlip official, 10 pairs | 1.0000 | 1.0000 | gap 0; CPA 0/10 |
| signed-only, 10 pairs | 0.9991053 | 1.0000 | direction sign alone is insufficient |
| signed geometry + role preservation, 10 pairs | 0.9000 | 0.1625 | gap 0.7375; CPA 9/10 |
| Role Swap official, 9 valid sources | 1.0000 | 1.0000 | role CPA 0/9 |
| Role Swap role-aware, 9 valid sources | 1.0000 | 0.1805556 | role CPA 9/9 |
| Reciprocal sanity, 9 valid sources | 1.0000 | 1.0000 | reciprocal MAE 0 for official and role-aware |

The 9-source summaries exclude pair 0004 because the role-aware original score is 0, matching the prior manual-validity selection. These are existing experiment artifacts; no model experiment was rerun here.

## Core failure mechanism

The strongest supported mechanism is loss of ordered structured semantics between detection and geometric verification. The scorer can answer whether some relation-compatible pair exists instead of whether the assigned subject instance satisfies `R(A, B)` with the assigned object instance. HFlip, role swap, reciprocal, and multi-instance probes are manifestations or controls around this one mechanism.

## Audit execution path

```text
explicit metadata
-> OrderedRelationQuery(subject, relation, object)
-> official 16-frame sampling / resize / GRiT detection
-> Detection(id, label, bbox, confidence-or-null)
-> separate subject_candidates and object_candidates
-> independent identity-first assignment
-> signed ordered geometry R(subject_box, object_box)
-> preserved official axis dominance and IoU penalty
-> frame score
-> unchanged arithmetic mean over sampled frames
```

The primary assignment strategy is exact label filtering, detector-confidence descending when confidence exists, then stable detector order. Relation geometry is not an assignment input. Same-class queries without extra identity descriptors are marked `same_class_role_ambiguity` and score 0. Missing entities score 0 with explicit reasons. Temporary detection loss scores that frame as 0; no tracking or temporal repair is implemented.

## Preserved modules and behavior

- Locked upstream `vbench/spatial_relationship.py` official entrypoint
- GRiT / Detectron2 ObjectDet model and official weight contract
- confidence threshold 0.5
- 16-frame middle sampling and short-video padding
- resize rule and tensor path
- official unsigned geometry for `--vbench`
- IoU threshold and overlap penalty
- frame/video/dataset arithmetic aggregation
- official raw frame and video result semantics
- one official model per GPU worker, with per-video calls to the unchanged upstream scorer so a corrupt video does not invalidate its whole shard

## Modified modules

- `公共/audit-core/src/vbench_audit_core/outputs.py`: generic metric payload flattening, dynamic CSV columns, and `run.log` creation
- `指标/spatial_relationship/src/spatial_relationship/cli.py`: metric-specific orchestration on top of common CLI/input/device/output infrastructure
- `文档/upstream-mapping.md`: source facts and plumbing mapping

## Added modules

- `schemas.py`: ordered query, detection, role-bound candidates, geometry and diagnostics schemas
- `relation.py`: official parity scorer and signed ordered geometry
- `models.py`: role binding, identity-first assignment and internal ablation enum
- `diagnostics.py`: full/compact/off diagnostics and video aggregation
- `backends/vbench.py`: locked-fork official delegation and GRiT plumbing
- `backends/audit.py`: ordered-role audit scoring
- `backends/__init__.py`

## Unit and counterfactual tests

The local suite contains 23 tests. Twenty-two model-free tests pass and one real model parity test is skipped because the required CUDA, Detectron2/GRiT runtime, weight, and real video fixture are unavailable.

Covered contracts include official source geometry parity, upstream SHA/dirty gate, official metadata conversion, per-video official failure isolation, correct role, role swap, reciprocal equivalence, horizontal flip sensitivity, missing entity, identity corruption, confidence/order assignment, multi-instance distractor resistance, same-class ambiguity, temporary detection loss, random reciprocal algebra, single-video CLI smoke, batch `--both`, shared run-id, GPU-list forwarding, interrupted-run recording, diagnostics and output files.

Two isolated wheels were built and installed into a temporary Python 3.10 environment. The installed `spatial-relationship` entrypoint and `uv run --active --no-sync spatial-relationship --help` both succeeded.

## Official parity evidence

The copied official geometry parity helper was compared directly with the `get_position_score()` AST extracted from the locked source and matched on horizontal, vertical, overlap, and reversed-box cases. The plumbing test proves exact metadata nesting and raw-result preservation while delegating to `compute_spatial_relationship()`. The opt-in real parity test runs that reference entrypoint and the extracted one-model/per-video adapter on the same video, metadata, weight, device, and locked SHA, then compares frame results, video result, and dataset result.

Real end-to-end parity is not verified. The official GRiT weight is absent from `~/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth`, CUDA is unavailable, and no accessible parity video fixture was provided. No claim is made about real sampled frames, detections, frame results, or final numerical parity in this environment.

## Diagnostics and outputs

Single-video runs default to full per-frame diagnostics; batch runs default to compact video diagnostics. Set `VBENCH_AUDIT_DIAGNOSTICS=full|compact|off` to control volume without changing the public CLI.

Each result preserves `video`, `prompt`, `subject`, `relation`, `object`, `backend`, `score`, `status`, and `failure_reason`, plus the configured diagnostics. Outputs are isolated under `spatial-relationship/<backend>/<run-id>/`, and `--both` shares one run-id.

## Remaining hypotheses and validation gaps

- Audit now retains Detectron2 `Instances.scores` through the wrapper's raw-prediction method. A real-model run is still needed to verify field presence and ordering on the locked environment; absence safely falls back to null and stable detector order.
- Same-class role assignment may become possible if future metadata provides independent identity descriptors; this stage intentionally abstains with numeric 0.
- Real multi-GPU spawn behavior is unit-covered only through deterministic sharding and CLI forwarding, not exercised on hardware.
- Human GT accuracy, balanced accuracy, agreement, and pairwise counterfactual agreement remain future validation work.
- Temporal tracking, persistence, majority voting, learned pair selection, replacement detectors, and threshold tuning are explicitly out of scope.
