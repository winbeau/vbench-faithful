# Upstream mapping

## Reference gate

- Reference path: `/home/msy625/vbench1`
- Remote: `https://github.com/msy625/VBench.git`
- Branch: `master`
- Commit: `13dee903cc97e2633ed6e8f50dea61bc90717935`
- Worktree: clean at inspection time
- Submodules: none reported by Git
- Scope: use root `vbench/` VBench dimensions only; do not import `VBench-2.0/`, `vbench2_beta_i2v/`, `vbench2_beta_long/`, or `vbench2_beta_trustworthiness/`.

## Dimension entry points

| Metric | Upstream module | Entry point | Current status |
|---|---|---|---|
| dynamic-degree | `vbench/dynamic_degree.py` | `compute_dynamic_degree` | adapter pending RAFT/torch/CUDA closure |
| motion-smoothness | `vbench/motion_smoothness.py` | `compute_motion_smoothness` | adapter pending dependency audit |
| subject-consistency | `vbench/subject_consistency.py` | `compute_subject_consistency` | adapter pending semantic input mapping |
| scene | `vbench/scene.py` | `compute_scene` | adapter pending semantic input mapping |
| human-action | `vbench/human_action.py` | `compute_human_action` | adapter pending UMT dependency audit |
| spatial-relationship | `vbench/spatial_relationship.py` | `compute_spatial_relationship` | official plumbing adapter and ordered-role audit implemented; real GPU parity pending |
| overall-consistency | `vbench/overall_consistency.py` | `compute_overall_consistency` | adapter pending ViCLIP dependency audit |
| multiple-objects | `vbench/multiple_objects.py` | `compute_multiple_objects` | adapter pending dense-caption dependency audit |

## Dynamic degree evidence

The reference implementation imports `cv2`, `numpy`, `torch`, `tqdm`, `easydict`, and the bundled RAFT implementation. It samples frames with `max(1, round(fps / 8))`, uses RAFT with 20 iterations, computes the top 5% flow magnitude mean, applies a resolution-scaled threshold, and averages per-video boolean results. These facts are source observations only; no real inference was run in this implementation.

The adapter must preserve this behavior and vendor or otherwise package the required RAFT source under its license. It must not replace RAFT with OpenCV flow and call the result official.

## Dependency gate

The reference requirements include unpinned scientific/model packages and `transformers==4.33.2`; its setup check requires a CUDA-enabled PyTorch installation. The current inspected environment has Python 3.10 in `/home/msy625/miniconda3/envs/vbench`, PyTorch `2.13.0+cu130`, CUDA unavailable, and no `nvidia-smi` access. Therefore model dependencies are intentionally not added to the workspace or lock file yet.

## Spatial Relationship mapping

Inspection was repeated against the reference gate above before implementation.

### Original execution path

`compute_spatial_relationship()` initializes `DenseCaptioning` in ObjectDet mode with the GRiT weight, loads explicit spatial metadata through `load_dimension_info()`, distributes prompt records, and calls `spatial_relationship()`. Each video is loaded with `load_video(..., num_frames=16)`, whose exact `middle` sampler uniformly divides the source frame range and pads short videos with the last sampled frame. If the shorter side exceeds 768, the tensor is resized so that the shorter side becomes 720. GRiT detections are then passed to `check_generate()`. Per-video output is the mean of 16 frame scores; the dataset result is the mean over all frame scores.

### Confirmed official scoring behavior

- Metadata contains ordered `object_a`, `object_b`, and `relationship` fields.
- `check_generate()` pools every bbox whose label equals either object into one `frame_obj_locats` list. Object role is no longer represented in this candidate list.
- It enumerates unordered list-index pairs and takes the maximum geometric score in each frame.
- For both left and right, `get_position_score()` checks only `abs(dx) > abs(dy)`. For both top and bottom it checks only `abs(dy) > abs(dx)`. Direction sign is not checked.
- The IoU threshold is 0.1. A dominant-axis pair scores 1 when IoU is below 0.1 and `0.1 / IoU` otherwise.
- `dense_pred_to_caption_tuple()` exposes label and integer XYXY bbox to the official evaluator but discards per-instance score. The same `DenseCaptioning` wrapper exposes raw predictions through `run_det_tensor()`, and Detectron2 `Instances.scores` is present in the predictor path. Audit reads that original score when available and records `confidence=None` only if the field is absent.

### Plumbing adaptations

- `spatial_relationship.backends.vbench` validates the exact upstream remote, branch, SHA, and clean worktree before importing `vbench.spatial_relationship`.
- Unified `metadata.json` entries are converted to the exact nested VBench full-info shape in a temporary file.
- The parity adapter delegates to `compute_spatial_relationship()` unchanged. Runtime workers initialize the same upstream `DenseCaptioning` ObjectDet model once, then call the unchanged upstream `spatial_relationship()` once per video. This plumbing boundary preserves the scorer while allowing a corrupt video to fail independently instead of invalidating the whole GPU shard.
- Worker shards write temporary JSON results; the parent validates duplicate/missing videos and restores original input order before aggregation.
- The audit backend reuses upstream `load_video`, the 16-frame middle sampler, resize condition, `DenseCaptioning`, ObjectDet setup, 0.5 detector threshold, and GRiT weight. It uses the wrapper's raw-prediction method solely to retain `Instances.scores`; role representation, identity-first assignment, and signed geometry are the scoring changes.
- `VBENCH_AUDIT_UPSTREAM` may point to another checkout only if its remote, branch, SHA, and dirty state exactly match the locked gate. `VBENCH_AUDIT_GRIT_WEIGHT` selects an existing weight file; no automatic download occurs.

### Paper check

The repository path `VBench1.0_paper.pdf` is a 284-byte 404 HTML response, not a valid PDF. The official [arXiv paper](https://arxiv.org/abs/2311.17982) was therefore checked through its official HTML rendering. It defines Spatial Relationship as evaluating whether synthesized objects' spatial relationship follows the text prompt, focuses on left/right and top/bottom, and uses rule-based evaluation. The paper does not specify the code-level role-pooling or IoU formula; those facts come from the locked source.
