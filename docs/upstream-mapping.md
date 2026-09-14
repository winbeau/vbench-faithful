# Upstream mapping

## Current source identity

The active source of truth is the official Vchitect checkout described by
[`configs/upstream.toml`](../configs/upstream.toml):

- Remote: `https://github.com/Vchitect/VBench.git`
- Fixed revision: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`
- Default path: sibling `../VBench`; `VBENCH_AUDIT_UPSTREAM` may select an explicit checkout.
- A detached HEAD at the fixed revision is allowed. Runtime checks require the
  expected remote, revision, clean state, and source hashes before import.
- The eight adapters target only root `vbench/` VBench 1.0 modules. Real model
  execution and numerical parity are not verified in this workspace because
  weights/CUDA and some external model builds are unavailable.

## Current dimension entry points

| Metric | Upstream module | Entry point | Current status |
|---|---|---|---|
| dynamic-degree | `vbench/dynamic_degree.py` | `compute_dynamic_degree` | adapter and audit backend present; real parity unverified |
| motion-smoothness | `vbench/motion_smoothness.py` | `compute_motion_smoothness` | adapter present; model runtime parity unverified |
| subject-consistency | `vbench/subject_consistency.py` | `compute_subject_consistency` | adapter present; model runtime parity unverified |
| scene | `vbench/scene.py` | `compute_scene` | adapter and audit variants present; model parity unverified |
| human-action | `vbench/human_action.py` | `compute_human_action` | adapter and explicit-target audit present; model parity unverified |
| spatial-relationship | `vbench/spatial_relationship.py` | `compute_spatial_relationship` | adapter and ordered-role audit present; GRiT parity unverified |
| overall-consistency | `vbench/overall_consistency.py` | `compute_overall_consistency` | adapter and repair backend present; model parity unverified |
| multiple-objects | `vbench/multiple_objects.py` | `compute_multiple_objects` | adapter and audit backend present; model parity unverified |

## Historical reference gate (pre-refactor)

- Reference path: `/home/msy625/vbench1`
- Remote: `https://github.com/msy625/VBench.git`
- Branch: `master`
- Commit: `13dee903cc97e2633ed6e8f50dea61bc90717935`
- Worktree: clean at inspection time
- Submodules: none reported by Git
- Scope: use root `vbench/` VBench dimensions only; do not import `VBench-2.0/`, `vbench2_beta_i2v/`, `vbench2_beta_long/`, or `vbench2_beta_trustworthiness/`.

The following mapping was captured from the older fork and is retained for
provenance only; it is not the active source identity.

## Historical dimension entry points

| Metric | Upstream module | Entry point | Current status |
|---|---|---|---|
| dynamic-degree | `vbench/dynamic_degree.py` | `compute_dynamic_degree` | official adapter and source/time/persistence-aware audit implemented; real CUDA parity pending |
| motion-smoothness | `vbench/motion_smoothness.py` | `compute_motion_smoothness` | adapter pending dependency audit |
| subject-consistency | `vbench/subject_consistency.py` | `compute_subject_consistency` | adapter pending semantic input mapping |
| scene | `vbench/scene.py` | `compute_scene` | adapter pending semantic input mapping |
| human-action | `vbench/human_action.py` | `compute_human_action` | official filename baseline and explicit-target temporal-evidence audit implemented; real CUDA parity pending |
| spatial-relationship | `vbench/spatial_relationship.py` | `compute_spatial_relationship` | official plumbing adapter and ordered-role audit implemented; real GPU parity pending |
| overall-consistency | `vbench/overall_consistency.py` | `compute_overall_consistency` | adapter pending ViCLIP dependency audit |
| multiple-objects | `vbench/multiple_objects.py` | `compute_multiple_objects` | adapter pending dense-caption dependency audit |

## Dynamic Degree mapping

Inspection was repeated against the reference gate above. Only root `vbench/dynamic_degree.py` and its root VBench 1.0 call chain are used; similarly named VBench 2.0 and beta modules are excluded.

### Paper and source facts

The local `VBench1.0_paper.pdf` is a 284-byte 404 HTML response and cannot be rendered as PDF. The official [arXiv paper](https://arxiv.org/abs/2311.17982) was checked through its official HTML rendering. It describes Dynamic Degree as detecting large motion with RAFT, averaging the largest 5% optical-flow strengths to decide whether a video is static, and reporting the proportion of non-static videos. Exact sampling, thresholds, and short-video behavior below are source facts, not paper claims.

The locked source path is:

`compute_dynamic_degree -> DynamicDegree -> OpenCV decode -> interval=max(1, round(fps/8)) -> adjacent sampled pairs -> InputPadder -> RAFT(iters=20, test_mode=True) -> flow magnitude -> largest int(H*W*0.05) pixels -> mean -> threshold=6*min(H,W)/256 -> score>threshold -> count -> count_num=round(4*sampled_count/16) -> video bool -> mean of video bools`.

Python's rounding makes `count_num` zero for some very short sampled sequences. `check_move()` tests `count >= count_num` inside the transition loop, so a two-sampled-frame video can return true even for a zero score. This behavior remains unchanged in `--vbench`. Audit records corrected effective-count information only as compatibility diagnostics; neither count participates in its structured intensity/persistence evidence or optional scalarization.

The direct dependencies are OpenCV, NumPy, PyTorch, EasyDict, tqdm, SciPy through RAFT utilities, and the bundled RAFT source. The root RAFT code carries the BSD 3-Clause license. The existing Things checkpoint is `/home/msy625/.cache/vbench/raft_model/models/raft-things.pth`, size 21,108,000 bytes, SHA-256 `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`.

### Plumbing adaptations

- `dynamic_degree.backends.vbench` validates remote, branch, SHA, clean worktree, and import origin before model loading.
- Runtime workers initialize the locked upstream `DynamicDegree` once and reproduce `infer()` while retaining official transition diagnostics. The opt-in parity test compares the adapter boolean with direct upstream `infer()` on the same model, video, checkpoint, and device.
- Audit reuses the same upstream RAFT architecture, checkpoint loading, InputPadder, and 20-iteration inference. It unpads flow back to the decoded image shape before audit-only affine and normalization logic.
- Audit decoding preserves source frame indices and OpenCV timestamps when strictly increasing, with an explicit nominal-FPS fallback. Sampling remains approximately 8 FPS.
- GPU workers use the shared deterministic round-robin partition and temporary JSON result protocol; the parent rejects duplicate or missing videos and restores input order.

## Historical dependency gate

The reference requirements include unpinned scientific/model packages and `transformers==4.33.2`; its setup check requires a CUDA-enabled PyTorch installation. Dynamic Degree declares the direct dependency names and the upstream NumPy `<2` constraint, but does not invent exact model-library versions. The current environment has Python 3.10 and PyTorch `2.13.0+cu130`, but CUDA/NVML is unavailable. `uv.lock` remains absent pending a verified cross-metric model dependency solution.

## Human Action mapping

Locked path: `compute_human_action -> load_dimension_info -> human_action -> load_video(num_frames=16) -> UMT ViT-L/16 K400 -> sigmoid -> rounded Top-5 >= 0.85 -> exact filename-label match -> video bool -> dataset mean`.

The prompt returned by `load_dimension_info` is discarded at the call site. Official derives its target from the basename and this remains unchanged in `--vbench`. Audit requires explicit `target_action` or an exact K400 prompt, preserves target probability/rank, and adds up to four deterministic temporal windows. Its weighted target-probability mean and threshold coverage remain structured and unsummed because no scalar mapping is calibrated.

The existing UMT checkpoint is `/home/msy625/.cache/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth`, size about 579 MiB, SHA-256 `bfee78a03bf806fbc0c216309e8f92792011b7db6556fa860e9d869c1e69fecd`. No weight was downloaded or copied into Git.

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
- `VBENCH_AUDIT_UPSTREAM` may point to another checkout only if its remote, branch, SHA, and dirty state exactly match the locked gate. `VBENCH_AUDIT_GRIT_WEIGHT` selects an existing weight file. Model constructors outside this adapter may fetch pretrained assets unless the caller supplies local paths and offline settings; this workspace run did not download weights.

### Paper check

The repository path `VBench1.0_paper.pdf` is a 284-byte 404 HTML response, not a valid PDF. The official [arXiv paper](https://arxiv.org/abs/2311.17982) was therefore checked through its official HTML rendering. It defines Spatial Relationship as evaluating whether synthesized objects' spatial relationship follows the text prompt, focuses on left/right and top/bottom, and uses rule-based evaluation. The paper does not specify the code-level role-pooling or IoU formula; those facts come from the locked source.
