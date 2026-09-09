# Scene Implementation Report

## Official audit

The locked VBench 1.0 checkout is `/home/msy625/vbench1`, `master`, commit `13dee903cc97e2633ed6e8f50dea61bc90717935`, with a clean worktree. The local `VBench1.0_paper.pdf` is an invalid 284-byte 404 HTML response; the intended semantic definition was checked against the official VBench 1.0 paper rendering at https://arxiv.org/abs/2311.17982. Scene is a Video-Condition Consistency / Semantics dimension: the generated video should match the intended scene (for example, ocean rather than river).

The exact source path is:

`VBench.evaluate()` -> `init_submodules()` -> `vbench.scene.compute_scene()` -> `load_dimension_info(..., dimension="scene", lang="en")` -> distributed `scene()` -> `load_video(..., num_frames=16, return_tensor=False, width=384, height=384)` -> `tag2text_transform(384)` -> `tag2text_caption.generate()` -> `check_generate()` -> per-video frame success rate -> global dataset success rate.

Official details confirmed from `vbench/scene.py` and `vbench/utils.py`:

- Metadata uses `auxiliary_info.scene.scene.scene` labels in `VBench_full_info.json`; the adapter preserves this contract.
- Tag2Text is initialized with `pretrained=$VBENCH_CACHE_DIR/caption_model/tag2text_swin_14m.pth`, `image_size=384`, and `vit="swin_b"`.
- MP4 frames are decoded by Decord `VideoReader`; `get_frame_indices(..., sample="middle")` uniformly partitions the video, takes interval midpoints, and repeats the last frame for short videos. The target is 16 frames.
- The official evaluator ignores the full prompt during correctness. It splits the auxiliary scene label on spaces and requires every token to occur as a Python substring of the generated caption. There is no threshold, CLIP, ViCLIP, text encoder, classifier, or cosine similarity.
- Frame correctness is Boolean, but the video score is the continuous fraction of successful sampled frames. The dataset score is the global successful-frame count divided by global frame count. Distributed workers gather video records and rank 0 recomputes that same ratio.

## Core contract

Scene correctness is grounded in spatially distributed environmental evidence rather than substitutable foreground objects or isolated local semantic cues.

## Core failure

VBench 1.0 proxies environment verification with generic whole-frame captioning followed by lexical scene-label matching. Because evidence is neither localized nor required to be distributed, a salient object or local cue can substitute for the global environment.

## Repair

Only two repairs are added:

1. **Environment-aware representation:** each sampled frame is scored as one global view plus a deterministic non-overlapping 2x2 grid of spatially distributed regional views.
2. **Spatially distributed support aggregation:** the pure function in `backends/audit.py` computes `env_mean = mean(s_1, ..., s_K)` and `frame_score = s_g * env_mean`.

## Preserved

The official metadata shape, Decord decoding, 16-frame middle sampling, short-video padding, per-frame temporal philosophy, and dataset-facing audit-core inputs/outputs/devices conventions are preserved. The official backend in `backends/vbench.py` is a wrapper and does not apply the repair.

## Modified

The repaired modes replace whole-frame-only representation and lexical-only correctness with a continuous image-text evidence scorer interface. Model-specific code is isolated in `models.py`; the metric does not depend on a concrete encoder implementation.

## Added

- Deterministic non-overlapping spatial views (no segmentation or hard object deletion).
- Continuous global and regional scene support.
- Global-regional spatial aggregation with an explicit product formula.
- Per-frame diagnostics: sampled frame index, global score, regional scores, environment mean, final frame score, scene label, and evaluator mode.

## Ablations

- `official`: locked VBench Tag2Text and lexical frame-rate parity backend.
- `global`: the new semantic scorer on only the whole frame; isolates representation/model change.
- `environment_grounded`: global plus environment views and distributed aggregation; the complete repair.

## Limitations

- Environment crops are generic fixed regions, not true segmentation.
- Scene objects (stove, bed, table, cabinets, road) may be valid environmental evidence as well as foreground content.
- Fixed region geometry can introduce spatial bias.
- Temporal persistence and transition behavior are intentionally not repaired.
- OpenCLIP is optional and real model execution requires the user's CUDA, `open_clip_torch`, checkpoint, and decoder environment; pure logic remains runnable without those dependencies.
