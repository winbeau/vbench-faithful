# Scene Audit Plan

## Scope and locked reference

- Reference checkout: `/home/msy625/vbench1`
- Remote: `https://github.com/msy625/VBench.git`
- Branch/SHA: `master` / `13dee903cc97e2633ed6e8f50dea61bc90717935`
- Reference worktree is clean. No VBench 2.0 or beta implementation is used.
- `/home/msy625/vbench1/VBench1.0_paper.pdf` is a 284-byte HTML 404 response, not a valid PDF. The paper definition is therefore checked against the official arXiv rendering of VBench 1.0 and recorded as such.

## Official Scene semantic definition

The VBench 1.0 paper places Scene under Video-Condition Consistency / Semantics and asks whether a generated video is consistent with the intended scene in the prompt; its example is that a prompt for “ocean” should produce ocean rather than river. The paper says the evaluator captions generated scenes with Tag2Text and checks correspondence with the scene description. It does not describe image-text cosine similarity, CLIP, ViCLIP, or a classifier for Scene.

## Confirmed execution path

`VBench.evaluate()` in `vbench/__init__.py` loads dimension metadata, initializes `init_submodules()`, imports `vbench.scene`, and dispatches `compute_scene()`.

1. `vbench.utils.load_dimension_info(json_dir, dimension="scene", lang="en")` filters entries whose `dimension` contains `scene`, reads `prompt_en`, and passes `auxiliary_info["scene"]` through unchanged.
2. Standard metadata in `vbench/VBench_full_info.json` uses the nested shape:
   `auxiliary_info.scene.scene.scene = "alley"` (and analogous labels such as `beach`, `bedroom`, etc.).
3. `vbench.utils.init_submodules(["scene"])` configures Tag2Text with:
   `pretrained=$VBENCH_CACHE_DIR/caption_model/tag2text_swin_14m.pth`, `image_size=384`, `vit="swin_b"`; absent weights are downloaded from the official Tag2Text URL.
4. `vbench.scene.compute_scene(json_dir, device, submodules_dict)` constructs `tag2text_caption(**submodules_dict)`, switches it to eval/device, calls `load_dimension_info`, and distributes prompt dictionaries with the official distributed helpers.
5. `vbench.scene.scene(model, video_dict, device)` decodes every video with `load_video(video_path, num_frames=16, return_tensor=False, width=384, height=384)`. MP4 decoding is via Decord `VideoReader`; `load_video` uses `get_frame_indices(..., sample="middle")`.
6. `get_frame_indices()` uses uniform intervals over the video, takes the midpoint of each interval, and pads short videos by repeating the final sampled frame. The target sample count is 16.
7. Each sampled RGB frame is transformed by `tag2text_transform(384)` (PIL conversion, resize to 384x384, tensor conversion, ImageNet normalization), batched, and sent to `model.generate(..., tag_input=None, return_tag_predict=True)`. Only generated captions are used; the returned tag predictions are ignored.
8. `check_generate(key_info, predictions)` reads `key_info["scene"]`, splits it on spaces, and counts a frame correct only when every token is an exact Python substring of the generated caption (`q in pred`). There is no threshold and no prompt text in this correctness check beyond the auxiliary scene label.
9. Per-video Scene score is `success_frame_count / len(cur_video_pred)`, a continuous frame-success rate in `[0, 1]` (not Boolean). The function also returns `success_frame_count` and `frame_count`.
10. Dataset score is the global count ratio `sum(success_frame_count) / sum(frame_count)` across videos on each rank. For distributed execution, `gather_list_of_dict()` gathers video records and rank 0 recomputes the same global ratio; video records are gathered, not averaged independently.

## Core failure mechanism to test

The official evaluator proxies global environment verification with generic whole-frame caption generation followed by lexical scene-label matching. It does not localize evidence or require spatially distributed environmental support, so a salient foreground object or local scene-associated cue can potentially substitute for global scene configuration.

## Implementation plan after this audit

- Preserve an official backend wrapper with the metadata, Decord/16-middle-frame sampling, Tag2Text configuration, lexical matching, frame/video scoring, and distributed contract above. It will not include repairs.
- Add a model-independent Scene evidence scorer interface in `models.py` and keep model loading there. The repaired path will consume continuous scores rather than captions as its sole correctness representation.
- Add deterministic, generic overlapping views per sampled frame: global plus upper-wide, lower-wide, left-wide, and right-wide views. These are environment-biased views, not segmentation and not object deletion.
- Implement the pure aggregation function `frame_score = global_score * mean(regional_scores)` (with documented score normalization if an encoder requires it). Do not use `max(region_score)`.
- Expose three modes sharing the same runner/inputs/outputs/devices/schema conventions: `official`, `global`, and `environment_grounded`.
- Keep official temporal philosophy: score sampled frames independently, then mean over frames for each video; no temporal persistence repair.
- Emit per-frame diagnostics containing sampled index, scene label, global score, regional scores, environment mean, final frame score, and mode.
- Reuse `公共/audit-core` CLI, input enumeration, output writer, device checks, and sharding conventions; do not modify it unless a backward-compatible shared capability is demonstrably missing.
- Add focused parity, aggregation, local-cue robustness, and CLI smoke tests. Real Tag2Text/encoder parity tests will be conditional and explicitly skipped when CUDA/checkpoints/dependencies are unavailable.

## Audit boundaries

The audit found no CLIP, ViCLIP, text encoder, classifier, or threshold in the official Scene path. The repair therefore targets only environment-aware representation and spatially distributed support aggregation. Temporal persistence, segmentation, object detection, prompt parsing, and unrelated evaluator changes are out of scope.
