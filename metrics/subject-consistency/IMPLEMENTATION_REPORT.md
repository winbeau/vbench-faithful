# Subject Consistency Implementation Report

## Official data flow

Ordinary MP4 input is decoded with the locked VBench1.0 `load_video(video_path)` call and therefore uses all decoded frames. VBench's `dino_transform(224)` preprocesses each whole frame. The official `dino_vitb16` model produces one feature per frame, followed by L2 normalization. For adjacent pairs `S[0,1], S[1,2], ..., S[T-2,T-1]`, VBench clamps negative cosine similarities to zero and combines adjacent-pair and fixed-first-frame similarities with equal weights. The video score is the mean over the `T-1` transitions.

The official backend preserves VBench's dataset aggregation inconsistency: single-GPU results are weighted by the number of frame transitions, while multi-GPU results are averaged over video scores.

## Core failure mechanism

The fixed first frame is a privileged global anchor. Consistency measured against that one position is not symmetric over all temporal positions and can represent accumulated subject drift differently depending on where evidence occurs.

## Original vs Ours

The method relationship is an aggregation change over the same decoded frames and whole-frame DINO representations:

```text
Original:
L = mean adjacent-pair similarity
A = mean first-frame-to-all similarity
S_original = 0.5 * L + 0.5 * A

Ours:
L = mean adjacent-pair similarity
G = mean all-pairs similarity
S_ours = 0.5 * L + 0.5 * G
```

For `L`, the adjacent pairs are `S[0,1], S[1,2], ..., S[T-2,T-1]`. For `G`, the implementation uses the strict upper triangle of the similarity matrix, so every unordered frame pair is included exactly once. Thus, Local adjacent consistency `L` is preserved, the fixed first-frame anchor term `A` is removed, and symmetric trajectory-wide all-pairs global consistency `G` is added.

One-sentence description: “Preserve local continuity, replace the privileged first-frame anchor with symmetric trajectory-wide pairwise consistency.”

中文：保留相邻帧局部连续性，仅将固定首帧锚定替换为对整段视频所有帧对称的全局一致性。

The `0.5 / 0.5` weighting is the current simple default; it is not presented as a separate contribution or as theoretically optimal.

## Preserved / Removed / Modified / Added

| Component | Status | Scope / explanation |
| --- | --- | --- |
| Video input | Preserved | Same decoded video input. |
| Frame decoding | Preserved | Uses the locked all-frame `load_video(video_path)` path. |
| DINO preprocessing | Preserved | Uses the official `dino_transform(224)`. |
| Whole-frame DINO representation | Preserved | Uses the official `dino_vitb16` representation. |
| L2 normalization | Preserved | Kept as part of the representation pipeline, not a contribution. |
| Local adjacent evidence | Preserved / reorganized | `L` uses adjacent pairs `S[0,1]` through `S[T-2,T-1]`. |
| First-frame anchor | Removed | `A` is not used by Ours. |
| Global all-pairs consistency | Added | `G` uses the strict upper triangle, with no privileged frame. |
| Temporal aggregation | Modified | Replaces `A` with `G` while retaining `L`. |
| Video-level score structure | Modified | `S_original` becomes `S_ours`. |
| Dataset aggregation | Modified at implementation level only, not method contribution | Official single-GPU transition weighting and multi-GPU video averaging are preserved for parity; Ours uses arithmetic mean over successful video scores. |

No ROI, SAM2, segmentation, or tracking is enabled. ROI remains outside the main method because earlier fixed-ROI evidence did not consistently preserve subject-corruption sensitivity.

## Tests

Synthetic normalized-feature tests cover Original formula parity, identical features, two-frame equivalence, negative-cosine clamping, local discontinuity, accumulated drift, all-pairs permutation symmetry, short inputs, counterfactual gap contracts, shared representation, and CLI output. These tests validate the method contract; they do not turn L2 normalization, equal weighting, `T < 2` handling, diagnostics, or dataset aggregation normalization into paper contributions. Real DINO parity is intentionally not a required unit test.

## Known limitations

- Whole-frame DINO can still respond to background and scale changes; the repair changes temporal aggregation, not subject representation.
- All-pairs similarity has quadratic temporary memory in the number of decoded frames, although the full matrix is not persisted in diagnostics.
- The local DINO repository and ViT-B/16 checkpoint must already exist. The CLI does not download either asset.
- The bundled local paper file is a 284-byte HTML response rather than a valid PDF; exact behavior was therefore grounded in the locked VBench1.0 source. A replacement official PDF could not be fetched in the current network environment.
