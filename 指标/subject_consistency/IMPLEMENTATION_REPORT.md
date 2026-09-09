# Subject Consistency Implementation Report

## Official data flow

Ordinary MP4 input is decoded with the locked VBench1.0 `load_video(video_path)` call and therefore uses all decoded frames. VBench's `dino_transform(224)` preprocesses each whole frame. The official `dino_vitb16` model produces one feature per frame, followed by L2 normalization. For every `t > 0`, VBench clamps negative cosine similarities to zero and computes equal-weight previous-frame and fixed-first-frame similarity. The video score is the mean over `t > 0`.

The official backend preserves VBench's dataset aggregation inconsistency: single-GPU results are weighted by the number of frame transitions, while multi-GPU results are averaged over video scores.

## Core failure mechanism

The fixed first frame is a privileged global anchor. Consistency measured against that one position is not symmetric over all temporal positions and can represent accumulated subject drift differently depending on where evidence occurs.

## Original vs Repair

- Original: whole-frame DINO plus previous-frame similarity plus a fixed first-frame anchor.
- Repair: the same whole-frame DINO features plus Local adjacent consistency and symmetric Global all-pairs consistency.

The repaired score uses the simple equal-weight default `0.5 * Local + 0.5 * Global`. This weight is not claimed to be theoretically optimal.

## Preserved / Removed / Added

- Preserved: all-frame MP4 decoding, official DINO preprocessing, DINO ViT-B/16, whole-frame features, L2 normalization, negative-cosine clamp, and explicit adjacent continuity.
- Removed: the privileged fixed-first-frame anchor from the repaired mode only.
- Added: symmetric all-pairs global consistency, computed from one matrix multiplication and the strict upper triangle.

No ROI, SAM2, segmentation, or tracking is enabled. ROI remains outside the main method because earlier fixed-ROI evidence did not consistently preserve subject-corruption sensitivity.

## Tests

Synthetic normalized-feature tests cover official formula parity, identical features, two-frame equivalence, negative-cosine clamping, local discontinuity, accumulated drift, all-pairs permutation symmetry, short inputs, counterfactual gap contracts, shared representation, and CLI output. Real DINO parity is intentionally not a required unit test.

## Known limitations

- Whole-frame DINO can still respond to background and scale changes; the repair changes temporal aggregation, not subject representation.
- All-pairs similarity has quadratic temporary memory in the number of decoded frames, although the full matrix is not persisted in diagnostics.
- The local DINO repository and ViT-B/16 checkpoint must already exist. The CLI does not download either asset.
- The bundled local paper file is a 284-byte HTML response rather than a valid PDF; exact behavior was therefore grounded in the locked VBench1.0 source. A replacement official PDF could not be fetched in the current network environment.
