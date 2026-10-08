# 32-video optimization work record

## Current contract

- Default project mode: the selected nine repairs, retaining their logic; seven additional dimensions use our separately identified accelerated implementations.
- Official mode: pinned original VBench, with its efficient single-process compute path and original reducers.
- Accelerated seven acceptance: every video's absolute error must be at most `max(1e-6, 0.01 * abs(official_score))`; complete identical input coverage is required. This is a numerical agreement gate, not a human-quality claim.
- Establish all-16 timings first, then analyze, implement, validate and commit one dimension before moving on. Object Class, Color, Spatial Relationship and Multiple Objects may be treated together as the GRiT group.
- Keep this file current to retain context. Do not broaden implementation while the active item is unvalidated. Archive unvalidated parallel drafts rather than mixing them into the active change.

## Fixed comparison

Same 32 distinct native videos (64 seconds), 512 dimension–video records; physical H100 GPU 3. [Annotations](../../configs/benchmarks/same32-20261008.json), [full first-stage report](../validation/h100-same32-20261008.md). No media or frozen research results changed.

| Dimension | Original single process, s | Current project, s | Project coverage |
| --- | ---: | ---: | ---: |
| `scene` | 29.69 | 54.19 | 32/32 |
| `human_action` | 9.31 | 32.98 | 32/32 |
| `object_class` | 50.61 | 66.08 | 32/32 |
| `subject_consistency` | 5.95 | 25.78 | 23/32 |
| `background_consistency` | 2.84 | 26.03 | 32/32 |
| `dynamic_degree` | 27.15 | 9.83 | 32/32 |
| `spatial_relationship` | 50.83 | 29.98 | 11/32 |
| `multiple_objects` | 50.69 | 20.45 | 32/32 |
| `color` | 64.90 | 84.02 | 9/32 |
| `motion_smoothness` | 2.86 | 6.44 | 32/32 |
| `temporal_flickering` | 0.25 | 3.63 | 32/32 |
| `aesthetic_quality` | 5.71 | 11.22 | 32/32 |
| `imaging_quality` | 9.53 | 13.68 | 32/32 |
| `temporal_style` | 6.13 | 10.48 | 32/32 |
| `overall_consistency` | 5.97 | 10.70 | 32/32 |
| `appearance_style` | 9.58 | 14.31 | 32/32 |

Totals including setup: original 338.50 s; current project 429.78 s; project before this optimization 554.19 s. The seven non-repair dimensions in this snapshot still call original VBench. Per-dimension project times include isolated worker/environment overhead; original times share a process.

## Accepted: GRiT group

Completed: discard unused rendering; share byte-identical ObjectDet frame outputs within this run, separately from Color DenseCap; preserve all model settings and scoring. Exact GPU equivalence passed for all repaired input records, semantic outputs and evidence. The two spatial/object consumers avoid 1,024 repeated frame forwards.

Profiled 64 frames per protocol on H100: ObjectDet takes 5.36 s, including 1.49 s for a redundant second ROI/text pass. Its two raw heads are identical on all 512 cohort frames. Color takes 6.79 s, including 2.62 s primary text generation; its raw heads differ on all 512 frames and must stay separate. Timers are nested; text is included in ROI time.

`6998024`: reuse an independently copied ObjectDet ROI output only within the same frame, when model identity, decoder settings and feature/proposal identities match. Both raw evidence heads remain present. CPU contracts: 109 passed, 1 skipped. Same-32 H100 acceptance passed: all 128 input records retain exact scores, statuses, semantic outputs and raw visual evidence. Object Class falls from 66.08 to 54.61 s (visual stage 50.95 to 39.11 s); the four sequential stages total 200.52 to 191.88 s. Standalone group wall time including setup is 198.60 s. No previous-run inference hits; 510 duplicate ROI/text forwards removed. Color retains both original heads. Coverage stays 32/11/32/9 in Object/Spatial/Multiple/Color order. [Receipt](../validation/h100-grit-roi-20261008.json).

## Accepted: Scene

Profile: 129 distinct semantic requests, 17 batches at size 8; model loading 6.68 s, inference 13.22 s. Representative visual profiling (64 frames) finds caption decoding dominant; tag encoding itself is only 0.14 s. `fed43db` changes only semantic batch size to 32, keeping order, all requests, greedy decoding, token limits and newline stopping. CPU: 41 passed. H100 same-32: scores, diagnostics, captions and raw semantic outputs exact; 32/32 scored. Semantic stage 21.19 → 13.53 s; whole dimension 54.19 → 48.88 s (visual stage varied 33.00 → 35.34 s). Standalone run including setup 56.66 s. [Receipt](../validation/h100-scene-batch32-20261008.json).

## Accepted: Human Action

Profile: model loading 6.21 s, 32 serial prompt generations 10.87 s. `a1ba5a2` batches generation, then invokes the original router's prediction, repair-v2.1 compiler and scorer on each original request. CPU: 41 passed, 1 skipped. Same-32 H100 acceptance: 32/32, exact raw text, evidence, diagnostics and scores. Semantic stage 19.28 → 10.51 s; whole dimension 32.98 → 24.85 s. Standalone including setup 31.44 s. [Receipt](../validation/h100-action-batch32-20261008.json).

## Reviewed: Background Consistency

Profile: COCO boxes 4.63 s, 398 SAM image encodes 7.36 s, 461 SAM mask predictions 4.06 s, CLIP tokens 1.06 s. Candidate `fe5554c` captured the original SAM image-encoder CUDA kernels without precision/batch changes. All 32 scores and diagnostics match exactly. However, whole-dimension time was 26.40 s versus 26.03 s; a subsequent profile showed 398 graph replays and image-encoder time 6.52 s, only 0.84 s saved. The candidate is withdrawn because the end-to-end gain is unproven. Source/tests are archived in `/tmp/vbench-speed32/drafts-cuda-graph/` and remain recoverable from the commit. [Attempt receipt](../validation/h100-background-graph-20261008.json). Existing repair and thresholds remain unchanged.

## Accepted: Subject Consistency

Profile: SAM image encoding 6.47 s, mask prediction 3.89 s, box detection 3.73 s, DINO patch encoding 3.64 s. The 26 videos entering DINO called it 416 times. `d1c8b4a` batches each video's 16 frames, keeping float32, transforms, all frames and scoring formulas. Legacy callers retain batch size 1. CPU: 104 passed. H100: coverage and failure states remain 23/32; localization/crop diagnostics exact. Scores differ by at most 1.1921e-7; all differing floating diagnostic fields are within 2.9803e-7, checked against a tight 1e-6 absolute numerical tolerance. This is **not bitwise equivalence**. Dimension 25.78 → 24.74 s, standalone with setup 27.27 s. [Profile](../validation/h100-subject-profile-20261008.json), [acceptance](../validation/h100-subject-batch16-20261008.json).

## Accepted: GRiT semantic batching

`99a7e9f` uses the generation helper validated for Human Action on Spatial and Multiple Objects, and independent prompt-only batches on Object Class and Color. CPU: 119 passed, 1 skipped. All 128 input records retain exact scores/statuses, raw semantic text, compiled records and visual evidence. Four stage totals 200.52 → 168.33 s relative to the first full optimized run, or 191.88 → 168.33 s relative to the ROI-only group run. Current Object/Spatial/Multiple/Color: 53.64/17.87/15.55/81.27 s; coverage unchanged 32/11/32/9. Standalone group including setup 174.99 s. [Receipt](../validation/h100-grit-text-batch32-20261008.json).

## Accepted: Appearance Style

`8744f4c`: cache each distinct style's CLIP text embedding within this invocation; batch all original PIL-transformed frames in groups of 32. Keep native logit scaling and frame-weighted aggregation. Same-32 H100: 32/32, **zero per-video score error**, aggregate error 1.94e-16; worker 14.00 → 6.80 s, standalone validation wall 9.77 s. No previous-run inference reuse. [Receipt](../validation/h100-appearance-accelerated-20261008.json). Full CPU suite 1,461 passed, 3 skipped; locked uv sync, CPU overlay and all 16 CLI helps passed. Internal worker `--mode accelerated` and `scripts/benchmark_accelerated_dimension.py` are available; public mode routing is intentionally deferred until the seven implementations pass.

## Accepted: Imaging Quality

`0d7de51`: batch MUSIQ frames in groups of 32 and disable unnecessary autograd. Preserve native FP32, longer-side preprocessing, all frames, child-module modes, per-video 0–100 scores and the /100 aggregate. Same-32: all pass; maximum absolute score error 2.0266e-5 (0–100 scale), maximum relative error 3.4568e-7; worker 14.94 → 6.83 s, standalone wall 9.93 s. CPU: 33 passed. [Receipt](../validation/h100-imaging-accelerated-20261008.json).

## Accepted: Aesthetic Quality

`da6e21c` packs original CLIP-L/14 frames across videos. Profiling then found loading took 5.65 s and all 512 frame encodes only 0.70 s. `4361696` constructs the original CLIP visual tower on meta, copies every original visual weight strictly, preserves mixed parameter dtypes, and omits unused text weights/random initialization. Original transforms, normalized features and aesthetic head remain unchanged. Same-32 H100: 32/32, **zero per-video and aggregate error**; worker 10.55 → 5.38 s, standalone wall 9.02 s. [Profile](../validation/h100-aesthetic-profile-20261008.json), [acceptance](../validation/h100-aesthetic-accelerated-20261008.json).

## Accepted: Temporal Style

`a60b1af` batches four original eight-middle-frame clips and distinct text queries; retains FP32 and native cosine/video-mean scoring. Batching alone passed numerical gates but did not reduce worker time. `af701f5` skips random initialization that strict native checkpoint loading overwrites and omits backward checkpoint wrappers in no-grad inference. `fa20644` avoids rehashing the 1.7 GB ViCLIP checkpoint after the controller has already fully verified it in this invocation, under the existing immutable-runtime contract; standalone calls without that receipt still hash it. Same-32: all pass; worker 11.07 → 8.77 s, standalone wall 13.44 s, no previous-run inference reuse. Video feature batches can be shared within the current run with actual transformed-input hashes and explicit hit/miss receipts. [Profile](../validation/h100-temporal-profile-20261008.json), [acceptance](../validation/h100-temporal-accelerated-20261008.json).

## Accepted: Overall Consistency

`8161f0c` uses the verified common ViCLIP encoder with this metric's own prompts, cosine scores and native mean. Independent fresh inference passes all 32 numerical gates, but its worker was 11.15 s versus 10.32 s official: standalone speedup is not established. A fresh pair test first computes Temporal Style, then Overall Consistency in separate workers sharing only that run's verified video features. The second worker has eight feature-batch hits (all 32 videos), zero misses and takes 6.79 s; both dimensions pass their own official comparisons. No previous-run result cache is used. [Standalone receipt](../validation/h100-overall-standalone-20261008.json), [fresh paired receipt](../validation/h100-viclip-pair-20261008.json). Full CPU suite: 1,469 passed, 3 skipped; locked uv sync and all 16 CLI helps passed.

## Accepted: Motion Smoothness

`122c1fc` batches up to eight AMT interpolation pairs per video, retaining native RGB decode, FP32, resizing/padding, uint8 conversion and odd-frame errors. The native unpaired final odd frame is still excluded. Same-32: all pass; worker 6.31 → 5.30 s, standalone wall 9.93 s. CPU: 47 passed, including unequal lengths and unpaired-tail coverage. [Receipt](../validation/h100-motion-accelerated-20261008.json).

## Active item: Temporal Flickering

Verify the native all-frame pixel metric and implement bounded CPU decoding without unnecessary model/CUDA startup. Check all 32 scores before integrating the public modes.

## Subsequent queue

Dynamic already outperforms its original reference; retain its native protocol. Nine-repair pass is now complete except further changes justified by the final timing comparison. Background retains its existing implementation; Subject's tiny float differences are explicitly recorded above.

For the seven accelerated dimensions, proceed individually in this order: Appearance Style, Imaging Quality, Aesthetic Quality, Temporal Style, Overall Consistency, Motion Smoothness, Temporal Flickering. Shared encoder work may be prepared when working on its first consumer, with other dimensions remaining unchanged until their own acceptance. Mode routing and final 16-dimensional end-to-end acceptance follow the validated implementations.

Five unvalidated parallel drafts (batch helper and four accelerators) were archived byte-for-byte at `/tmp/vbench-speed32/drafts-seven/` with SHA-256 manifest. They are not active or validated code.

## Versions and checks

- `1aa01e2`: both starting baselines.
- `c999242`: current-run GRiT sharing, rendering elision, bounded semantic CPU threads.
- `b2d2727`: fixed cohort, manifest generator and native API timer; measured optimized code.
- Root Python 3.11.14; locked uv environment and CPU torch overlay checks passed; 1,433 tests passed, 3 skipped; all 16 CLI helps passed.
- All 43 checked frozen/pre-existing research files remain byte-identical.
