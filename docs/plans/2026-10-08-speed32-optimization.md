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

## Active item: Background Consistency

Profile localization, image encoding and model loading before changing computation. Preserve masks, preprocessing and the selected patch-frame calibration.

## Subsequent queue

After Background Consistency acceptance, handle Subject Consistency. Dynamic already outperforms its original reference; retain its native protocol. Revisit the GRiT group's semantic overhead with the validated generation helper before final acceptance if needed.

For the seven accelerated dimensions, proceed individually in this order: Appearance Style, Imaging Quality, Aesthetic Quality, Temporal Style, Overall Consistency, Motion Smoothness, Temporal Flickering. Shared encoder work may be prepared when working on its first consumer, with other dimensions remaining unchanged until their own acceptance. Mode routing and final 16-dimensional end-to-end acceptance follow the validated implementations.

Five unvalidated parallel drafts (batch helper and four accelerators) were archived byte-for-byte at `/tmp/vbench-speed32/drafts-seven/` with SHA-256 manifest. They are not active or validated code.

## Versions and checks

- `1aa01e2`: both starting baselines.
- `c999242`: current-run GRiT sharing, rendering elision, bounded semantic CPU threads.
- `b2d2727`: fixed cohort, manifest generator and native API timer; measured optimized code.
- Root Python 3.11.14; locked uv environment and CPU torch overlay checks passed; 1,433 tests passed, 3 skipped; all 16 CLI helps passed.
- All 43 checked frozen/pre-existing research files remain byte-identical.
