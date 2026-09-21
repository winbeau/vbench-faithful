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

The aggregation variant enables no ROI, segmentation, or tracking. A second,
representation-level variant was added afterwards and is documented below; the
two are separate ablations and must not be conflated.

## Subject-localised variant (`--audit-variant subject_masked`)

Rationale: the whole-frame reduction cannot distinguish "the subject changed"
from "the background, lighting, or camera moved". The repair therefore gives the
score an explicit subject evidence unit instead of changing the aggregation
again.

Pipeline:

```text
decoded frames (loaded once, shared by both stages)
  -> text-conditioned localizer, prompted with a phrase, returns per-frame
     instance masks + presence
  -> masks resampled onto the backbone's patch grid as coverage weights
  -> per-frame subject vector = mask-pooled patch tokens, L2-normalised
  -> position-invariant all-pairs cosine, fixed denominator C(T, 2)
```

Design decisions that the contract experiments depend on:

- **The phrase never comes from parsing the prompt.** It is read from an
  explicit metadata field (the official annotations carry `subject_en`);
  a missing phrase fails the sample instead of falling back. This keeps the
  repair from re-inventing the string-matching failure mode of the four
  semantic dimensions.
- **The localizer is swappable.** `SubjectMaskProvider` is a protocol and the
  default implementation reads frozen per-video `.npz` masks
  (`NpzSubjectMaskProvider`), so the localizer runs offline, its output is a
  hashable input recorded in diagnostics (`mask_source`, `mask_source_sha256`),
  and the metric's runtime keeps no localizer dependency. An in-process SAM 3
  adapter (`Sam3SubjectMaskProvider`, per-frame image mode, no tracker) is
  provided for convenience.
- **Frames are decoded once.** The same `load_video` tensor is passed to the
  localizer and to the patch extractor, so masks and tokens cannot come from two
  different decodes.
- **Two instance modes.** `union` pools the union of all matched instances into
  one vector per frame; `mean` pools each instance and averages the vectors.
  `union` needs no cross-frame instance matching.
- **Missing evidence defaults to `zero`, not to exclusion.** A frame without
  subject evidence contributes a zero similarity to every pair it belongs to and
  the denominator stays `C(T, 2)`. Excluding such frames would let a corruption
  that destroys detectability *raise* the score. `exclude` remains available as
  the detection-conditioned bound; the plan's reporting rule is to give both.
- **`max_frames` is a declared deviation**: `None` keeps every decoded frame,
  exactly like the official path. A cap uses VBench-style interval midpoints.

Original proposed contract (the equal-area construction is archived; see the
current full-background protocol below):
`clean > subject-corrupt` on every position; the three positions tie (margin-free
relative range); an equal-area, same-operator **background** corruption moves the
score much less than the subject corruption; camera translation and scale changes
are near-invariant. The last two are the discrimination axes that the current
aggregation repair does not address.

Verified here: the masking, pooling, instance-mode, missing-policy, sampling,
resampling and CLI wiring logic, with synthetic features and masks
(`tests/test_subject_evidence.py`, `tests/test_masked_pipeline.py`), including a
test that a background-only change is invisible to the masked score and visible
to the whole-frame score when the synthetic subject features themselves are
unchanged. This does not establish background-invariant real DINO tokens.

The [2026-09-20 pilot](../../docs/counterfactual-reports/subject_region_discrimination_v2.md)
ran real DINO and independent MobileSAM on H100, with seven human-confirmed
scoring boxes. Full-background blur preserves subject pixels and edits their
entire complement; there is no equal-area claim. Median absolute background
score change was 0.0286 for Official, 0.0316 for the aggregation repair, and
0.0294 for masked repair. The masked stability criterion did not pass, and
subject-corruption sensitivity failed on five of seven clips. The user then
judged only the presenter and swimmer images plausible candidates; the entire
run remains a pilot with explicit image/mask-quality limitations. These numbers
do not establish formal method performance. Frozen E0 parity was not measured
in that pilot; the subsequent all-1440 extension below measures it explicitly.

## Tests

The experimental `--audit-variant subject_isolated` isolates native subject
pixels before resizing or DINO attention, then optionally normalizes a square
subject crop (`--subject-view crop`, default for this explicit variant).
Independent scoring masks, fixed gray fill, 10% crop margin, and aligned mask
projection are recorded in diagnostics. The localizer still sees the original
frame. For a fixed mask, outside-mask pixels cannot affect encoder input;
localization drift is not covered by that conditional guarantee.

The [development ablation](../../docs/counterfactual-reports/subject_isolation_development.md)
replayed all seven pilot clips. Background absolute-change medians were 0.0294
(old masked), 0.0189 (isolation with full framing), and 0.0072 (isolated crop).
The mean and worst case did not improve for the crop variant, and the paired CI
crosses zero. This archived development result does not establish overall
superiority. Uniform full-video blur remains a quality control; temporal subject
change is evaluated separately with the partial-window interventions. Old v2
results and its failed criterion remain intact.

The subsequent [two-candidate run](../../docs/counterfactual-reports/subject_isolation_quality2.md)
froze new human image confirmations and scoring boxes before inference. The crop
method reduced mean background absolute change from Official's 0.014485 to
0.005910; six of six partial-window subject interventions reduced its score.
Both uniform full-video subject-blur scores increased. Coffee masks omit body
parts and n=2 is insufficient for general superiority. All 96 reference and
108 representation records succeeded; the 36 post-pool records replayed exactly.
The affected tests, including frozen-cohort and prompt-leakage checks, report
107 passed. The seven-clip development report remains separate.

The subsequent [official extension](../../docs/counterfactual-reports/subject_official_extension_20260920.md)
scored all 1440 official videos with per-frame automatic COCO boxes prompting
MobileSAM, separately from the human-box protocol. Test preference accuracy is
58.53% for Official, 59.61% for the original aggregation repair, and 48.76% for
the preregistered isolated-crop zero candidate; its paired difference CI is
[-13.57, -5.74] percentage points. The candidate therefore fails the broad
effectiveness test. All 1440 Official and aggregation scores match their frozen
references within 1e-6 after correcting float32-before-Resize preprocessing.
Background construction also covers 288 candidates, with 34 automatic accepts.
Minor contour inaccuracies are tolerated under the user's coarse-localization
standard; representation and missing-evidence effects are evaluated directly.

A later development-selected CLS candidate preserves those coarse masks, uses
conditional frame-pair similarity, and falls back to Official when fewer than
two subject frames exist. Its fixed followup protocol scores all 860 test videos
and 306 background variants successfully. With a common zero tie margin, test
accuracy is 57.05% versus Official's 58.45%; the paired difference CI is
[-3.88, +1.09] percentage points. Full-background mean absolute change is
0.053575 versus 0.060218, but the paired improvement CI crosses zero. This
mitigates the original severe regression without establishing an excellent
overall repair. It is available through `run_subject_cls_candidate.py`; all
intermediate methods and failed criteria remain in the extension report.

Synthetic normalized-feature tests cover Original formula parity, identical features, two-frame equivalence, negative-cosine clamping, local discontinuity, accumulated drift, all-pairs permutation symmetry, short inputs, counterfactual gap contracts, shared representation, and CLI output. These tests validate the method contract; they do not turn L2 normalization, equal weighting, `T < 2` handling, diagnostics, or dataset aggregation normalization into paper contributions. Real DINO parity is intentionally not a required unit test.

## Known limitations

- Whole-frame DINO can still respond to background and scale changes; the
  aggregation variant changes temporal aggregation, not subject representation.
  The masked variant changes the representation; it is a separate ablation.
- The masked variant's score depends on the localizer. A frame where the phrase
  is not found is scored by the missing policy, not treated as ground truth, and
  the localizer used to build the counterfactual masks must not be the one used
  at scoring time or the experiment is self-fulfilling.
- Measurements now include the seven-clip pilot, two human-prompted candidates,
  and all 1440 official natural videos under a separate automatic protocol.
  Natural-set accuracy degrades for the new candidate despite passing score
  parity. Nonempty masks are not ground truth; missing-frame zero penalties
  and representation changes both require further investigation.
- All-pairs similarity has quadratic temporary memory in the number of decoded frames, although the full matrix is not persisted in diagnostics.
- The local DINO repository and ViT-B/16 checkpoint must already exist. The CLI does not download either asset.
- The bundled local paper file is a 284-byte HTML response rather than a valid PDF; exact behavior was therefore grounded in the locked VBench1.0 source. A replacement official PDF could not be fetched in the current network environment.
