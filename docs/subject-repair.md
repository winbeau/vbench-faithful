# Subject representation repair and background-blur dataset

Current protocol: **v2, whole-background complement**, following the user's
2026-09-20 clarification. Preserve the designated subject and blur **every pixel
outside its binary mask**. This is the main dataset. The earlier mirrored-box
proposal is archived in `configs/subject-repair/protocol_mirror_v1.json`; its
rejection results are not evidence against the current construction.

The falsifiable question is whether background blur changes Official more than
the masked repair, while subject corruption still affects the repair. A small
background response alone is insufficient: a constant or failed detector could
also be insensitive. No result is selected to make the repair win.

The [stability goal](plans/2026-09-20-subject-stability-goal.md) prioritizes
background-intervention invariance and subject-change response. Human preference
is auxiliary. The [new experiment report](counterfactual-reports/subject_stability_20260920.md)
records temporal localization candidates on the entire developmental cohort,
including incomplete coverage and negative results. The completed primary run and
the user's subsequent acceptance are reported below.
On the same 60 follow-up clips at the fixed 2× blur strength, an opening-quarter
background edit changes Official by 0.088412 and v5 by 0.006389 on average
(absolute paired differences). A single first-frame background edit gives
0.102377/0.001275. These reused inputs are a mechanism diagnostic.
Subject-response size matters too: the quarter-window control lowers v5 by
0.115544 on average, with 57/60 drops at least 0.05. The single-frame control
lowers it by only 0.031723, with 5/60 reaching 0.05; positive signs alone do not
establish effective sensitivity. Changes at most 0.01 are negligible for this
assessment. Those 60 clips did not meet the quarter-window joint target. The
balanced 720-input official experiment (all 72 prompts, 180 videos per generator)
provided broader verification below. The earlier 1,440-input plan was never run;
the user reduced the cohort before construction started.

The expanded quarter-window run has now completed all 241 numerical constructions
(2,169 variants, no runtime failures) from those 720 candidates. On the fixed 240
primary clips, Official changes from 0.936417 to 0.816382 and v5 from 0.951413 to
0.941724. Their paired mean absolute differences are **0.120035/0.010596**;
the signed mean v5 drop of 0.009689 must not be substituted for the absolute
criterion. Subject corruption lowers v5 by 0.099319 on average, with 221/240 drops
at least 0.05. Background joint success is 74/240, or 70/240 when also requiring
that subject response. The original strict threshold is still missed; after seeing
this result, the user accepted a mean change around 0.01. This practical acceptance
does not alter the preregistered threshold or its stored failure flag. All 479
construction rejections remain reported. The user then requested a quick wrap-up;
the remaining supplemental single-frame scoring was stopped at 204/241 completed
clips, with partial outputs preserved and no full-cohort claim. The
[stop receipt](counterfactual-reports/subject_stability_20260920/official720_user_stop.json)
confirms that the completed primary outputs are unchanged and the owned workers
and waiting postcheck have stopped. This task is closed under the latest user
acceptance and scope. Actual mask and encoder-coverage checks found no omitted frames
in clean, background, or local subject conditions. Full subject corruption remains
unscorable for 91/240 primary clips.

The earlier [official extension](counterfactual-reports/subject_official_extension_20260920.md)
has scored all 1440 natural videos and evaluated 2160 human pairs. On 1290 test
pairs, Official reaches 58.53%, the original aggregation repair 59.61%, and the
automatic-localizer isolated-crop candidate 48.76%. The candidate is not a
successful overall repair. Official and original aggregation scores reproduce
all frozen 1440 values within 1e-6. Background construction was also expanded
to 288 candidates, of which 34 pass automatic gates.

A followup candidate keeps those coarse masks, reads isolated DINO CLS features,
and uses Official only when fewer than two subject frames are available. With
zero tie margin for every method, its test accuracy is 57.05% versus Official's
58.45%. All 34 background bases are scored; mean full-background change is
0.053575 versus Official's 0.060218, with a paired improvement CI crossing zero.
The runner `run_subject_cls_candidate.py` and its separate frozen protocol retain
all negative results. Background stability and subject sensitivity still need
improvement; these reused evaluations do not establish overall superiority.

The user's current practical standard is a roughly correct subject region,
preserving its main parts (about 80% is sufficient). Minor boundary omissions
or background inclusion do not require repeated review or delay experiments.
This is not a measured contour IoU threshold. Full-score evaluation, missing
evidence handling and subject-change sensitivity remain the effectiveness tests.

## Run the measured v5 repair

`--audit-variant subject_hybrid` uses current-frame target detections and a
clip-local anchor only for missing detections. Each actual frame is segmented
independently with MobileSAM. It needs metadata with `subject_en`, the pinned
local detector, MobileSAM and DINO weights, and the existing MobileSAM source
installed in the model environment. No construction masks are supplied.

```bash
subject-consistency --both --audit-variant subject_hybrid \
  --video /path/to/clip.mp4 --metadata /path/to/metadata.json --gpu 0 \
  --subject-detector-checkpoint /root/wenbiao_zhao/models/subject-repair/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
  --subject-mobilesam-checkpoint /root/wenbiao_zhao/models/subject-repair/mobile_sam.pt \
  --dino-repo /root/.cache/vbench/dino_model/facebookresearch_dino_main \
  --dino-weight /root/.cache/vbench/dino_model/dino_vitbase16_pretrain.pth \
  --output /path/to/new-output
```

This variant fixes the tested 0.5 detector threshold, 512 detector input,
subject union, all frames, gray-filled 10% margin crop, all-pairs aggregation,
and exclude policy. Options that would silently change those choices are
rejected. Fewer than two usable subject frames produces an explicit failed
result with coverage diagnostics, not a successful zero. Results include the
current clip's localizer diagnostics and model hashes. `--both` retains the
official backend alongside this repair. The existing default remains the
historical aggregation repair; this option is an explicitly selected candidate.

This evaluates subject consistency; it does not certify natural human-preference
superiority. The dose experiment and the CLI entry share the frozen v5 method,
while their input protocols and evaluation results remain separately recorded.

The [2026-09-20 seven-clip run](counterfactual-reports/subject_region_discrimination_v2.md)
is an archived pilot, not the main experiment. After viewing the images, the
user retained only the presenter and swimmer as plausible image candidates.
The other five are excluded from future main-cohort consideration; the original
scores and denominator are preserved. The presenter's construction mask also
retains audience members, so neither image preference nor a scoring box counts
as construction-mask approval. New raw-image screening precedes construction
review and cohort freezing; no new scores are used to choose candidates. The
full natural extension includes every official video without image-quality filtering.

## Model separation

| Role | Component | Prohibited dependency |
| --- | --- | --- |
| Construction localization | SegFormer-B0 ADE20K + GrabCut | Never used as scoring masks |
| Scoring localization | MobileSAM vit_t, independent human-confirmed clean-frame box | Never reads SegFormer masks or DINO features |
| Representation | DINO ViT-B/16 patch tokens | Never chooses localization prompts or pixels |
| Optional semantic normalization | Text-only DeepSeek / optional shared-Qwen subject LoRA | No video, detections, captions derived from video, or masks |

The human-box protocol above is retained for the two confirmed candidates.
The expanded automatic protocol separately supplies per-frame Mask R-CNN
target-class boxes to MobileSAM and explicitly records `human_confirmed=false`.
Its scoring masks remain independent of SegFormer construction masks; it does
not relabel automatically generated boxes as human confirmations.

Breaking any separation makes the corresponding experiment circular. One
MobileSAM prompt is frozen per **original clip** and reused unchanged on clean,
background-corrupt, and subject-corrupt variants, across all temporal positions.
MobileSAM runs again on each variant. Only zero/exclude reuse that variant's
already-computed scoring masks. Human confirmation is not fabricated by code.

The pinned MobileSAM dense positional grid uses `cumsum(ones)`, which PyTorch
2.5.1 rejects on CUDA in strict deterministic mode. A local runtime adapter
constructs the identical half-integer grid with `arange`; the positional weights
and sparse prompt encoder stay unchanged. CUDA checks compare the complete
positional encoding bit for bit on 64×64, 7×11 and 1×1 grids. The external
MobileSAM checkout is unchanged, and deterministic mode remains enabled.
This component equivalence check is not full model or E0 baseline parity.

Official benchmark subjects come from the complete official `subject_en`
annotation, joined by `prompt_en`; missing subjects are rejected, never inferred
from the first word of a prompt. The frozen 25-base list and the metadata-only
60-base expansion are in `configs/subject-repair/`. Expansion preserves the
published 25, existing prompt splits, and chooses remaining prompts by a fixed
hash. Neither detector scores nor metric scores enter this ordering.

## Pixel contract and formats

Construction consumes the pinned VBench `load_video` decode: all frames, native
resolution, RGB uint8. SegFormer uses 512×512 input. `class_map.yaml` is JSON
syntax (a valid YAML subset), with verified ADE20K label IDs in
`ade20k_labels.json`. Direct categories and the specified castle union are
supported; animal species are not silently collapsed to ADE20K's coarse animal
class. All model assets stay outside the repository, with fixed source revisions
and SHA256 values in `assets.lock.json`.

GrabCut seeds use an 11×11 elliptical erode/dilate kernel, OpenCV RNG seed 0,
four iterations. Remove components smaller than 20 pixels. For person, retain
the largest component and fill internal holes smaller than 200 pixels. Each
decoded frame records area and all applicable rejection reasons. Missing
official subject, unmapped class, unusable seeds, area below 1% or above 50%,
missing input, and insufficient frames remain visible in counts. Frame/base
reason counts overlap and are not summed as if mutually exclusive.

For the current family, the editing masks are:

```text
clean:               no edited pixels
background_corrupt:  1 - construction_subject_mask
subject_corrupt:     construction_subject_mask
```

There is **no equal-area claim** in v2. Centered subjects and other background
objects are allowed. Both interventions use the same Gaussian operator: sigma
18 for person, 12 for other supported subjects, frozen by category rather than
tuned to metric results. Optional mosaic uses block size 22/16 respectively.
Feathering is 1.5 pixels **inward into the editing mask**. A single float64 blend
from the original pixels is rounded with `np.rint`, then converted to uint8;
interventions never accumulate on previously corrupted frames. Every frame is
saved as lossless PNG and decoded again to check exact equality.

These pixel assertions are conditional on the construction mask. SegFormer is
semantic segmentation: multiple boats may be retained, and touching people can
survive person connected-component filtering as one region. The 7 accepted
clips pass the automatic construction rules; their masks have not received a
human contour audit. The actual-frame preview exposes these cases rather than
calling every nonzero pixel the correct designated subject. Human-confirmed
MobileSAM scoring boxes do not certify or repair the construction masks.

The primary temporal condition is **full** (every frame). The original
`w = round(.25*T)` start/middle/end windows are additional controls, reported
separately. Python round uses ties-to-even. A quarter-duration window that does
not fit is rejected rather than widened silently.

```text
construction/
  index.jsonl, summary.json
  masks/<video_uid>.npz       # uint8 masks[T,H,W], other_instances[T,H,W], area[T], metadata_json
  frames/<video_uid>/*.png    # accepted construction decodes only
  manifests/<base_id>.json    # source/video/weight hashes, area, per-frame reasons
dataset/
  index.jsonl, summary.json, protocol.json
  construction_masks/<base_id>.npz
  clips/<base_id>/clean/*.png
  clips/<base_id>/<full|start|middle|end>/<background_corrupt|subject_corrupt>/*.png
  manifests/<base_id>.json    # per-file SHA256, parameters, framewise assertions
score-run/
  scores.jsonl, run.json, statistics.json
  scoring_masks/<clip_id>.npz # independent MobileSAM masks, prompt/model hashes, role=scoring
```

Every derived frame asserts `result[edit_mask == 0] == original[edit_mask == 0]`.
In particular, background blur changes **zero subject pixels**, and subject blur
changes **zero background pixels**. The background itself is intentionally
changed in the background-corrupt level. Outside a temporal window, the entire
frame remains unchanged. NPZ ZIP metadata and entry order are fixed; PNG hashes
and manifests are deterministic within the recorded software environment.
Cross-version OpenCV/codec/GPU bitwise equality is not assumed.

## Scoring and statistics

The scorer uses the existing metric APIs and the same PNG RGB tensor for every
column, with the pinned `dino_transform(224)` short-side resize and no crop.

| Column | Representation and aggregation |
| --- | --- |
| Official / origin | Whole-frame DINO CLS; previous-frame and first-frame cosine, equally mixed |
| Published aggregation repair | Whole-frame DINO CLS; 0.5 adjacent + 0.5 all-pairs, exactly as shipped |
| Masked repair zero | DINO patch tokens pooled inside independent MobileSAM masks, then frame L2 normalization; all pairs, fixed C(T,2) |
| Masked repair exclude | Same masks/tokens; condition on pairs whose frames have subject evidence |

The published repair is **not pure all-pairs**. It is preserved as a separate
control. Masked zero includes clips with zero or one detected frame as score
zero. Exclude is undefined with fewer than two present frames and records that
failure. Coverage is reported separately. Mask projection follows DINO's actual
resized image and stride-16 grid; an unrepresented right/bottom tail is not
stretched into represented patches. GPU masks and patch tokens share a device.

### Development variant: isolate before encoding

`--audit-variant subject_isolated --subject-view crop` adds an explicit
development variant. The previous `subject_masked` implementation remains an
ablation and the shipped default aggregation variant is unchanged.

```text
native RGB frame + independent scoring mask
  -> remove background before any resizing or DINO attention (fixed RGB 128)
  -> optional mask-derived square crop, 10% margin, padding, 224x224
  -> DINO patch tokens + masks transformed with the same crop
  -> mask pooling, frame L2 normalization, all-pairs zero/exclude
```

The mask threshold is .5; absent instances contribute no input pixels. Cropping
preserves aspect ratio and uses fixed gray padding. `--subject-view full` keeps
the native framing as a separate ablation. Parameters are frozen in
`configs/subject-repair/isolation_development_protocol.json` before development
scoring. The localizer still sees the original, unmodified frame; only the
encoder sees the isolated view. No construction support, clean-reference mask,
metric score or DINO feature chooses the crop.

For a fixed scoring mask, changes outside its support produce exactly identical
encoder inputs, even when the encoder mixes information globally. This fixes a
structural weakness of late patch pooling. It is a conditional property:
incorrect masks and masks that change under background blur can still make the
end-to-end score change. Crop boundaries and resampling can also introduce
variation when localization changes. The development ablation explicitly
measures these effects instead of claiming localization invariance.

Uniform subject blur throughout a video is retained as a reported quality
control, not treated as logically guaranteed identity inconsistency. Sensitivity
to temporal subject change is separately evaluated with the existing partial
start/middle/end interventions. This interpretation does not rewrite the old
v2 protocol or its recorded failures. No aesthetic/sharpness bonus is added to
force a consistency metric to satisfy a quality ranking.

```bash
subject-consistency --audit --audit-variant subject_isolated --subject-view crop \
  --video /inputs/clip.mp4 --metadata /inputs/metadata.json \
  --subject-masks /inputs/independent-scoring-masks --subject-missing-policy zero

python -m scripts.counterfactual.diagnose_subject_isolation \
  --dataset output/subject-repair/subject-quality2-background-provisional-20260920 \
  --reference output/subject-repair/subject-quality2-float-reference-20260920 \
  --protocol configs/subject-repair/isolation_quality2_float_protocol.json \
  --dino-repo /models/facebookresearch_dino --dino-weight /models/dino_vitbase16_pretrain.pth \
  --device cuda:0 --output output/subject-repair/new-isolation-development
```

All seven old clips remain in the development ablation, including rejected
image-quality cases; this is not a confirmatory cohort or an automatic adoption
decision. Each exact variant replays its separately generated, hash-verified
MobileSAM masks. Clean masks are not substituted on corrupt variants.
Construction/scoring overlap is measured in a separate diagnostic branch after
scores are computed; it never feeds the encoder or sample selection.

The subsequent [two-candidate diagnostic](counterfactual-reports/subject_isolation_quality2.md)
used new human-reviewed coffee/guitar images and independent human scoring boxes
from `localizer_quality2.jsonl`. `isolation_quality2_protocol.json` fixed the crop
method and both source-prompt-disjoint cases before scoring. Mean background
absolute change was 0.014485 for Official and 0.005910 for the new crop method;
all six partial-window subject interventions lowered its score. These are two
selected development examples with approximate scoring masks. Under the user's
current standard, minor omissions do not block experiments. Uniform full-video
subject blur increased both new-method scores. No overall
superiority claim is made from these two selected videos.

`diagnose_subject_isolation --protocol PATH` validates the frozen cohort, prompt
separation, reference dataset, human localizer hash, and representation parameters.
It verifies the dataset and requires the same DINO weights as the reference run.
The new diagnostic does not rewrite the archived seven-clip results.

The v2 preregistration is copied and hashed into every dataset before scoring.
Primary: median absolute clean-to-background score change on the common
supported bases, plus the paired Official-minus-masked reduction in absolute
change. A preregistered descriptive stability tolerance is 0.02 on the 0–1 score
scale. Report signed changes too, so score increases are not hidden. Bootstrap
entire source-prompt clusters, paired across columns (10,000 resamples, seed
20260920, percentile 95% CI). Do not treat time positions or frames as independent
bases. Fewer than two prompt clusters cannot produce an inferential CI.

Also report `clean > subject_corrupt` and all failures. Secondary
`R = median(drop(background)/drop(subject))` uses the common bases with subject
drop at least .05 in every compared column, separately for zero/exclude. Keep
negative background drops. Report per-column populations and exclusions. The
old equal-area criterion that Official's R CI should contain 1 does **not**
apply to full-background v2. Analysis gates never feed back into base selection.

Rejected/unsupported inputs remain in the preregistered denominator with zero
contribution, alongside a supported-only mean and explicit counts. NOT RUN
columns remain unmeasured; they are not numerical zero. A scorer refuses missing,
duplicated, extra, or nonfinite score records before computing statistics.

## Reproduction

Use a new output directory for each invocation. No command writes `data/`,
`results/`, `splits/`, `runs/`, an existing upstream checkout, or prior outputs.
Model execution needs externally provisioned assets and optional dependencies;
the CPU algorithm tests need none of those weights. The recorded H100 model
environment is distinct from the Python 3.11.14 locked CPU test environment.

```bash
# Join all official subject labels; use --count 60 for the frozen expansion.
uv run --no-sync --group test python -m scripts.counterfactual.prepare_subject_bases \
  --annotations /inputs/annotations/Subject_consistency.json --count 25 \
  --output output/subject-repair/new-bases.jsonl

# In the provisioned model environment; weights are local and no fallback downloads occur.
python -m scripts.counterfactual.generate_subject_masks \
  --bases configs/subject-repair/bases_expanded60.jsonl \
  --video-root /inputs/videos --model-dir /models/segformer-b0-ade20k \
  --device cuda:0 --output output/subject-repair/new-construction

python -m scripts.counterfactual.build_region_discrimination \
  --construction output/subject-repair/new-construction \
  --output output/subject-repair/new-dataset
python -m scripts.counterfactual.build_region_discrimination \
  --verify output/subject-repair/new-dataset

# Preview stored clean/background pairs for every accepted clip.
python -m scripts.counterfactual.review_subject_regions \
  --variants output/subject-repair/new-dataset --output output/subject-repair/new-preview

# Independent clean-frame boxes: open the resulting HTML, confirm once per clip, export.
python -m scripts.counterfactual.review_subject_regions \
  --dataset output/subject-repair/new-dataset --output output/subject-repair/new-review
python -m scripts.counterfactual.freeze_subject_localizer \
  --reviewed /inputs/localizer.reviewed.jsonl --dataset output/subject-repair/new-dataset \
  --output output/subject-repair/new-localizer.jsonl

# Expose the pinned MobileSAM checkout on PYTHONPATH; use the same DINO as Official.
python -m scripts.counterfactual.score_region_discrimination \
  --dataset output/subject-repair/new-dataset --localizer output/subject-repair/new-localizer.jsonl \
  --mobilesam-checkpoint /models/mobile_sam.pt \
  --dino-repo /models/facebookresearch_dino --dino-weight /models/dino_vitbase16_pretrain.pth \
  --device cuda:0 --output output/subject-repair/new-scores
```

Set `VBENCH_AUDIT_UPSTREAM` to the read-only VBench checkout at `fd18b3d`.
`--whole-frame-only` records masked columns as NOT RUN while measuring the first
two columns. `--analyze-scores FILE` recomputes statistics without models. The
local `subject-consistency --audit-variant subject_masked` and generic
`NpzSubjectMaskProvider` remain available; only the experiment scorer enforces
this experiment's independent MobileSAM provenance.

## Semantic data and optional LoRA reuse

`subject_semantics.py` validates the exact four-field schema:

```json
{"subject":"person","phrase":"person in red jacket","count":1,"status":"ok"}
```

The subject vocabulary is the complete official `subject_en` union the verified
ADE20K labels. Phrase tokens must occur in the original prompt in order and be
at most six; count is 1/2/null. None/ambiguous use null subject/count and empty
phrase, and remain unsupported. Structural/extractive validity is not semantic
accuracy; the human review measures that remaining uncertainty.

MovieGen is reused from its pinned official text snapshot. Existing spatial,
objects and action LoRA training records contribute **prompt text and source
groups only**, never their task labels. Union source groups and repeated prompt
text before splitting. Already-trained source groups remain train. Reserve 200
independent held-out groups before teacher calls; rewrites inherit the parent's
split, group and source prompt ID. Generated phrases must also be extractive
from the parent prompt. Invalid outputs and unchanged rewrites are quarantined,
not silently repaired into targets.

The DeepSeek client and request ledger are reused read-only from
`vbench-prompts-compile`. `deepseek.json` freezes a dated release declaration,
expected returned model and system fingerprint, templates, vocabulary, and
request limits. The public API alias is not an immutable checkpoint: replay
saved requests/responses, and reject identity drift. The version declaration is
based on the [official DeepSeek change log](https://api-docs.deepseek.com/updates/).
Only `{"prompt": ...}` text reaches the teacher. The optional `ReusedSubjectHead`
loads a named subject adapter into an already-loaded Qwen/PEFT router, reuses its
generation code and restores the prior adapter; it does not duplicate backbones
or train anything.

```bash
python -m scripts.counterfactual.prepare_subject_semantics \
  --moviegen ../vbench-prompts-compile/data/raw/moviegen \
  --training-prompts /inputs/spatial-train.jsonl /inputs/objects-train.jsonl /inputs/action-train.jsonl \
  --output output/subject-repair/new-prompt-pool
python -m scripts.counterfactual.label_subject_silver \
  --pool output/subject-repair/new-prompt-pool \
  --prompt-compile-root ../vbench-prompts-compile --output output/subject-repair/new-silver
python -m scripts.counterfactual.validate_subject_silver output/subject-repair/new-silver
python -m scripts.counterfactual.review_subject_semantics \
  --pool output/subject-repair/new-prompt-pool --output output/subject-repair/new-human-review
# After independent human review, score the complete reserved subset, including invalid teacher cases.
python -m scripts.counterfactual.review_subject_semantics \
  --pool output/subject-repair/new-prompt-pool --reviewed /inputs/subject_human_review.jsonl \
  --silver output/subject-repair/new-silver/silver.jsonl --output output/subject-repair/new-agreement
```

LLM/human agreement and its Wilson interval are calculated only after all
reserved human reviews exist. Invalid teacher outputs count as disagreement.
The complement of the agreement lower confidence bound is a label-noise proxy
under the human-reference/sampling assumptions, not proof of a global bound.
Optional head predictions are evaluated **only** on that exact human subset.
Without it: agreement **NOT RUN**, head accuracy **NOT RUN**, training **NOT RUN**.

## Validation and limits

```bash
uv lock --check
uv sync --locked --group test
uv pip install --python .venv --index https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
uv run --no-sync --group test pytest tests metrics
```

Tests cover pixel invariance, background complement including central subjects,
archived mirror replay, exact PNG/NPZ/manifest replay, pooling and missing-policy
denominators, DINO grid borders, frozen independent prompts, repeated localization
on corrupted frames, complete score grids, paired prompt bootstrap, extractive
schema, split leakage, and the human-review gate.

DINO self-attention can mix background information into subject patch tokens;
masked pooling does not mathematically guarantee background invariance. SegFormer
and MobileSAM mistakes remain possible. The two historical castle/girl images
are feasibility examples only, not statistical observations in this run.
Natural-preference measurements and frozen E0 parity are measured for all 1440
Subject videos in the [extension report](counterfactual-reports/subject_official_extension_20260920.md);
the new candidate loses preference accuracy. This does not establish parity
for other dimensions or certify masks as human ground truth. No model is trained
or fine-tuned, and no existing research inputs or outputs are rewritten.
