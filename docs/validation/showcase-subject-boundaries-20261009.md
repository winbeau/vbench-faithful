# Subject showcase: tighter per-frame blur boundaries

The courtyard cat, glasshouse macaw and cloister woman now use background
blur that follows the subject independently in each frame. The earlier
temporal union retained every position occupied by the subject across the
clip, leaving substantial clear background around the current silhouette.
Replacing that union with the existing per-frame SAM2 masks substantially
reduces the excess protected area. The originals, blur strength, continuous
0.5–5 second interval and scoring implementation are unchanged.

Fresh inference gives VBench counterfactual scores **0.846956, 0.818317 and
0.768834**. Maximum absolute Ours change is **0.003779**. After seeing the
woman's 0.768834 score, the user explicitly chose to retain that strength
and cancel a proposed lighter version. Only cat and macaw are in the
initially requested 0.8 range; the woman's stronger decline is an accepted
exception, not a claimed range pass.

These are exploratory showcase examples. They do not replace the frozen
paper evidence or establish uniform robustness. The
[complete receipt](showcase-subject-boundaries-20261009.json) and
[all fresh scores](showcase-subject-boundaries-20261009.csv) retain the
original, previous counterfactual and new counterfactual for every case.

## Measured scores

Both backends score the complete five-second browser media. Ours change is
new counterfactual minus original; no score is adjusted or calibrated here.

| Case | VBench original | Previous counterfactual | New counterfactual | Ours original → new counterfactual | Ours change |
|---|---:|---:|---:|---:|---:|
| `23-subject-cat` | 0.970728 | 0.860184 | **0.846956** | 0.976292 → 0.976186 | −0.000106 |
| `25-subject-parrot` | 0.958404 | 0.867993 | **0.818317** | 0.920666 → 0.924445 | +0.003779 |
| `26-subject-woman` | 0.969498 | 0.896534 | **0.768834** | 0.941640 → 0.942274 | +0.000634 |

VBench declines relative to the originals are **0.123771 / 0.140087 /
0.200664**. All nine Ours inputs retain 120/120 scoring-present frames.
The twelve scores on the three originals and three previous counterfactuals
exactly reproduce their earlier records. The independent scoring localizer
receives no construction masks.

## Boundary change

Every active frame now uses its own existing SAM2 mask, a **3 × 3 elliptical
guard** (roughly one pixel), and a three-pixel outward blend into the blurred
background. The former protection was the temporal union of all 120 masks
dilated by 15 × 15. The Gaussian sigma remains **90 / 90 / 160 pixels** for
cat, macaw and woman respectively. Frames 12–119 are edited; frames 0–11
remain complete and unedited before encoding. The five-case Subject row
continues to share the same 0.5–5 second blur interval.

| Case | Previous protected area | New protected area, edited-frame mean | Previous extra area beyond the current mask | New guard area beyond the current mask |
|---|---:|---:|---:|---:|
| Cat | 20.113% | 14.206% | 6.099% | 0.193% |
| Macaw | 25.198% | 9.187% | 16.290% | 0.279% |
| Woman | 35.971% | 13.881% | 22.363% | 0.273% |

Area percentages are fractions of the complete frame. They describe the
predicted construction masks, not measured segmentation accuracy. Paired
contact sheets were inspected at frames 0, 20, 59, 90 and 119; old/new
boundary overlays were inspected at twelve temporal samples. Cat ears,
paws and tail, macaw head, wings and long tail, and the woman's hair, coat,
hand and legs follow their current positions in those sampled views.

The original woman's frame-zero SAM2 mask is incomplete. It lies outside
the edited interval, so the entire frame is preserved; no mask substitution
or scoring feedback is introduced. Small detached mask islands remain in
33 edited woman frames, at most 243 pixels per frame before dilation.
Sampled agent inspection is not human ground truth, and perfect semantic
isolation across all frames is not claimed.

Construction asserts exact protected RGB pixels and exact unedited frames
before encoding. Browser delivery uses H.264 CRF18, 1024 × 640, 24 FPS and
120 frames. Encoding can change protected pixels: mean decoded absolute
error on protected pixels in edited frames is **1.875 / 2.466 / 2.043 RGB
levels** on the 0–255 scale. Lossless construction controls and raw masks
remain under ignored output. The original video bytes are unchanged.

A focused sigma-90 woman follow-up with detached-island cleanup was planned
after the first scores. The user cancelled it in favor of the already
scored version. Its construction process was stopped; no additional GPU
evaluation or lighter video is included in these results or selected for
publication. The cancelled plan remains in the receipt.

## Execution and validation

One fresh launch, `subject-boundary-r1`, completed **18/18 score rows** on
nine unique five-second videos: three originals, three previous edits and
three tightened edits. Total unique input duration is **45 seconds**, scored
by both backends. Evaluation wall time was **128.86 seconds**; this is an
affected-dimension check rather than a full-16 rerun or speed benchmark.

- Host: `H200-target-server`, NVIDIA H200 NVL, driver 580.126.20.
- Physical GPU 3: `GPU-9aa453a3-63f9-ea74-f4ce-7d119e9e69ba`; isolated workers use logical `cuda:0`.
- Clean scoring checkout: `d7a2423038f54d946c81307b0613e81398d3091c`.
- Original VBench: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`.
- Ours: frozen `hybrid / preencode_crop / union / exclude / all_pairs`, detector threshold 0.5, crop margin 0.1, crop side 224.
- Score reuse: disabled. Existing pinned assets and environments were reused.
- Local protocol and summaries: `output/showcase-refine-20261009/subject-boundaries-20261009/`.
- Local candidate media: `output/showcase-refine-20261009/candidates/subject-boundary-v1/`.
- Local inference outputs: `output/showcase-refine-20261009/evaluation/subject-boundary-r1/`.
- Remote root: `/data/chenjiayu/wenbiao_zhao/vbench-showcase-refine-20261009`.

The receipt identifies assets, plans, scripts, source masks and result files
by SHA-256, with each displayed score tied to its exact media file. Selected
videos were checked for 120 decodable frames, 24 FPS and five-second duration
before publication. The website retains previous scores, media identities
and provenance in case history. Other cases and original videos are
preserved, including the separately measured
[cat/SUV Background replacements](showcase-background-replacements-20261009.md).
