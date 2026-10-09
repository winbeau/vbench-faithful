# Showcase row timing and rollback — 2026-10-09

All five Subject examples qualify for the same **0.5–5.0 second continuous
background-blur window**. Original VBench counterfactual scores range from
**0.8215 to 0.8965**; the largest absolute Ours change is **0.01087**.
The dog and horse use tighter construction masks and reach 0.8215 and 0.8461
while Ours changes by about +0.0005 in each case.

The proposed slower Background schedules were rejected: every candidate with
a minimum one-second clear/blur segment still scored at least 0.9 in original
VBench. The user's later instruction prioritizes numerical contrast and calls
for rolling back such changes. Existing Background media and their matching
scores are therefore retained. **Kyoto and desert remain unresolved weak
examples**, including after testing faster alternatives; this report does not
claim that every case reached 0.8 or a 0.10 decline.

These are outcome-guided illustrative choices, not held-out validation or a
replacement for the paper's frozen results. The [complete score table](showcase-row-timing-20261009.csv)
retains all 61 new counterfactuals and ten original anchors. The
[receipt](showcase-row-timing-20261009.json) records every run, media identity,
selected construction, rejected schedule and retained website score.
The [earlier experiment](showcase-refinement-20261009.md) remains unchanged.

## Subject selection

Every video lasts five seconds, with 120 frames at 24 FPS. Frames 0–11 are
clear; frames 12–119 have background blur. Each clip has one clear-to-blur
transition, with another transition at the playback loop boundary. Scores
are original → counterfactual; the final column is counterfactual minus
original, not a relative error.

| Case | Official | Ours | Official decline | Ours change |
|---|---|---|---:|---:|
| `06-subject` (Shiba) | 0.941003 → 0.821490 | 0.967848 → 0.968356 | 0.119513 | +0.000508 |
| `23-subject-cat` | 0.970728 → 0.860184 | 0.976292 → 0.972803 | 0.110543 | −0.003490 |
| `24-subject-horse` | 0.953543 → 0.846131 | 0.966364 → 0.966867 | 0.107412 | +0.000503 |
| `25-subject-parrot` | 0.958404 → 0.867993 | 0.920666 → 0.909801 | 0.090411 | −0.010866 |
| `26-subject-woman` | 0.969498 → 0.896534 | 0.941640 → 0.940393 | 0.072964 | −0.001247 |

Dog and horse protect each frame's independent SAM2 foreground mask, dilated
with a 7×7 elliptical kernel. Gaussian sigma 90 already produces the desired
contrast; sigma 160 adds little. Cat, parrot and woman retain the previous
conservative temporal-union mask with a 15×15 dilation kernel. Their sigmas
remain 90, 90 and 160 respectively. The woman's intermittent per-frame mask
failures make that temporal union preferable to a tighter per-frame mask.

Eighteen Subject candidates were measured: six each for dog and horse, and
two each for the other three. Alternatives include a 1–5 second blur window;
the woman's official score is 0.906047 for that window, so it was rejected.
The shared 0.5–5 second window meets the below-0.9 requirement for all five.
Parrot and woman still have declines smaller than 0.10. The parrot's Ours
decline is within the user's explicitly accepted approximate 0.01 range.
The working 0.015 selection screen is not a claim about the seven accelerated
dimensions' separate per-video error contract.

## Background rollback

Twenty-five slower candidates cover five schedules for every Background case:
1–4 second blur, 1–5 second blur, 0–4 second blur, and alternating one-second
blocks in both phases. Every clear and blurred segment is at least 24 frames.
Sigma stays 90. The museum uses the interior 75% of its construction foreground
mask; the other four use the complete mask. The surrounding background is
protected before encoding.

| Case | Lowest official score among slow candidates | Retained website official score | Decision |
|---|---:|---:|---|
| `07-background` (lakeside) | 0.924887 | 0.869865 | Keep existing alternating-frame version |
| `27-background-kyoto` | 0.964386 | 0.980424 | Keep previous version; 0.8 target unresolved |
| `28-background-terrace` | 0.940725 | 0.895820 | Keep existing alternating-frame version |
| `29-background-museum` | 0.905583 | 0.877029 | Keep existing alternating-frame version |
| `30-background-desert` | 0.947165 | 0.965632 | Keep previous version; 0.8 target unresolved |

Nine additional timing/phase candidates each were scored for Kyoto and desert,
including alternating 1/6/12-frame blocks, a 9-blurred/1-clear duty cycle,
first-frame-only blur, first-half blur, and interior-75% alternatives. Their
lowest original scores are **0.938680 / 0.925668**, with Ours scores
**0.991776 / 0.983377**. Neither meets the user's threshold, so these candidates
also stay out of the page. The retained Kyoto/desert scores are not silently
replaced by these scores for different videos.

The existing lakeside, terrace and museum videos remain explicitly labeled
synthetic alternating-frame foreground-blur stress tests. Slower switching
and a common Background row schedule were not achieved under the measured
numerical constraint. Original VBench compares each frame to both the first
and previous frame; these results are consistent with its timing sensitivity,
without establishing a general guarantee about blur duration or phase.

## Construction and execution checks

No original video, scoring formula, model weight, parser, missing-state policy
or selected paper method changed. No construction mask is supplied to the
scoring localizers. Construction asserts exact protected RGB pixels and exact
unmodified frames before encoding. Lossless construction controls remain in
the remote output directory; inference uses the browser H.264 CRF18 media.
Compression can change pixels outside the nominal edited region. The earlier
report records separately measured reencoding controls; they were not rerun
or counted as new evidence here.

The five selected contact sheets were inspected at frames 0, 20, 59, 90 and
119. Predicted construction masks and agent inspection are not segmentation
ground truth. This inspection and the pixel assertions do not prove perfect
semantic isolation on all frames.

All **142/142** score rows succeeded across **71 distinct media inputs**
(23 Subject, 48 Background), comprising 61 new counterfactuals and ten
original anchors. All 20 original-anchor scores across both backends exactly reproduce the earlier
records. Unique input duration is 355 seconds; both backends score each input.

Eight fresh launches ran on **H200-target-server**, H200 NVL physical GPUs
2–5, driver 580.126.20, with each isolated worker seeing logical `cuda:0`.
Per-launch wall times range from **68.19 to 214.93 seconds**. Inference reuse
was disabled; existing pinned model files and environments were reused.
These runs are not a new full-16 evaluation or a speed benchmark.

- Scoring checkout: `d7a2423038f54d946c81307b0613e81398d3091c`.
- Original VBench: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`.
- Local outputs: `output/showcase-refine-20261009/followup-rows/` and
  `output/showcase-refine-20261009/evaluation/row-*/`.
- Remote root: `/data/chenjiayu/wenbiao_zhao/vbench-showcase-refine-20261009`.
- Candidate rounds: `subject-tight-followup-v1`, `subject-row-followup-v1`,
  `background-row-min24-v1`, `background-followup-timing`,
  `background-followup-phase`.

The receipt includes GPU UUIDs, per-launch wall time, input/output locations,
asset/plan/result hashes, complete numerical outcomes and construction script
hashes. Media, model caches and raw worker logs remain outside Git.
