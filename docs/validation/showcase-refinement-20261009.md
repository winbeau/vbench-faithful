# Showcase target and blur refinement — 2026-10-09

Fresh H200 evaluation found a Subject example with an official decline of
**0.1211** and an Ours decline of **0.0032**, and a Background example with an
official decline of **0.1110** and an Ours decline of **0.0038**. The latter uses
synthetic alternating-frame foreground blur. Continuous blur was weaker.
The approved three-object extension also produces **official 1→1, Ours 1→0**
on both existing Multiple Objects pairs.

These are exploratory showcase choices selected after measuring candidates,
not an unbiased cohort estimate or replacements for the frozen paper results.
No scoring formula, selected weight, parser, missing-state policy or paper
method configuration changed. Every measured candidate, including weaker
outcomes, remains in the [complete score table](showcase-refinement-20261009.csv).
The [receipt](showcase-refinement-20261009.json) contains the exact protocol,
video hashes, source identity, devices, coverage, timings and output locations.

## Selected examples

All original and counterfactual media are five seconds, 120 frames at 24 FPS.
Scores below are original → counterfactual; declines are absolute score units.

| Dimension / case | Selected perturbation | Official | Ours | Official decline | Ours decline |
|---|---|---|---|---:|---:|
| Subject / `23-subject-cat` | Background blur, sigma 90, frames 1–119 | 0.970728 → 0.849635 | 0.976292 → 0.973085 | 0.121092 | 0.003208 |
| Subject / `25-subject-parrot` | Background blur, sigma 90, frames 1–119 | 0.958404 → 0.859762 | 0.920666 → 0.909064 | 0.098642 | 0.011602 |
| Subject / `26-subject-woman` | Background blur, sigma 160, middle 108 frames | 0.969498 → 0.896210 | 0.941640 → 0.940457 | 0.073288 | 0.001183 |
| Background / `07-background` (lakeside) | Full construction foreground mask, sigma 90, even frames including frame 0 | 0.980902 → 0.869865 | 0.991908 → 0.988146 | 0.111037 | 0.003762 |
| Background / `28-background-terrace` | Full construction foreground mask, sigma 90, even frames including frame 0 | 0.971992 → 0.895820 | 0.990514 → 0.985739 | 0.076172 | 0.004775 |
| Background / `29-background-museum` | Inner 75% of construction foreground mask, sigma 90, odd frames | 0.948792 → 0.877029 | 0.983632 → 0.978227 | 0.071763 | 0.005405 |

Frame indices are zero based. Alternating-frame blur changes only the selected
foreground pixels on 60 frames; it is an artificial temporal stress test, not
a claim about a natural filming artifact. The parrot decline is close to 0.10,
not at least 0.10. Woman, terrace and museum remain below a 0.10 official decline.

The user accepts an Ours change around 0.01, explicitly including the previously
observed parrot change of about 0.013. Selection uses 0.015 as a working screen
and reports the actual value. This is separate from the seven accelerated
dimensions' official-score relative-error contract. The CSV retains the initial
0.01 screen as well; a failed strict screen is not hidden or relabeled.

For a continuous-blur presentation, lakeside's frames 1–119 give official
**0.980902 → 0.920045** and Ours **0.991908 → 0.990501**. Museum's inner-75%
middle-90% variant gives official **0.948792 → 0.900444** and Ours
**0.983632 → 0.980903**. Neither reaches a 0.10 decline.

## Range and timing sweep

**86 new blur counterfactuals** were measured across the five existing Subject
and five existing Background sources. No new source video was generated.

- Subject round 1: five sources × four variants. Sigma 90 at middle 50%/75%
  duration; sigma 160 at middle 75%/90% duration.
- Subject round 2: cat and parrot × two variants. Sigma 90 at middle 90%
  duration and every frame after the first.
- Background round 1: five sources × seven variants. Sigma 90; full mask at
  middle 25%/50%/75%/90% duration and every frame after the first; inner 50%/75%
  mask at middle 90% duration.
- Background round 2: lakeside, terrace and museum × four timing variants.
  Full-mask sigma 90 blur alternating every 1, 6 or 12 frames, plus a repeating
  9-blurred/1-clear schedule.
- Background round 3: the same three sources × five phase/range variants.
  Full-mask first-frame-only, first-half, and even-frame blur; inner-75% mask
  with either even-frame or odd-frame blur.

The last two Background rounds were chosen after inspecting the first round.
The official implementation averages each frame's similarity to the first
frame and to its predecessor, so changing the phase can change both terms.
This explains the follow-up design; it does not establish uniform gains.
Increasing blur area or duration was not monotonically better on every clip.

Subject construction protects the temporal union of independent SAM2 masks,
dilated with a 15×15 kernel. This conservative union also protects some nearby
background, and avoids exposing the subject when an individual construction
mask fails. Background construction edits only within the original per-frame
foreground mask; it does not expand the edit into the surrounding background.
The 50%/75% subsets retain pixels with the largest interior distance, with a
three-pixel inward blend. Percentages refer to predicted construction masks,
not manually verified semantic area.

Construction asserts exact protected RGB pixels and exact unedited frames
before encoding, and retains lossless RGB controls. Actual evaluation uses the
browser H.264 CRF18 videos and unchanged original bytes. Reencoding controls
were measured separately: maximum absolute official/Ours changes were
**0.000415 / 0.000672** for Subject and **0.003363 / 0.000225** for Background.
Delivery encoding can therefore change pixels outside the nominal edit support.
Selected contact sheets were inspected by the agent; this is not human ground
truth. Construction masks are never supplied to scoring localizers.

## Baseline review

Fresh inference reproduces **34 archived scores exactly**: seven Subject
inputs and ten Background inputs, each with both backends. In particular:

| Input | Official | Ours |
|---|---:|---:|
| Parrot original | 0.958404084965 | 0.920666396618 |
| Parrot old counterfactual | 0.914973441793 | 0.921416282654 |
| Woman original | 0.969497772075 | 0.941640019417 |
| Woman old counterfactual | 0.907934538457 | 0.942104101181 |
| Museum original | 0.948792016807 | 0.983631807215 |
| Museum old counterfactual | 0.941082096901 | 0.978480320351 |

The lower original Subject scores alone do not demonstrate an implementation
error. Both original Subject examples have all 120 frames present and 7,140
pairs. Parrot's adjacent score is about 0.9800 while its minimum pair similarity
is about 0.8421; woman's corresponding values are about 0.9943 and 0.8780.
These diagnostics distinguish the all-pairs repaired score from an adjacent
similarity. No scoring change was made in response to these values.

## Multiple Objects: default repair contract

The default selected repair already parses and checks the complete entity
list, including three or more targets, with adjacent-frame confirmation.
The same production route was used for these GPU results:

| Case | Official target metadata | Ours target query | Removed object | Official | Ours |
|---|---|---|---|---|---|
| `03-objects` | cup and dining table | A cup, a dining table, and a vase are visible. | vase | 1 → 1 | 1 → 0 |
| `19-objects-vineyard` | bottle and dining table | A bottle, a dining table, and a wine glass are visible. | wine glass | 1 → 1 | 1 → 0 |

These are explicitly different target sets. Original VBench unpacks exactly
two entries from `key_info.split(' and ')`; a three-entry string would raise
an error rather than silently check two. The original receives two declared
targets, while Ours checks all three. The clips were already generated; the
new strings are evaluation queries, not claims about their generation prompts.
This demonstrates target-set capacity, not better accuracy on identical targets.

Commit `d27ec22` documents this default contract and adds regression cases for
three/four objects, independence from original metadata, missing evidence and
temporal confirmation. The affected tests passed: **107 passed**. Runtime
scoring code and trained parsers were unchanged.

## Human Action: diagnosed, deferred

The ballet wording issue is in target canonicalization. The recorded parser
actually emits `{"actions":["ballet dancing"]}`; normalization maps that
unsupported alias to `other`. The declared lexical scope guard does not recover
it (`outside_declared_subject_form`). `dancing ballet` resolves normally.
Both wordings have identical video and model-input hashes and identical UMT
top-1 evidence: `dancing ballet`, score **0.9996496438980103**. This is not a
visual-model disagreement. The lexicon, parser and weights remain unchanged
as requested; this case has not been presented as a repaired success.

## Execution and artifacts

All runs use clean scoring checkout
`d7a2423038f54d946c81307b0613e81398d3091c`, original VBench
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, and verified pinned model assets.
The archived evaluator lacked its own `.git`, so its historical plan inherited
an ancestor repository's HEAD. A file audit verifies **504/504 runtime, script
and configuration files** match the isolated d7a2423 checkout exactly; fresh
receipts now identify the correct checkout.

The 17 launches ran on **H200-target-server**, physical H200 NVL GPUs 4–7,
driver 580.126.20, with each isolated worker seeing `cuda:0`. Every result is
fresh (`reuse: false`). All **264 executed score rows** succeeded. Deduplicating
repeated original anchors by dimension and video SHA yields **234 scores for
117 distinct dimension/media inputs**: Subject 36, Background 77, Objects 4.
Each input lasts five seconds. Per-launch wall times range from **59.96 to
236.13 seconds**; the JSON records every wall time and GPU UUID. These are
counterfactual evaluations on a shared host, not a new all-16 speed comparison.

Local experiment root: `output/showcase-refine-20261009/`.

- `selected/<case>/`: checked original/counterfactual symlinks and `scores.json`;
  `selected/index.json` lists all eight selected pairs.
- `candidates/<round>/<case>/<variant>/`: delivery MP4, construction receipt and
  contact sheet. Full lossless controls also remain in the remote experiment.
- `evaluation/<run>/`: immutable input manifest, evaluation YAML, plan, results,
  logs and GPU receipt.
- `measurements.csv`, `measurements.json`, `protocol.json`, `scripts/` and
  `receipts/`: all measured outcomes, construction/evaluation scripts and audits.

The corresponding remote root is
`/data/chenjiayu/wenbiao_zhao/vbench-showcase-refine-20261009`.
Media and environments remain outside Git. Published paper tables and the
existing website score records were not overwritten by this exploration.
