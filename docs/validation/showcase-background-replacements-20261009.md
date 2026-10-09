# Background showcase: newly generated cat and vehicle sources

On 2026-10-09, two newly generated nonhuman source videos replaced the small
human foregrounds in the Kyoto lane and desert arch examples. Both new pairs
use continuous foreground blur from **0.25 to 5 seconds**. Fresh inference on
the exact browser videos gives an original VBench decline above 0.1, with Ours
changes of approximately 0.008 and 0.011. Scoring code, models and policies are
unchanged.

These are illustrations selected after observing candidate scores. They are
not a held-out evaluation, an update to the frozen paper cohorts, or evidence
of uniform robustness. In particular, the blurred cat loses foreground
detection in most frames; the measured score stability does not establish
successful foreground isolation by the scorer.

[Full receipt and exact generation prompts](showcase-background-replacements-20261009.json)
and [all unique-media scores](showcase-background-replacements-20261009.csv)
retain every evaluated candidate. The earlier
[row-timing report](showcase-row-timing-20261009.md) remains unchanged: its
rejected schedules were tested on different, older source videos.

## Selected pairs

All values below score the complete five-second browser video. Decline is
original minus counterfactual, separately for each backend.

| Website case and new foreground | VBench original → counterfactual | VBench decline | Ours original → counterfactual | Ours decline |
|---|---:|---:|---:|---:|
| `27-background-kyoto`, cat | 0.974012 → 0.873090 | 0.100922 | 0.977877 → 0.969630 | 0.008247 |
| `30-background-desert`, SUV | 0.975272 → 0.850434 | 0.124838 | 0.986131 → 0.975265 | 0.010867 |

The selected cat uses Gaussian sigma 30 pixels; the SUV uses sigma 20 pixels.
Both edit the full independent foreground mask, with a three-pixel inward
edge blend, during frames 6–119 at 24 FPS. Frames 0–5 remain unedited before
encoding. There is one continuous 114-frame blur interval, not alternating
clear and blurred frames within the clip. Looping the five-second clip
restarts its initial clear interval.

Both originals and both counterfactuals are replaced together. The website
retains the old scores, media hashes and full provenance in case history;
their original media remain recoverable from the preceding `gh-pages` commit.
The other three Background pairs retain their earlier alternating-frame
edits. This replacement therefore establishes a common schedule for these
two cases, not for the full Background row.

## New source generation and construction

The first frames were created with the built-in **imagegen** tool: a large
ginger-and-white cat in a rainy Kyoto lantern lane and a large red off-road
SUV beside a desert arch. Each complete generated image was resized to
1024 × 640 using Lanczos, without cropping. Exact image prompts, image hashes,
prepared first-frame hashes and video prompts are in the JSON receipt.

Both videos were newly inferred using the pinned LTX-2.5 `DistilledPipeline`,
BF16, no quantization, first-frame conditioning strength 1 and no prompt
enhancement. The model produced 121 frames; only the terminal frame was
dropped for delivery: **120 frames, 24 FPS, 5 seconds**, with no temporal
resampling. Browser media use H.264 CRF18. The two generation runs took
139.05 and 138.39 seconds, respectively.

| Source | Seed | Observed motion in sampled frames | Construction mask area: min / median / max |
|---|---:|---|---|
| Cat | 2026100927 | Turns its head, takes several steps and crouches; more motion than the requested paw shift | 9.73% / 14.74% / 16.09% |
| SUV | 2026100930 | Creeps forward and stops; some generated illumination evolution remains | 23.30% / 26.16% / 27.23% |

Generation source: `9ec55f9f22798a3198d9c923856824821bc3317e`.
Model revision: `2356ce76915d6c48d313d7e8b25900e1dd3abaa8`.
The generation checkout reports an untracked `scripts/` directory; the
actual inference wrapper is hashed separately in both receipts. It is not
represented as a clean generation checkout.

Independent SAM2.1 small construction masks use manually specified initial
boxes and points declared before scoring. The source and weight hashes,
mask hashes and all prompts are retained. Twelve temporal samples were
inspected for each source, plus frames 0, 20, 59, 90 and 119 of the selected
paired contact sheets. This agent inspection is not human ground truth and
does not prove perfect isolation across all frames. Construction masks are
never passed to either scoring backend.

Construction asserts exact equality outside the foreground mask and on
unedited frames before encoding. Browser reencoding can change those
pixels. Mean absolute decoded error outside the construction mask on edited
frames is **1.750 / 1.832 RGB levels** for cat/SUV on the 0–255 scale.

Separately scored no-edit reencoding controls reduce VBench by
**0.003166 / 0.005478** and Ours by **0.000273 / 0.000363** for cat/SUV.
These controls quantify observed reencoding effects; their values are not
subtracted from the selected results.

## Candidate selection and limitations

Round 1 tested eight variants per source: continuous sigma 30/60/90,
interior-75% masks, 24-frame blocks and alternating frames. Round 2 tested
eight more per source, varying lower blur strength and continuous duration.
Round 3 tested one additional SUV candidate, sigma 20 during 0.25–5 seconds,
to match the selected cat schedule. All **33** counterfactuals remain in the
CSV and receipt, including weak and rejected outcomes.

The user requested a VBench counterfactual below 0.9 and accepted an Ours
decline around 0.01. An absolute Ours change of 0.015 was used as a working
screen during exploration, not as a paper acceptance contract. The selected
outcomes have smaller declines than this screen. Selection depended on the
observed scores, continuous timing and inspected appearance.

| Scoring-localizer diagnostic | Cat original | Cat counterfactual | SUV original | SUV counterfactual |
|---|---:|---:|---:|---:|
| Frames with no detected foreground, out of 120 | 0 | **109** | 0 | 0 |

The existing Background policy treats absent detected foreground as
whole-frame background evidence, rather than a missing score. Thus all
scores are defined, but the cat's stability must not be interpreted as
robust cat detection under blur. Its long blur interval and the repair's
all-pairs aggregation can affect stability; this experiment does not isolate
their causal contributions. The SUV retains detected foreground on all
frames, which still does not establish mask accuracy. Per-frame diagnostic
records for both selected pairs are included in the receipt.

## Fresh inference and identities

All **80/80 executed score rows succeeded** across five fresh launches.
There are **37 unique videos and 74 unique video/backend scores**: two
originals, two reencoding controls and 33 counterfactuals. Three repeated
original observations contribute six additional score rows, all exactly
equal to the initial anchor values. Unique media duration is 185 seconds;
executed media duration including repeats is 200 seconds, each scored by
both backends.

| Run | Inputs | Score rows | Physical H200 GPU | Evaluation wall seconds |
|---|---:|---:|---:|---:|
| `replacement-background-cat-r1` | 10 | 20 | 4 | 126.27 |
| `replacement-background-cat-r2` | 9 | 18 | 4 | 121.59 |
| `replacement-background-suv-r1` | 10 | 20 | 5 | 126.37 |
| `replacement-background-suv-r2` | 9 | 18 | 5 | 137.73 |
| `replacement-background-suv-r3` | 2 | 4 | 5 | 80.62 |

Inference ran on **H200-target-server**, NVIDIA H200 NVL, driver 580.126.20.
Isolated workers saw their assigned physical GPU as logical `cuda:0`.
GPU 4 UUID is `GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`; GPU 5 UUID is
`GPU-7d459912-b567-59fb-8abf-57c18713cb84`. Score reuse was disabled for
every run. Existing pinned assets and environments were reused. These are
Background-only model inference runs, not a full-16 rerun or speed benchmark.

- Clean scoring checkout: `d7a2423038f54d946c81307b0613e81398d3091c`.
- Original VBench: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`.
- Assets manifest SHA-256: `27a63580d01d7ff26a99b4b097b40b3ec60288c4752886580946d152726867f7`.
- Ours: frozen `patch_frame_calibrated`, gain 1.75, detector threshold 0.8.
- Local source/media root: `output/showcase-refine-20261009/replacements-20261009/`.
- Local evaluation outputs: `output/showcase-refine-20261009/evaluation/replacement-background-*/`.
- Remote root: `/data/chenjiayu/wenbiao_zhao/vbench-showcase-refine-20261009`.

Saved imagegen assets are `sources/27-background-kyoto-cat-v2/input/source-image.png`
and `sources/30-background-desert-suv-v2/input/source-image.png` under the local
source root. Their exact prompts are in the neighboring `source.json` and
the checked-in receipt; the LTX prompts are in `prompt.txt`. Selected videos
are identified by complete SHA-256 values in the receipt and were checked
for 120 decodable frames, 24 FPS and five-second duration before publication.
Raw media, masks, environments and worker logs remain under ignored output.
