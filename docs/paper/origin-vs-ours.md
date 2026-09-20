# Origin vs Ours — dimension cards (paper material)

Every claim about the original metric below is read from the locked upstream
checkout (`VBench@fd18b3d`, hashes pinned in `configs/upstream.toml`). File and
line references are to that revision. Claims about our repair are marked
*(report)* when they come from `docs/counterfactual-reports/` rather than from
code read for this note.

## The unifying thesis

VBench 1.0 scores are point estimates whose reduction is **not invariant to a
nuisance factor that the dimension is supposed to be blind to**. Each dimension
picks a different nuisance factor:

| dimension | the nuisance factor its reduction is entangled with |
|---|---|
| subject_consistency | **where** in time the change happens (fixed first-frame anchor) |
| dynamics_degree | the **sampling rate** (no time normalisation, fps-dependent decimation) |
| motion_smoothness | the **sampling rate** and the **temporal order** of changes |
| multiple_objects | the **degree** of visibility (binary detection, no confidence) |
| scene | **partial** environmental coverage (per-word substring match) |
| spatial_relationship | the **sign** of the geometry (direction-blind) |
| human_action | the **file name** (the label is parsed from it) |

The audit's contribution is not "a better metric for each dimension" — it is a
**contract per family**: a declared relation between a controlled counterfactual
edit and the score, checked by whether the metric's *ordering* follows the
declaration. A violation localises which nuisance factor the original reduction
is entangled with.

---

## 1. subject_consistency

**Origin.** `vbench/subject_consistency.py:30-69`. Per frame, DINO features are
L2-normalised, and the frame contributes
`(cos(prev, cur) + cos(first, cur)) / 2`, floored at 0. The reported score is
`sim / cnt`, a **global frame-weighted mean over every frame of every video**
(`:68`).

**Problem.** The second term anchors **every** frame to the **first** frame. A
change at the start of the clip therefore destroys `first_image_features` itself
and poisons that term for all later frames, while the same change at the end
moves only the last few. The score is consequently a function of *when* an edit
happens, not only of *what* it is — and the global mean dilutes a short edit by
the clean frames around it. Both are contract violations for a family that
declares the three placements same-rank.

**Ours.** *(report)* `temporal_relocation` places one identical corruption at
start / middle / end and declares the three same-rank; the confirmatory
construction broadcasts a single per-base **median box** to every frame so the
three placements differ only in *when* (`scripts/counterfactual/build.py`,
`VBENCH_AUDIT_SUBJECT_BOX=median`). The report must be read **by contract half**:
sensitivity and position invariance separately.

**Experiment.** P1.5, same 23 bases, both constructions, both backends:
sensitivity saturated at 1.000 everywhere; median relative range of the three
position scores 0.1299 → 0.1017 (Official) and 0.0296 → 0.0215 (Repair); paired
bootstrap CI **includes zero** for both. Official still only 26% within ±5%.
Published headline row: 0.5917 → 0.8500, paired Δ +0.2583 [+0.158, +0.350].

---

## 2. dynamics_degree

**Origin.** `vbench/dynamic_degree.py:42-93`. Frames are decimated by
`interval = max(1, round(fps / 8))` (`:100`). Between consecutive sampled frames
RAFT flow is computed and reduced to `max_rad` = **mean of the largest 5% of
per-pixel displacement magnitudes** (`:50-56`). A frame "moves" if
`max_rad > thres` with `thres = 6.0 * (min(H,W)/256)` (`:61`). The video is
dynamic if at least `count_num = round(4 * n_sampled/16)` frames move (`:84-93`).
**The per-video output is Boolean**, and the reported score is the **fraction of
videos classified dynamic** (`:148`).

**Problem.** Three entanglements. (a) `max(1, ...)` means that for `fps < 8` the
interval stops tracking `fps/8`, so the inter-frame time step is `1/fps` and
grows without bound as the frame rate drops. (b) The displacement is never
divided by the time step, so for constant velocity `max_rad ≈ v·Δt` and the score
tracks the sampling interval rather than the motion. (c) The threshold is in
pixels and the output is Boolean, so all magnitude information above the
threshold is discarded.

**Ours.** *(report)* The shipped v2 repair applies `d / dt**p` with a fitted
diffusive exponent `p = 0.5`, replacing the archived v1 `d/dt` (which mirrored
the violation with `p = −0.511` instead of removing it).

**Experiment.** Counterfactual row 0.8333 → 0.7722, Δ −0.0611 [−0.150, +0.017]
(parity). Signed log-log exponent: Official **+0.491**, v1 **−0.511**, v2
**+0.019**. Independent holdout on 30 prompt- and video-disjoint bases: per-base
`fps2/fps8` median ratio Official 1.8474, α=0 1.9352, **α=0.5 0.9676**, α=1
0.4838; within ±20% 0.133/0.133/**0.200**/0.167. Natural preference set: Official
0.6845 vs Repair v2 0.5690, paired Δ **−0.1155** [−0.1496, −0.0806].

---

## 3. motion_smoothness

**Origin.** `vbench/motion_smoothness.py:111-167`. Every **second** frame is
kept (`extract_frame(..., start_from=0)`, `:63-67`), AMT interpolates the
midpoint between each kept pair, and the score is
`(255 − mean|Δ|) / 255` over the skipped frames (`:152`, `:156-167`). It is an
**interpolation-error** measure, not a motion measure.

**Problem.** The reduction reads absolute pixel differences only. It never reads
the **direction** of the motion, so a local temporal reversal and a
hold-and-jump of similar length are not separated; and the decimation is by a
fixed factor of 2 regardless of `fps`, so the temporal gap being interpolated is
`2/fps` seconds and the score again tracks the sampling rate.

**Ours.** *(report)* The audit repair is a RAFT-flow estimator whose discontinuity
is a weighted sum of a magnitude-change term and a **direction-change** term,
with the temporal aggregation and the direction definition changed at `4d53fa2`
(unaligned pixel-wise direction; top-k k=3; 0.5/0.5 weights).

**Experiment.** Counterfactual row (measured at `feeb770`) 0.8300 → 0.8800,
+0.0500 [−0.015, +0.110] — parity, with strict order 10/20 vs 7/20. Natural
preference set (1440 videos): Official 0.6364 [0.6101, 0.6636] vs Repair
**0.3248** [0.2992, 0.3512], paired Δ **−0.3116** [−0.3473, −0.2760] — **below
chance**, with per-video Spearman between the two scores of only **−0.1032**.
**Both rows describe the pre-`4d53fa2` revision and must be re-scored.**

---

## 4. multiple_objects

**Origin.** `vbench/multiple_objects.py:38-72`. Per frame, GRiT returns a label
set; the frame scores 1 iff **both** named objects are in that set
(`:43-46`). The score is `success_frames / total_frames`, a **global
frame-weighted co-presence rate** (`:72`).

**Problem.** The predicate is **binary and confidence-free**: it reads the label
list, never the detection scores, and it asks for *co-presence* rather than for
*degree of visibility*. A frame where the weaker object is barely visible scores
exactly as much as one where it is fully visible, so the reduction cannot
express how visible the weakest object is — which is the property the dimension
is named for.

**Ours.** *(report)* `weakest_object_visibility` builds a graded occlusion ladder
(occlusion_000…100) plus a never-co-present `conjunction_control`, and declares
the ladder ordered with the control as a separate absolute-low-score predicate
(plan §11.4).

**Experiment.** On the occlusion-only ladder (control removed from the ranked
family, `contract_split` reports `invariance n_pairs = 0`) the row is
**0.5450 → 0.7850, paired Δ +0.2400 [+0.155, +0.330]** — a clean ordered win.
Detection evidence: control ≤ occlusion_100 in 18/25 (Repair) / 22/25 (Official);
weak-target median area ratio 0.1575 with a **62×** spread (0.0066–0.4101);
24 of 49 ranked candidates rejected because GRiT cannot ground both targets.

---

## 5. scene

**Origin.** `vbench/scene.py:29-62`. tag2text captions each of 16 frames at
384×384; a frame succeeds iff **every whitespace-separated word** of the scene
label appears as a **substring** of the caption
(`q in pred` for each `q in key.split(' ')`, `:33`). Score = global frame
success rate (`:62`).

**Problem.** The reduction is a **per-word substring test on free-form text**:
word order and syntax are ignored, stopwords such as "a" match almost any
caption, and there is no notion of *how much* of the environment is present.
Partial coverage therefore cannot be expressed, and the score is a property of
the caption model's vocabulary overlap rather than of the video's environment.

**Ours.** *(report)* `environment_coverage` substitutes a donor environment into
a quadrant of the source clip and declares that the score must drop; the family
is ordered and the donor is recorded in `transformation_parameters` for replay.

**Experiment.** Row 0.3850 → 0.9300, paired Δ +0.5450 [+0.425, +0.680]. **This
must not be quoted as a Repair win**: the two headline numbers are mutually
inconsistent and the informative one is Official's. See `scene.review.md`.

---

## 6. spatial_relationship

**Origin.** `vbench/spatial_relationship.py:25-115`. For each frame, boxes for
the two named objects are collected, and every unordered pair is scored by
`get_position_score`. The frame takes `max` over pairs, with 0 if no pair is
found; the video score is the **global mean frame score** (`:136`).

**Problem.** Three independent defects, all visible in the code:
`abs(x_distance) > abs(y_distance)` (`:69`, `:76`) **never reads the sign**, so
"left of" and "right of" are scored identically; `check_generate` appends boxes
to a single list without recording **which box is A and which is B** (`:105-113`),
so even the intended direction is unavailable; and a frame with no detected pair
scores **0** (`cur_score = [0]`, `:106`), so *detection failure is scored as a
relation violation*. Upstream is therefore direction-blind by construction.

**Ours.** *(report)* `directional_flip` mirrors the clip and declares
`score(original) > score(flip)`; `pick_detectable.py` gained a directional gate
that refuses to build the family unless the relation is confirmed on the source.

**Experiment.** Row 0.3667 → 0.0667, −0.3000 [−0.467, −0.133] — **neither a win
nor a loss**: the premise was never verified on this source, the repair has no
evidence on 79.4% of frames, and 25 of 30 test pairs tie at 0:0.

---

## 7. human_action

**Origin.** `vbench/human_action.py:44-110`. UMT scores 16 frames over
Kinetics-400; the top-5 entries with sigmoid ≥ 0.85 form `cat_ls`; the video
succeeds iff the **action parsed out of the file name**
(`video_path.split('/')[-1]...split("person is ")[-1].split('_')[0]`, `:78`) is
in `cat_ls`. Score = `cor_num / cnt`, a **per-video accuracy** (`:109`).

**Problem.** The ground truth is read from the **file name**, so the score is not
a function of the video alone: renaming the file changes it. Compounding this,
the 0.85 gate makes the outcome brittle at the boundary.

**Ours.** *(report)* `filename_invariance` changes only the file name and
declares the score invariant. The repair reads its target from
`metadata[...]["target_action"]` instead, so under this family its input is
**byte-identical** and its CV of exactly 0 is an identity rather than a
measurement.

**Experiment.** Row 1.0000 / 1.0000, [0.000, 0.000] — **degenerate and
unfalsifiable**: for Official the file name *is* the label, so the invariance
expectation cannot hold; for the repair nothing is perturbed. Mean within-base
CV Official 1.4142 vs Repair 0.0000 (dev 4/5 bases, test 15/20 bases).

---

## What still has to be checked before these go into the paper

1. `motion_smoothness` rows are pre-`4d53fa2` (marked in `CONSOLIDATED.md`).
2. `scene`'s informative number and the required ablation are the subject of an
   open review finding — do not quote the +0.5450.
3. `subject_consistency`'s position-invariance headline rests on a different base
   set and the tracked-box construction; only the same-base P1.5 comparison is
   currently fair.
4. The "ours" column for scenes 2–7 is taken from the reports; the audit code
   paths (`metrics/<dim>/src/<dim>/backends/audit.py`) have not all been re-read
   line by line for this note.
