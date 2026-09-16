# multiplt_object — origin vs ours (paper material)

**Thesis.** VBench 1.0's Multiple Objects score is the **frame-weighted mean of a
hard, confidence-free co-presence predicate**. Its reduction is therefore
entangled with the nuisance factor this document names **degree of visibility**:
the score asks *whether* both named objects were detected in a frame, never *how
visible* the weakest of them was. Suppressing the weaker target does not move the
score until the detector's own confidence drops below a threshold it does not
report — and the dimension is named for the completeness of the object set.

Source of record: `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
`vbench/multiple_objects.py`, 88 lines,
`source_sha256 = fa5047aa560158f017e46dea3ae763ed083df08ece63e47f61a7ea4fc55be672`
(`configs/upstream.toml:47-51`), verified byte-identical on the locked scoring
checkout. All upstream line numbers below are that revision. Ours:
`metrics/multiple-objects/src/multiple_objects/`, `scripts/counterfactual/`;
frozen tree `/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/`. Every
number in §4 was recomputed from that tree with
`/root/wenbiao_zhao/venvs/vbench/bin/python` (statistics only, no GPU), and every
"code fact" below was read in the files, not inferred.

---

## 1. Origin — the complete reduction, video → scalar

**Detector.** GRiT ObjectDet via the official loader
`DenseCaptioning.initialize_model_det` (`grit_model.py:16-17`), which calls
`init_demo(..., task="ObjectDet")` (`image_dense_captions.py:88`, default
`confidence_threshold = 0.5`); `setup_cfg` writes that number into
`MODEL.ROI_HEADS.SCORE_THRESH_TEST` and
`MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH` (`:76-77`).

**Input.** One video path per prompt; `load_video(path, num_frames=16)`
(`:56`). Decoding is middle sampling: `get_frame_indices(16, T, sample="middle")`
(`utils.py:69-93`) splits the clip into 16 equal intervals and takes each
interval's midpoint, padding with the last frame when `T < 16`. **Resolution is
untouched** unless `min(h,w) > 768`, in which case the shorter side is scaled to
720 (`:58-61`).

**Per-frame detection.** `get_dect_from_grit(model, video_tensor.permute(0,2,3,1))`
(`:62`). For each frame (`:29-35`):

```text
ret = model.run_caption_tensor(frame)                      :31
if len(ret[0]) > 0:  pred.append(set(ret[0][0][2]))        :32-33
else:                pred.append(set([]))                  :34-35
```

`run_caption_tensor` returns a **one-element tuple** whose sole element is the
caption list from `dense_pred_to_caption_tuple` (`grit_model.py:34-36`;
`image_dense_captions.py:53-63`): each entry is
`(object_description[i], box, object_type)` with
`object_type = predictions["instances"].det_obj.data` (`:60`), itself a copy of
the ObjectDet head's `pred_object_descriptions`
(`grit/modeling/meta_arch/grit.py:38-40`). So `set(ret[0][0][2])` is

```text
L_t = set of ObjectDet **label strings** in frame t      :33
```

**Code fact:** `L_t` is a set of strings; **no confidence value and no box
survives into it.** The detector's own scores are consumed by the ROI head gate
`filter_mask = scores > score_thresh` (`grit_roi_heads.py:442`) before any label
is produced.

**Per-frame label.** `check_generate(key_info, predictions)` (`:38-46`) splits
the prompt's object string on the literal `" and "` (`:40`), strips both keys
(`:41-42`), and counts frames in which **both** appear in the set:

```text
a, b = object_info.split(" and ")                          :40
cur_cnt = Σ_t 1[ a ∈ L_t  ∧  b ∈ L_t ]                     :43-45
```

**Scalars.** `cur_success_frame_rate = cur_cnt / len(pred)` (`:64`, the
per-video value stored in `video_results`); accumulators `success_frame_count +=
cur_cnt` and `frame_count += len(pred)` (`:65-66`); and the returned dataset
scalar (`:72`, re-derived the same way across ranks at `:83-88`):

```text
S_official = ( Σ_v Σ_t 1[ a_v ∈ L_{v,t} ∧ b_v ∈ L_{v,t} ] )
           / ( Σ_v T_v )                                   :72
```

A **frame-weighted mean of a Boolean co-presence indicator**, over all videos of
the prompt list — and the weighting is not uniform per clip: in the
counterfactual tree 114 of 150 clips are 16-frame 8 fps sources and 36 are
33-frame 10 fps CogVideo GIFs, so a 33-frame clip carries 2.06× the weight of a
16-frame one. The per-video `success_frame_rate` (`:64`) is stored but never
returned.

---

## 2. Origin — what breaks, and why it is the nuisance factor we name

### F1 (code fact, `:33` + `:44`) — the predicate is confidence-free

Only `set(ret[0][0][2])` reaches `check_generate`; the ROI scores that decided
membership are gone. Formally, membership is already a thresholded quantity,

```text
a ∈ L_t  ⇔  ∃ instance i in frame t :  label_i = a  ∧  s_i > 0.5   (roi_heads:442 + :33)
```

but `s_i` itself is not observable downstream. Two frames whose weakest object
scores `0.51` and `0.99` are represented by the identical value `1`.

### F2 (inference from F1) — the score is not a function of visibility degree

Let `g_t(a) = max{ s_i : label_i = a }` be the per-target confidence the repair
later recovers. Then

```text
S_official = (1/T) Σ_t 1[ g_t(a) > τ ∧ g_t(b) > τ ],        τ = 0.5
S_repair   = (1/T) Σ_t SoftMin_β( g_t(a), g_t(b) ),         β = 10
```

`S_official` is invariant to any change of `g_t` that does not cross `τ`: it is
**constant on the whole pre-threshold band** and **saturates at 1** once the
weaker target is detected above `τ` — a step function of the quantity the
dimension is named for, whereas `S_repair` is strictly monotone in `min_t g_t`
wherever both targets clear the candidate cut.

### F3 (code fact, `:64`, `:72`) — the frame weighting is all-or-nothing

A frame contributes 1 or 0, so a 16-frame clip with the weaker object fully
visible in 8 frames and absent in 8 scores exactly 0.5 — as does a clip where it
is *almost* invisible throughout yet still detected. This is the "temporal union
is not co-presence" direction of the plan's §11.4 control, and it destroys any
notion of *how much* of the object set is present in a frame.

### Why "suppress to the limit" can still be scored as present

The suppression operator we use replaces the weak target's tracked box with its
own mean colour (`transforms.py:382-419`, `_clamp_box` at `:187-193`), and the
control uses the same operator (`transforms.py:458-464`). Nothing in that edit is
guaranteed to remove the target from the **detector's** view: GRiT reads a
full frame with intact context, and the measured per-target confidence for the
suppressed object stays well above `τ` on part of the tree. Concretely, on the
20 test bases the official score at full suppression (`occlusion_100`) is
**non-zero on 7 bases** (values 0.062 … 0.9375; mean 0.1781, and only 13/20 clips
reach exactly 0.0). Where the label survives, F1–F3 make the score
indistinguishable from the fully visible case. **This is the nuisance factor in
one sentence: the metric gates on a hidden threshold instead of measuring the
visible degree, so an intervention that changes degree without crossing that
threshold is invisible to it.**

### Minimal counterexample and falsifiable prediction

The minimum is the ladder's own endpoints: a clip whose weaker target is a flat
patch of its own mean colour still scores `S_official = 0.9375` (one test base),
i.e. as complete as a fully visible clip. Prediction (testable without a repair):
`S_official` should be **non-monotone in suppression severity** and should have a
non-zero floor; both are observed in §4 (`0.5844 → 0.4938 → 0.4781 → 0.4188 →
0.1781` across severities 0→100%; the `025` and `050` means differ by only
0.0156 while the intervention removes 25–50 % of the target's box).

---

## 3. Ours — contract, visibility degree, and the control's status

**The ladder is ordered and confidence-aware.** `weakest_object_visibility`
(`transforms.py:382-419`) alpha-blends the tracked **weaker** target toward the
mean of its own box at severities `0.0, 0.25, 0.5, 0.75, 1.0`
(`VISIBILITY_LEVELS`, `:379`) and stamps
`expected_rank = round((1 - severity) * 4)` (`:410`), giving levels
`occlusion_000 … occlusion_100` with ranks 4…0. The weaker target is chosen by
the detector's own median box area, not by any score (`build.py:_weaker_target`).

**Where degree enters.** Our repair keeps the same GRiT ObjectDet checkpoint but
lowers only the ROI candidate threshold to 0 (`MultipleObjectsConfig`,
`schemas.py:35-39`: `official_threshold=0.5`, `repair_candidate_threshold=0.0`,
`softmin_beta=10.0`) and captures the pre-text-decoder ROI score by hooking
`fast_rcnn_inference_GRiT` / `forward_object` (`backends/vbench.py:137-184`).
Per frame it takes, per target, the maximum captured score among exact-label
matches (`metric.py:96-117`) and reduces the two with `SoftMin_β`
(`metric.py:69-82`, β = 10); the video score is their mean
(`aggregate_video`, `metric.py:144-152`), and the official decision is
recomputed from a **separate 0.5 pass** on the same frame
(`backends/vbench.py:240-247`; `metric.py:120-141`). That is the `g_t(a) > τ`
quantity F1 cannot see, made first-class.

**The control is no longer a ranked level.** `temporal_conjunction_control`
(`transforms.py:422-455`) suppresses A in the first half and B in the second so
no frame holds both (plan §11.4). It used to be stamped `expected_rank = 0`
(`:447`), tied with `occlusion_100`; `run_dimension.py` now removes it from every
rank-based statistic (`CONTROL_LEVELS`, `:49`; ladder filter, `:1196`) and
reports it through `contract_predicate_stats` (`:347-396`). The reason is argued
in `multiplt_object.review.md` §2: equality between a same-frame full occlusion
and a clip whose halves hold different objects is not a severity tie, and no
continuous score can satisfy a tie criterion at all. The ordered half is hence
the whole composite: `contract_split[*]["invariance"]["n_pairs"] == 0` on both
splits and both backends.

**What we do not fix.** (i) The weak target's median box covers
**0.0066 – 0.4101** of the frame (62× spread, §4), so the same nominal severity
removes wildly different pixel budgets across bases; the ladder is a per-base
relative intervention, not a calibrated one, and the fix is a minimum-area filter
or an area-normalised occlusion. (ii) **24 of the 49 scanned candidates were
rejected** because GRiT could not ground both named targets
(`configs/counterfactual/README.md:57-64`), so the whole dimension is conditional
on the same detector whose confidence we are auditing. (iii) The repair's score
never reaches 0 for an absent target (minimum 0.0473 on the test split), so
absence is not a fixed point of `S_repair` either; the control's mean
(0.2009) sits only 0.043 above full occlusion (0.2436).

---

## 4. Experiment

All figures: `scores/multiplt_object__cpa.json` and the per-clip
`scores/multiplt_object__{official,repair}.jsonl`, recomputed on the host.
Coverage `cpa.json["coverage"]`: dev 30/30, test 120/120, both backends.

### 4.1 Ordered ladder — CPA and paired CI (§3, claim: degree enters the score)

| backend | split | pairs | CPA | 95 % CI |
|---|---|---:|---:|---|
| official | dev | 50 | 0.2200 | [0.0000, 0.5000] |
| official | test | 200 | **0.5450** | [0.4300, 0.6500] |
| repair | dev | 50 | 0.8200 | [0.6400, 0.9800] |
| repair | test | 200 | **0.7850** | [0.6900, 0.8650] |

Paired difference over the same 20 test clusters,
`cpa.json["paired"]["tie_aware"]`: **Δ +0.2400, 95 % CI [+0.155, +0.330]**;
leave-one-out range [+0.2158, +0.2526]; 17/20 bases favour the Repair, 0 favour
Official, 3 tie. **Supports §3**: the confidence channel orders degrees that the
official predicate collapses. **Weakens the strongest reading**: exact-outcome
decomposition of the 200 pairs shows +0.2100 of the delta comes from ordering 42
pairs Official leaves *exactly tied* and −0.1250 from inverting 25 other ties, so
only **+0.0300 is strict-versus-strict discrimination** (see §5).

### 4.2 Contract halves (§3, claim: the ladder carries no tie contract)

`contract_split[backend][split]["invariance"]["n_pairs"] = 0` for all four
combinations; the ordered (sensitivity) half equals the composite above. The
legacy 6-level pool survives only as `paired_halves` (sensitivity
+0.2800 [+0.1886, +0.3771], invariance −0.6000 [−0.8000, −0.4000], 25 bases) —
a different pool that includes the retired control pairing. **Supports §3**; also
the source of the report's mixed-generation defect flagged in §5.

### 4.3 Plan §11.4 control (§3, claim: the control is a level predicate)

`cpa.json["conjunction_control"]`: control ≤ `occlusion_100` in **18/25**
(Repair, 0.72) and **22/25** (Official, 0.88) bases; control means **0.2009**
(Repair) / **0.1275** (Official) against `occlusion_100` means **0.2436** /
**0.1825**. Test frame evidence: the control's mean weakest-target confidence is
**0.1597** against **0.4814** at the clean level. **Supports §3**: both backends
score the control as the least complete clip, what §11.4 asks. **Rebuts** the
retired tie criterion: Official reaches exactly 0.0 on 12/20 `occlusion_100` and
12/20 control clips, the Repair on **0/20** of either — absence is not a fixed
point for either estimator.

### 4.4 Frame evidence — detection profile, Spearman, strict order
(§3, claim: the repair is monotone where the official is not)

Mean joint co-presence by level (identical for both backends, since the Repair
reads the same 0.5 pass): `o000 0.5844 → o025 0.4938 → o050 0.4781 → o075 0.4188
→ o100 0.1781`, control 0.1281. Mean weakest confidence: `0.4814 → 0.4538 →
0.4400 → 0.3816 → 0.1894`, control 0.1597. `cpa.json["order"]`:

| backend | split | mean Spearman | median Spearman | strict order |
|---|---|---:|---:|---:|
| official | test | 0.6204 | 0.7225 | 0/20 |
| repair | test | 0.6550 | 0.9000 | 3/20 |
| official | dev | 0.8008 | 0.8008 | 0/5 |
| repair | dev | 0.7000 | 0.9000 | 2/5 |

Rank-gap match rates (test): Official `0.4500 / 0.5167 / 0.6250 / 0.8500` with
tie rates `0.4250 / 0.3500 / 0.2500 / 0.1000`; Repair `0.7000 / 0.7833 / 0.8750
/ 0.9500`, ties **0.0000** everywhere. **Supports §3** and quantifies F2: the
official predicate is a coarse step function (67/200 test pairs exactly tied),
the Repair strictly finer. **Partially rebuts** "clean ordered win": 17/20 test
bases are not strictly ordered, and on dev the Repair's mean Spearman is below
Official's.

### 4.5 Weak-target area (§3 limitation i)

`cpa.json["weak_target_area"]` (test, from the manifest's per-frame tracked
boxes): n = 20, median **0.15745**, IQR **0.0350 – 0.34463**, range
**0.00659 – 0.41009** → **62.2×**; Pearson(area, per-base Δ) = +0.101. Controls
(frozen tree only): dropping area < 0.02 leaves Δ +0.2375 [+0.1500, +0.3312];
dropping the two bases where Official is constant across all five rungs leaves
+0.2000 [+0.1222, +0.2833]. **Supports** reporting the span as a limitation
while **rebutting** the hypothesis that the win is an artefact of box size; it
leaves open a per-base severity validity gate (2 flat + 3 near-flat bases ≈ 0.04
of the delta).

### 4.6 Detector eligibility (§3 limitation ii)

`configs/counterfactual/README.md:57-64`: `multiplt_object` **scanned 49, kept
25 (5 dev / 20 test), rejected 24**; `subject_consistency` scanned 25, rejected
0. The ranked pool behind that scan is **82 candidates (recomputed here: 33 dev
+ 49 test)**; the published selection is 25/25 bases
(`configs/counterfactual/bases_published.jsonl`). **Caveat for the paper:** the
committed eligibility summary is not reproducible from the current tree — the
same ranked-pool function returns 72 candidates for `subject_consistency`, which
cannot be scanned in 25 with 0 rejections — so quote "82-candidate ranked pool"
as the reproducible figure and "49 scanned / 24 rejected" as the recorded one,
or re-run `pick_detectable.py --summary` before the camera-ready. **Supports**
limitation (ii): the population is the subset of two-object prompts whose both
targets GRiT will name.

**Not claimed:** any natural-preference result (none exists for this dimension);
the composite as a pure ordering gain (§4.4: net +0.0850 of the +0.2400 is
tie-handling and +0.0300 is strict-versus-strict); any
statement about the prompts outside the kept 25.

---

## 5. Consistency with this dimension's review

The adversarial re-review (`docs/counterfactual-reports/multiplt_object.review.md`,
`2389b5a`) was written **under the occlusion-only ladder**, and this card
reproduces it; the specific checks the task asks for:

- **Review §2 (is removing the control justified?).** *Still stands, with one
  caveat this card must carry.* Justified: §11.4 asks for an absolute-level
  predicate, not a tie, and `contract_split` confirms 0 invariance pairs. The
  caveat is quantitative and is in `review §2(b)`: removing the control changed
  the pool, not just the composition — Official fell from 0.5967 (6-level) to
  0.5450, the Repair rose 0.7633 → 0.7850, so **+0.0733 of the published
  increase is re-weighted pair composition**, not new evidence. This card quotes
  only the occlusion-only numbers, per the task's discipline.
- **Review §3 (rank-0 is structural; no confidence rule separates present from
  suppressed).** *Still stands and is re-verified here*: the Repair scores
  exactly 0.0 on 0/20 `occlusion_100` and 0/20 control test clips (min 0.0473),
  Official on 13/20 and 12/20; control mean 0.2009 vs 0.2436. The new
  §4.4/§4.5 material (frame evidence, area span) is the evidence §3 asked for;
  §2's formalisation (F1–F3) is the mechanism statement the review inferred.
- **Review §6 (the composite must not be read as pure discrimination).** *Still
  stands and is now sharper*: the exact-outcome decomposition in §4.1
  (+0.2100 ties repaired, −0.1250 ties inverted, +0.0550 / −0.0250 strict) is
  the quantified version of the review's "42 pairs won by breaking ties". The
  review's fixture caveat (49 scanned / 24 rejected) is reproduced in §4.6, with
  the reproducibility caveat the review did not have.
- **Review §4 (area).** Unchanged: tested and does not survive; the residual
  2-flat/3-near-flat-base issue is a fixture gate, not a scoring fix.
