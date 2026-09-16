# subject_consistency — origin vs ours (paper material)

**Thesis.** VBench 1.0's subject-consistency score is a monotone reduction of
whole-frame DINO ViT-B/16 features whose **temporal anchor** entangles it with a
nuisance factor it should be blind to: **where in the timeline an edit happens**.
Source of record: `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
`vbench/subject_consistency.py`, 82 lines,
`source_sha256 = 55f5c9a3e5be543380a709b94712903fa67a76340c3c0cb2414ac090dbd1b78d`
(`configs/upstream.toml`), verified byte-identical on the locked scoring checkout.
All line numbers below are that revision. Ours:
`metrics/subject-consistency/src/subject_consistency/`; counters and statistics:
`scripts/counterfactual/`, frozen tree
`/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/`. Every number in §4
was independently recomputed from that tree with
`/root/wenbiao_zhao/venvs/vbench/bin/python` (statistics only, no GPU).

---

## 1. Origin — the complete reduction, video → scalar

**Input.** One video path. Decoding is the locked whole-frame path
`load_video(path)` (`:47`, from `vbench.utils`), resolution left as decoded,
frame rate ignored — every decoded frame is used. Preprocessing is the official
`dino_transform(224)` (`:37`, `:48`): resize/short-side + centre-crop to 224×224,
ImageNet normalisation. Frames are consumed **in file order**; no timestamps, no
deduplication.

**Per-frame features.** DINO ViT-B/16 runs on each frame separately (`:53`) and
the output is L2-normalised (`:54`):

```text
f_t = normalize( DINO(x_t) ) ∈ R^D,   ‖f_t‖_2 = 1        :53-54
```

**Per-frame score.** The first frame is stored as the anchor and contributes
nothing; every later frame contributes the average of two clamped cosines
(`:55-62`):

```text
a = f_0                                                  :55-56
∀ t ≥ 1:
    s_prev(t)  = max(0, cos(f_{t-1}, f_t))               :58
    s_first(t) = max(0, cos(f_0,     f_t))               :59
    c_t        = ( s_prev(t) + s_first(t) ) / 2           :60
    video_sim += c_t ;  cnt += 1                          :61-62
```

**Within-video scalar.** Written into the per-video record but **not returned**
as the score (`:64`):

```text
S_video(v) = ( Σ_{t=1}^{T-1} c_t ) / (T - 1)
```

**Dataset scalar — what `compute_subject_consistency` actually returns.**
`subject_consistency()` returns `sim_per_frame = sim / cnt` (`:68`), the
accumulator `sim` being summed across the **whole video list** (`:65`) and `cnt`
counting **every frame transition of every video** (`:62`). So for a set
`V = {v_1..v_N}` with `T_i` frames each:

```text
S(V) = Σ_i Σ_{t=1}^{T_i-1} c_t^{(i)}  /  Σ_i (T_i - 1)          :65, :68
```

**Output type.** Real, continuous, unbounded above by 1 (it is a mean of clamped
cosines in [0,1]); no threshold, no binarisation. The returned tuple is
`(S(V), video_results)` where `video_results` carries the per-video `S_video`,
which nothing downstream reads (`:69`, `:78-82`).

**Aggregation weights, stated explicitly.** Within a video every transition
carries equal weight `1/(T-1)`; across videos the weight of video `i` is
**`(T_i - 1) / Σ_j (T_j - 1)`**, i.e. proportional to frame count, *not* `1/N`.
Unmodified VBench generations in this audit are 16 frames (mp4, 8 fps) or 33
frames (CogVideo GIF, 10 fps), so the GIF generator's clips carry roughly 2.1×
the weight of an mp4 clip — a second nuisance axis, measured but not part of the
headline (see §4, "not claimed").

On multiple GPUs `all_results` is the unweighted mean of per-rank
frame-weighted means (`:79-81`), so effective weights depend on shard assignment.
This audit scores each clip in a single-video invocation, so `S(V) = S_video(v)`
holds exactly for us.

**One line of orientation.** All four quantities are means of cosine
similarities; what matters is *which pairs* enter: adjacent (`t-1, t`) and
against a **single fixed frame** (`0, t`).

---

## 2. Origin — what breaks, and why it is the nuisance factor we name

### P1 (code fact, `:59`) — the anchor term is a privileged frame

`S(V)` contains `Σ_t max(0, cos(f_0, f_t))`. Frame 0 enters `T-1` of the
`2(T-1)` cosine terms (50 % of the per-frame budget) plus one adjacent term, while
any interior frame `j ∉ {0, T-1}` enters exactly 3. The reduction is therefore
**not exchangeable** in the frame index.

### P2 (inference from P1) — the score is a function of *when*, not only *what*

Let `corrupt(v, w, k)` replace the `w`-frame window starting at `k` with a fixed
degradation (same mask, intensity and duration), and let
`σ(v) = S_video(v)`. The family's contract is `σ(corrupt(v,w,0)) = σ(corrupt(v,w,k))`
for all interior `k`. Under P1 the number of degraded cosine terms is:

```text
d(0)   = 2(T-1)          # f_0 is degraded, so every anchor term degrades
d(k)   = |N(k)|          # adjacent pairs touching the window: 2w for interior k, w at an end
```

For `T = 16, w = 4`: `d(0) = 30` versus `d(7) = 8`, a **3.75× difference in the
number of affected terms for an identical edit** — and with `T` growing the ratio
`2(T-1)/(2w)` grows without bound. No assumption about DINO's response is needed:
the *support* of the corrupted evidence is position-dependent.

### P3 (code fact, `:64` vs `:68`) — the per-video normalisation is computed and thrown away

`sim_per_images` divides by `T-1` and is stored; the returned value divides by the
*global* transition count. For a single 16-frame clip the two coincide, which is
why this audit's per-clip scores are faithful; over a dataset they do not, and
the returned score then mixes the two nuisances (position and clip length /
shard assignment).

### P4 (code fact, `:58-60`) — the two terms are different estimators of different quantities

`(cos(f_{t-1}, f_t) + cos(f_0, f_t))/2` averages a **local** continuity term with
a **global** term measured against one arbitrary reference. The first is a
first-difference; the second is a distance-to-reference. Their equal weighting is
a free parameter of the reduction, not a property of "consistency": an edit at
`t ≈ 0` moves the *reference itself*, so the same physical change alters the two
terms by different amounts depending on position. This is the mechanism P2 makes
measurable.

### Minimal falsifiable prediction (no model needed)

For any feature sequence, P1/P2 predict
`σ(corrupt(v,w,0)) ≤ σ(corrupt(v,w,k))` for interior `k`, with the gap
monotone in `T`. On the audit's frozen scores the *sign* claim is borne out and
is large: see §4 (Official's median relative position range 0.1017 with identical
boxes, versus 0.0215 for the repair, which does not read frame 0 preferentially).
The strict model-free version (an equality test on synthetic features) is a
unit-test-sized check we have **not** run; it stays a prediction.

---

## 3. Ours — contract, repair operator, and what it does not fix

**Declared contract** (`scripts/counterfactual/transforms.py`, family
`temporal_relocation`; `build.py` maps it to `subject_consistency`):

| level | `expected_rank` | declared relation |
|---|---:|---|
| `clean` | 1 | `score(clean) > score(corrupt_*)` for every placement |
| `corrupt_start` | 0 | the three placements are **same-rank** (must tie) |
| `corrupt_middle` | 0 | ” |
| `corrupt_end` | 0 | ” |

One fixed corruption (Gaussian blur radius 6 + 120° hue rotation over the tracked
subject box, `transforms.py` `_corrupt_region`, `CORRUPTION_INTENSITY`) is placed
in a `w = round(0.25·T)`-frame window at `start = 0`, `middle = (T-w)//2`,
`end = T-w`; the three windows are non-overlapping and identical in content
(`transforms.py` `temporal_relocation`). The confirmatory construction
broadcasts one **per-base median box** to every frame (`VBENCH_AUDIT_SUBJECT_BOX=median`,
default since `61a65bc`), so the three placements differ *only* in when.

**Repair operator** (`metrics/subject-consistency/src/subject_consistency/metric.py`,
`backends/audit.py`, `models.py` — identical backbone, preprocessing and features
to Official, `models.py::OfficialDinoFeatureExtractor`):

```text
L = mean_t cos(f_{t-1}, f_t)          local, unchanged        metric.py:34-36
G = mean_{i<j} cos(f_i, f_j)          symmetric all-pairs     metric.py:26-31, 39-40
Ŝ = 0.5·L + 0.5·G                                            metric.py:47-57
```

**Why it removes P1/P2.** `G` is invariant under permutation of the frame index
(`metric.py._upper_pairs` enumerates every unordered pair exactly once), so no
frame is privileged: an interior window of `w` frames owns `C(T,2) - C(T-w,2)`
of the `C(T,2)` global terms **at every placement**, and the mixed
clean–corrupt pair count `w(T-w)` is identical for `start`, `middle` and `end`.
For a fixed-size window the all-pairs half is therefore *exactly* position-neutral.
`G` is also a genuine trajectory-wide term, and it retains the accumulated-drift
sensitivity the anchor term was reaching for (`local_consistency` vs
`global_pairwise_consistency` on drifting features).

**What it does not fix.** (i) The `0.5·L` half is retained, and `L` counts
boundary-crossing adjacent pairs: an interior window has two boundaries, an
endpoint window one, so corrupted frames own `0.5·(w+1)/(T-1)` of `Ŝ` in the
middle versus `0.5·w/(T-1)` at the ends — a predictable residual of `0.0033`
(T=16) / `0.0016` (T=33) at a 0.1 drop, below the detection floor of every
experiment we have run.
(ii) Whole-frame DINO still responds to background and scale; the repair changes
aggregation, not representation (stated in
`metrics/subject-consistency/IMPLEMENTATION_REPORT.md`). (iii) Nothing here
addresses P3: we score clip-by-clip, so the dataset-level weighting question is
out of scope of the counterfactual family.

---

## 4. Experiment

`counterfactual-vbench`, published at `xjuIcthub/counterfactual-vbench`: this
family is 25 bases / 100 clips (5 dev, 20 test), 100 % coverage on both backends
(`docs/counterfactual-reports/subject_consistency.md`, frozen scores at code
`a044ac9`/report `5c4a1309`). **CPA** is the fraction of ordered level pairs whose
signed score difference matches the declared relation; the deciding interval is
the **paired** cluster bootstrap over `base_id`, and for the same-rank half the
tie-margin CPA saturates and is not quotable alone (AGENTS.md).

**4.1 The composite and its two halves (test, 20 bases, 120 pairs).** Official
`0.5917`, Repair `0.8500`, paired Δ `+0.2583` `[+0.1583, +0.3500]` (frozen
`subject_consistency__cpa.json`; I reproduce the point value exactly; the
Markdown report prints the wrong CI cell for this row — see §5).

| half | pairs | Official | Repair | paired Δ | relation to §3 |
|---|---:|---:|---:|---:|---|
| sensitivity, zero margin | 60 | 1.0000 | 1.0000 | 0.0000 | both detect the corruption (P2's *what*) |
| sensitivity, at each dev margin | 60 | 0.9333 | 0.7833 | −0.1500 | **refutes** "the repair is uniformly better" |
| invariance (three placements tie) | 60 | 0.2500 | 0.9167 | +0.6667 | the repair removes P1/P2 |

The composite is exactly the average of the two half-CPAs: `(0.9333 + 0.2500)/2 =
0.5917`, `(0.7833 + 0.9167)/2 = 0.8500`. **Supported:** the repair trades a sensitivity loss for a large invariance gain.
**Not supported:** "Repair is better on subject consistency" (the sensitivity half
is worse at the calibrated margin), and "position invariance 0.2500 → 0.9167" as a
standalone number (§4.2).

**4.2 Why the invariance half needs the margin-free statistic.** The invariance
CPA is a tie-margin statistic; sweeping the margin over the *same* test pairs
moves it from 0.00 (margin 0) to 1.00 (margin ≥ 0.10) for Repair, and Repair's dev
margin (0.04457) exceeds its own median position spread (0.0279) — so a large part
of the 0.9167 is the margin absorbing the spread, not the score tying. Comparing
both backends at a *shared* margin keeps the composite's sign (Δ +0.1500
`[+0.097, +0.205]` at 0.0139; +0.2333 `[+0.141, +0.321]` at 0.0446) but cuts the
headline by up to 42 %. The margin-free replacement is the plan §7.3 statistic:

| dataset | backend | median `max−min` | median rel. range `(max−min)/mean` | median CV |
|---|---|---:|---:|---:|
| published arm, 25 bases | Official | 0.0856 | **0.1154** | 0.0476 |
| published arm, 25 bases | Repair | 0.0239 | **0.0279** | 0.0118 |
| equal-area arm, 23 bases | Official | 0.0886 | **0.1017** | 0.0461 |
| equal-area arm, 23 bases | Repair | 0.0183 | **0.0215** | 0.0097 |

**4.3 P1.5 — same-base, two-construction control (23 bases, this is the fair
comparison).** Both arms rebuilt from the same 25 candidates; two bases fail GRiT
subject tracking (`truck`, `bird`), leaving 23 × 4 levels scored by both backends
(`subject_confirmatory/`, `subject_tracked/`;
`scripts/counterfactual/subject_position_profile.py`, 10 000 bootstraps).

- *Construction check.* In the equal-area arm all **23/23** bases place
  byte-identical window boxes at the three positions, constant across frames; in
  the tracked control the per-base area ratio has median **1.33**, max **5.08**.
  The tracked-box confound is therefore closed, and P1/P2 can be tested on
  position alone.
- *Sensitivity (supports §3):* `clean >` each corrupted placement in **23/23**
  bases × 2 backends × 2 constructions = 46/46 pairings, all 1.000 — saturated,
  and the exact 95 % interval on 23/23 is `[0.852, 1.000]`, so this half cannot
  discriminate between backends. It is a **ceiling check, not a result**.
- *Position invariance (supports §3):* medians above. Official is **4.7×**
  wider than Repair on the equal-area arm and **4.4×** on the tracked arm; the
  per-base paired difference is +0.0670 `[+0.0324, +0.1209]` (equal-area) and
  +0.0925 `[+0.0703, +0.1297]` (tracked); Official is wider on **21/23** and
  **20/23** bases, exact sign test `p = 6.6e-05` / `4.9e-04`; a cluster bootstrap
  over the 4 generators still excludes zero (`[+0.047, +0.112]`). A control for
  the trivial explanation: **0/23** bases have exactly equal position scores in
  either backend, so Repair is not snapping to ties.
- *Tolerance profile:* Official within ±5 %/±10 %/±20 % of the per-base mean:
  `0.261 / 0.435 / 0.826` (equal-area); Repair `0.783 / 0.957 / 1.000`.
- *The reduction from the construction fix is **not** established:* paired
  equal-area − tracked is −0.0282 `[−0.0625, +0.0135]` (Official) and −0.0081
  `[−0.0227, +0.0132]` (Repair), both including zero; the minimum detectable
  difference at n = 23 is 0.038 / 0.013, and reaching the observed shift would
  need ≈83 / ≈338 bases. The same-base between-backend contrast, by contrast,
  needs only ≈6 bases. **So: "Official is position-dependent and the repair much
  less so" is supported; "the equal-area construction removed the confound and
  that reduced the measured spread" is not.**

**4.4 Natural set.** **No natural-preference measurement exists for this
dimension** (`results/e0/raw_official_scores/` holds frozen E0 official scores
only; no repair natural scores and no paired natural comparison —
P1.1/P1.2 covered dynamics and motion only). Nothing here speaks to
human-preference agreement, and no replacement claim can be made.

**Not claimed:** the pooled CPA as a Repair win; `0.2500 → 0.9167` as a
standalone number (margin- and base-set-specific — on the equal-area arm with its
own dev margin the ordering inverts, Official 0.9825 vs Repair 0.9474); the
dataset-level frame weighting (P3) as measured; any natural-set behaviour.

---

## 5. Consistency with this dimension's review

The verdict of `docs/counterfactual-reports/subject_consistency.review.md`
(adversarial re-review, `2389b5a`) **still stands unchanged**, and this card is
consistent with it on every number:

- *Still standing:* the pooled composite is not a win; the invariance half is
  margin-dependent and must be quoted with the margin-free relative range; the
  equal-area construction closes the tracked-box confound (verified 23/23);
  identity of all published figures on independent recomputation; the two
  rendering defects in `subject_consistency.md` (paired CI printed from
  `paired.zero_margin`; `paired_halves` computed over 25 bases while labelled
  test-only) and the spatial-mode header leak.
- *Superseded by the new data and **not** repeated here:* the earlier review's
  claim that the repair "penalises the corruption least at the start"
  (`middle < end < start`) came from the tracked construction on the old base
  set and is withdrawn; and the "`0.5·local` boundary term" is downgraded to an
  open hypothesis with a predicted effect (0.0033 / 0.0016) below the detection
  floor. This card cites it only as a bounded, untested residual in §3.
- *New here:* the origin-side reduction is written out end to end with
  the dataset-weight law `(T_i − 1)/Σ(T_j − 1)` (§1, P3), and the falsifiable
  prediction of P2 is stated against the frozen dispersion numbers. Neither
  changes the review's verdict; both are origin-side facts the review did not
  need.
