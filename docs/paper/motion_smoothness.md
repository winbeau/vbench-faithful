# motion_smoothness — origin vs ours (paper material)

**Thesis.** VBench 1.0's Motion Smoothness score is the mean absolute error of an
AMT midpoint reconstruction on a **fixed stride-2 decimation** of the timeline. It
therefore reads two nuisance factors it should be blind to: the **temporal sampling
rate** (the reconstructed interval is `2/fps` seconds, and `fps` is never read) and
the **temporal direction/order** (the functional is a non-negative pixel error, so a
reversal is sensed only through the boundary displacement it creates, never as a
reversal). The audit repair replaces reconstruction error with direction-aware
optical-flow continuity, which satisfies the injected ordering contract but — on the
natural preference set — inverts the human signal, because its scale-free
acceleration ratio rewards frame-to-frame change that annotators penalise.

Source of record: `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
`vbench/motion_smoothness.py`, 192 lines,
`source_sha256 = 5ddc43f246b64e23d10540b8f9e6c302686e6a1ac9fa4e9942b5bd043d89e0b9`
(`configs/upstream.toml`), verified on the locked scoring checkout. All `vbench/`
line numbers below are that revision. Ours:
`metrics/motion-smoothness/src/motion_smoothness/` and `scripts/counterfactual/`.
Every number in §4 was recomputed on `h100-server` from the frozen trees with
`scripts/counterfactual/verify_motion_review.py` (CPU only; no GPU, no re-scoring).

**Revision banner — read before quoting any number.** Commit `4d53fa2` changed the
repair's defaults (unaligned pixel-wise direction, top-k aggregation `k=3`,
`0.5/0.5` magnitude/direction weights). The frozen counterfactual repair scores and
the natural-set measurement were produced at **`feeb770`/`ccbd89b`**, i.e. **pre-
`4d53fa2`**; every table below labels each row `pre` or `post(replay)`. The `post`
column is a **replay from a cached real RAFT pass**, not an end-to-end score, and no
natural-set number exists for the shipped defaults. §4.5 lists the minimal re-run.

---

## 1. Origin — the complete reduction, video → scalar

**Input.** One video path, decoded whole with OpenCV, BGR→RGB, every decoded frame
kept, no timestamps and no frame-rate read (`:33-45`). Frames are consumed in file
order.

**Decimation.** A fixed stride of 2, not a rate-dependent factor (`:63-67`, called
at `:120`):

```text
E_j = x_{2j},   j = 0 … K-1,   K = ceil(N/2)                  :63-67, :120
```

**Interpolation.** AMT-S (`AMT-S.yaml`, `amt-s.pth`, `:183-184`) predicts the
midpoint of every consecutive decimated pair; `iters = 1` (`:75`) and the temporal
embedding is fixed at `t = 1/2` (`:107`), so the output sequence is
`[E_0, M_0, E_1, M_1, …]` of length `2K-1` (`:137-146`):

```text
M_j = I_theta(E_j, E_{j+1}; t = 1/2),   j = 0 … K-2            :107, :137-146
```

**Comparison.** `vfi_score` re-extracts **odd indices of the original full frame
list** and **odd indices of the interpolated sequence** (which are exactly the
`M_j`), then averages `get_diff` over the shorter of the two (`:156-162`):

```text
ori[i]         = x_{2i+1}                                       :157
interpolate[i] = M_i                                            :158
vfi = mean_i  get_diff(ori[i], interpolate[i])                  :160-162
get_diff(a,b)  = np.mean( cv2.absdiff(a, b) )                   :165-167
```

`tensor2img` returns `uint8` (`third_party/amt/utils/utils.py:86-89`), so the
comparison is a mean absolute pixel difference in `[0, 255]` over three channels.

**Normalisation and aggregation** (`:152`, `:171-179`, `:182-192`):

```text
s(video) = (255 - vfi) / 255
S        = mean_videos s(video)
```

Closed form for one video of `N` frames (`H×W`, three channels, `K = ceil(N/2)`):

```text
s = 1 - (1/255) * 1/(K-1) * sum_{j=0}^{K-2} || x_{2j+1} - I_theta(x_{2j}, x_{2j+2}) ||_1 / (3HW)
```

**Code facts.** Only odd-indexed frames are evaluated; the interpolated interval is
always two source-frame periods, i.e. `2/fps` seconds; the functional is a
non-negative per-pixel error with no velocity, sign or direction term anywhere in the
file; the dataset score is an unweighted mean over videos. (Our own
`IMPLEMENTATION_REPORT.md` previously described the decimation as
`max(1, round(fps/8))`; the locked code has no fps dependence at all, `:65`.)

## 2. Origin — what breaks, and why these are the nuisance factors

### P1 (code fact, `:65`, `:120`) — the reconstructed interval is `2/fps` seconds

The stride is the constant 2 and `fps` is never read. For a fixed physical
trajectory, the displacement an odd frame must be predicted across is the motion
over `2/fps` seconds, so the error — and therefore the score — is a function of the
sampling rate. A clip at 8 fps and the same trajectory at 2 fps are scored on
4×-different displacements. This is the **temporal sampling rate** nuisance.

### P2 (code fact, `:156-167`) — the reduction is direction-blind

`cv2.absdiff` + `np.mean` is symmetric and sign-free; nothing in the pipeline reads
the sign of the motion or the temporal order of the two conditioning frames. The
score is a reconstruction *difficulty*, not a continuity.

### P3 (inference from P2) — a pure time reversal is nearly invisible

If `N` is even, reversing the whole clip maps the decimated pair `(E_j, E_{j+1})` to
the same pair in reverse order and keeps the target `x_{2j+1}` the same true middle
frame:

```text
x'_{i} = x_{N-1-i}  ⇒  (x'_{2j}, x'_{2j+2}) = (x_{N-3-2j}, x_{N-1-2j}),  x'_{2j+1} = x_{N-2-2j}
```

Under the (inferred, testable) order symmetry `I_theta(a,b;1/2) ≈ I_theta(b,a;1/2)`
of a midpoint interpolator, `vfi` is unchanged and `s(reverse(x)) ≈ s(x)`. A **local**
reversal is sensed only at its two boundaries, where the conditioning pair's
displacement changes; the reversed interior is as easy to interpolate as the
original. So the metric has no explicit representation of "a direction flip is a
different kind of unsmoothness than a stall"; it can only see the incidental
displacement change.

### P4 (code fact, `:157`) — the score depends on decimation phase

Only odd-indexed frames are ever compared. Dropping a single leading frame shifts
every parity, changing which frames are reconstructed and which are targets, with no
change in the underlying trajectory.

### Formalisation

```text
s(x) = 1 - c * sum_{j}  || x_{2j+1} - I(x_{2j}, x_{2j+2}) ||_1 ,   c = 1/(255*3HW*(K-1))
```

- **FPS control (plan 13.4.1):** for the same trajectory resampled to 8/6/4/2 fps
  with duration held fixed, P1 predicts `s` **falls monotonically** as fps falls.
- **Full-reversal control:** P3 predicts `|s(reverse(x)) - s(x)|` is small relative
  to the repair's change on the same pair, while our direction term should drop
  sharply. This is the minimal counterexample: a clip that is entirely played
  backwards is maximally "unsmooth" by the contract's logic, yet the official
  functional barely moves.
- **Phase control:** P4 predicts `s(x_0..x_{N-2}) ≠ s(x_1..x_{N-1})` beyond noise.
- **Observed consequence inside our own family:** the official score *does*
  separate level 2 from level 3 (19/20 bases, §4), but only through boundary
  displacement; on the two mildest adjacent contrasts it is near chance
  (`L0>L1` 8/20, `L1>L2` 11/20). That is the signature of a reconstruction-error
  functional, not of an ordering functional.

### Minimal counterexample (concrete, already measured)

Level 2 of our `temporal_jerk` family is a **hold-and-jump** and level 3 a **local
reversal**; the contract declares `L2 > L3`. On canonical constant-velocity
trajectories the `feeb770` estimator tied or inverted them (review §2) — the defect
the direction term was added to fix, and the mirror image of the origin's direction
blindness.

## 3. Ours — contract, repair operator, what `4d53fa2` changed, what stays broken

### 3.1 Contract

`temporal_jerk` (`scripts/counterfactual/transforms.py:472-556`) keeps frame count,
rate and duration fixed and injects one of five ordered edits: original (rank 4),
one duplicated frame (3), a duplicated-and-skipped window (2), a reversed segment
(1), three reversed segments (0); ranks at `:489`, `:501`, `:516`, `:534`, `:551`.
The contract is the ordered ladder `L0 > L1 > L2 > L3 > L4`, scored as all
`C(5,2) = 10` ordered pairs per base (`scripts/counterfactual/cpa.py:58-72`).

### 3.2 The operator swap

Ours decodes timed frames, estimates a RAFT flow for every consecutive pair with its
real timestamp delta (`backends/audit.py:20-25`), forms `v_t = flow_t / dt_t`
(`metric.py:92`), and scores the **temporal derivative of the velocity field**:

```text
magnitude_t = mean |v_t(aligned) - v_{t-1}| / (|v_t| + |v_{t-1}| + eps)     metric.py:341-345
direction_t = mean [1 - cos(v_{t-1}, v_t)]  above min_motion_magnitude      metric.py:361-384
D_t         = magnitude_weight * magnitude_t + direction_weight * direction_t   metric.py:395-397
D_video     = mean of the top_k largest D_t                                 metric.py:263-284
score       = exp(-D_video)                                                 metric.py:286-294
```

The point of the swap is not a better estimator of reconstruction difficulty — it is
that the contract is about *temporal order and continuity*, which is a property of
the velocity-field sequence and is expressible only with a term that reads
direction.

### 3.3 `4d53fa2`: three changes and their motivations

1. **Direction measured on the raw (unaligned) pixel-wise flow** (default
   `direction_alignment = False`, `schemas.py:31`; the aligned series survives as
   `aligned_direction_t`/`--direction-alignment`). Alignment warps the current field
   along the previous flow to follow a moving subject, but at a frame reversal it
   samples the reversed field at the pre-reversal location and can cancel the flip.
2. **Top-k temporal aggregation, `k = 3`** (`schemas.py:32-33`). The injected edit
   touches a bounded number of transitions (`transforms.py:495-545`); the global mean
   over the 15-32 transitions of a clip dilutes it.
3. **`0.5/0.5` magnitude/direction weights** (`schemas.py:29-30`, was `0.7/0.3`) so
   the direction term has equal footing with the acceleration term.

### 3.4 What is not solved

- **The ratio inverts at low motion.** `relative = |Δv| / (|v_t| + |v_{t-1}| + eps)`
  promotes scale-freeness, but near zero it divides flow noise by flow noise and
  returns values near 1; measured on the 125 derived clips,
  `Spearman(speed_scale, D_video) = -0.3318` and `Spearman(speed_scale, score) =
  +0.3318` — slower clips systematically score *less* smooth.
- **No between-clip calibration.** The score is a within-clip continuity, not a
  scale comparable across videos; §4.3 shows this is what destroys the natural-set
  result.
- **`L2 > L3` is still not universal** (17/20 on the `4d53fa2` replay) and the
  strict-order rate **falls** from 10/20 to 8/20 while CPA rises (one statistic
  improves, the other regresses; §4.1).
- **The `L2`/`L3` declaration is unvalidated.** `transforms.py:507-510` rewrites
  `[f7…f11]` to `[f7,f7,f7,f11,f11]` — three frames excised plus two holds, not the
  "skip two frames" the plan text says — and no base carries a
  `manual_validity_status`; whether a stall is milder than a reversal is an
  assumption, not ground truth.

## 4. Experiment

All counterfactual numbers: frozen tree, 25 bases / 125 clips, test split, 20 bases /
200 pairs. `pre` = frozen `feeb770`; `post(replay)` = `4d53fa2` replayed from the
cached per-transition components (the replay reproduces all 125 frozen scores to
`1.7e-16`, which validates both the cache and the provenance).

### 4.1 Counterfactual ordering (`scripts/counterfactual/verify_motion_review.py`)

| statistic (test) | Official `pre` | Repair `pre` | Repair `post(replay)` |
|---|---:|---:|---:|
| CPA | 0.8300 | 0.8800 | **0.9150** |
| paired Δ vs Official | — | +0.0500 [−0.015, +0.110] | **+0.0850 [+0.015, +0.155]** |
| mean Spearman (declared rank vs score) | 0.7650 | 0.8150 | **0.8800** |
| strict order `L0>…>L4` | 7/20 | 10/20 | **8/20** |
| adjacent `L0>L1` / `L1>L2` / `L2>L3` / `L3>L4` | 8/11/19/20 | 18/15/15/18 | 18/15/**17**/18 |
| level means (L0…L4) | 0.9539 / 0.9533 / 0.9519 / 0.9452 / 0.9341 | 0.5432 / 0.5032 / 0.4867 / 0.4805 / 0.4303 | 0.4346 / 0.3732 / 0.3497 / 0.3276 / 0.2750 |

Reading: the **paired interval is the decision** — `pre` is parity (crosses zero),
`post(replay)` excludes zero but only just. The two order statistics disagree under
`4d53fa2`: `L2>L3` improves 15→17 while strict order falls 10→8, because top-k
sharpens the average pair decision and lets one noisy transition break the chain.
Official's profile is complementary: near chance on the mild contrasts and almost
perfect on the severe ones (19/20, 20/20) — it measures error magnitude, not order.

### 4.2 Natural preference set (`P1_NATURAL_AND_CONTROL_RUNS.md` §P1.2, `pre` only)

Frozen E0 split, 1 440 videos, 2 160 pairs, prompt-disjoint dev/test, coverage 1.0,
`scripts/evaluate_pairwise_statistics.py` + `scripts/evaluate_paired_backend_delta.py`:

| backend `pre` | dev margin | test acc0 | tie-aware | paired Δ (95 % CI) |
|---|---:|---:|---:|---|
| Official | 0.0215 | 0.5682 | **0.6364** [0.6101, 0.6636] | — |
| Repair `feeb770` | 0.1459 | 0.3388 | **0.3248** [0.2992, 0.3512] | **−0.3116** [−0.3473, −0.2760] |

Kendall τ-b 0.4137 (Official) vs 0.0686 (Repair); per-video
Spearman(Official, Repair) = **−0.1032** (Pearson +0.0198); Official scores span
0.7513–0.9959 (median 0.9650), Repair 0.3396–0.9552 (median 0.4944), both fully
distinct and neither saturating. Harness caveat: both intervals resample **pairs**,
not `prompt_id` clusters as plan 5.4 requires (43 test prompts × 30 pairs); prompt
clustering widens them 35–46 % (repair [0.2907, 0.3628], paired [−0.3605, −0.2643]) —
conclusions survive, published widths do not.

### 4.3 The contradiction, confronted

**Why can one estimator be direction-correct on an injected ladder and
anti-correlated with humans on natural video?** Because the two tasks are different
statistics. CPA scores only pairs **within one base**: the two clips share source,
content, length and format, so every clip-level nuisance cancels. A variance
decomposition of the counterfactual test scores makes this explicit — between-base
variance share 0.9224 (Official) and 0.5776 (Repair), and *both* backends order the
five levels perfectly on average within a base (Spearman +1.000 of declared rank vs
within-base level mean). The natural set compares **different** clips, where the
clip-level nuisance is the whole signal.

That nuisance is measurable. A trivial CPU baseline — mean absolute grayscale frame
difference (MAD) over ≤12 frames, "lower is smoother", dev-calibrated margin — is
almost as accurate as Official and far above the repair:

| backend `pre` | tie-aware (pair CI) | prompt-clustered CI | non-tie direction accuracy |
|---|---:|---|---:|
| Official | 0.6364 [0.6101, 0.6636] | [0.5961, 0.6744] | 0.8075 |
| lower-MAD CPU baseline | **0.6000** [0.5736, 0.6264] | [0.5674, 0.6295] | 0.7935 |
| Repair `feeb770` | 0.3248 [0.2992, 0.3512] | [0.2907, 0.3628] | 0.4857 |

Official is a near-monotone function of that cue (`Spearman(Official, MAD) =
−0.9451`), while the repair runs the *other* way (`Spearman(Repair, MAD) = +0.2479`;
+0.1914 within `.mp4`, +0.1533 within `.gif`). The mechanism is the ratio
normalisation of §3.4: a slower clip has a smaller denominator, so its normalised
acceleration reads higher, `D` rises and the score falls. Decile means make the
inversion monotone in the wrong direction: repair score rises 0.4838 → 0.5947 from
the lowest to the eighth MAD decile, while Official falls 0.9866 → 0.8742.

**The decisive test of the mechanism, already run on the frozen scores.** If the
below-chance result is the motion-scale nuisance, it should shrink when the two
clips in a pair have the same motion scale. Conditioning the 1 501 non-tie pairs on
`|log(MAD_a/MAD_b)|`:

| motion-scale band | pairs | Repair direction acc. | Official direction acc. |
|---|---:|---:|---:|
| same (<10 %) | 91 | **0.6044** | 0.5824 |
| close (10–35 %) | 237 | 0.5570 | 0.6456 |
| far (>35 %) | 1 173 | **0.4621** | 0.8576 |

Matched-scale pairs are the only band where the repair is above chance — and it is
then *ahead* of Official. The entire deficit lives in the far band, exactly where the
scale-free ratio's between-clip variation dominates. So both headline results are
true, and they are not evidence about "smoothness" in the same sense: the
counterfactual ladder measures a within-clip ordering that the estimator does have;
the natural set measures a between-clip scale that it does not.

### 4.4 Testable predictions

1. **Re-run the natural set with the shipped `4d53fa2` defaults.** If §4.3 is right
   the below-chance gap persists; if it disappears, the direction term was the
   dominant nuisance and the mechanism is incomplete.
2. **Gate the denominator**, replacing `|v_t| + |v_{t-1}| + eps` with a clip-level
   floor `max(eps, rho * median speed)`: this should reduce `|Spearman(score, MAD)|`
   while leaving the within-base `L2>L3` ordering intact (`rho` tunable on dev).
3. **Full-reversal control** (P3): reversing a whole clip should leave the Official
   score nearly unchanged and drop the repair score sharply.
4. **FPS control** (P1, plan 13.4.1): at 8/6/4/2 fps the Official score should fall
   with fps while the repair's `dt`-normalised velocity stays approximately flat.

### 4.5 Minimal re-run list — to make the two published rows describe shipped code

Scores to redo (both are repair-only; Official is unchanged):

| # | scores | command | measured |
|---|---|---|---|
| 1 | counterfactual repair, 125 clips | archive `scores/motion_smoothness__repair*.jsonl` → `run_dim.sh motion_smoothness 1,2,3,4,5` | 73 s |
| 2 | natural repair, 1 440 videos | archive `natural-preference-runs/motion_smoothness/` → `run_natural.sh motion_smoothness repair 1,2,3,4,5` | 11 min 5 s |

Reusable: Official counterfactual scores and CPA, Official natural scores and
statistics, human labels/split/pair census. To recompute: repair CPA, dev margin,
rank-gap and adjacent-pair counts, strict order, Spearman, `D_mean`/`D_tail`, paired
`base_id`-clustered CI; natural dev margin, tie-aware accuracy, τ-b, model-level
Pearson, and the marginal/paired intervals **with `prompt_id` clustering**. Then
regenerate `motion_smoothness.md`, the `CONSOLIDATED.md` Raw result row,
`table2.csv`/`table2.json`/`SUMMARY.md`, `README.md` and the P1.2 tables. Also fix
the provenance defect: `run_dimension.py:1287` records the *render* revision as the
report's `code SHA` (the published report names `5c4a130`, a report-generator commit,
for `feeb770` scores) and the score rows carry no revision at all.

## 5. Consistency with this dimension's review

`docs/counterfactual-reports/motion_smoothness.review.md` (commit `b876571`) reaches
the same conclusions from the same evidence; this document is its paper-facing
condensation, not a second opinion. Both mark the `CONSOLIDATED.md` motion row and
P1.2 as **pre-`4d53fa2`** and give the same minimal re-run list (§4.5 here); both
treat the **paired** interval as the decision and report the replay as
`+0.0850 [+0.015, +0.155]`, explicitly not an end-to-end score; both flag the two
harness defects (the report's `code SHA` is the render revision; the natural
intervals resample pairs, not `prompt_id` clusters). The review's §2 mechanism is
§4.3 here — the motion-scale band conditioning is new in this round and is the
strongest single piece of evidence for it. Neither document claims a replacement:
the honest row is *audit-only* — the injected within-clip ordering is satisfiable
(and improved by `4d53fa2` on CPA), the human ordering is not, and the natural metric
is reproducible by a baseline that uses no model.
