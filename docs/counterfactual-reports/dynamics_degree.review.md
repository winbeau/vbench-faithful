# Review: `dynamics_degree` counterfactual report

Scope: the archived `docs/counterfactual-reports/dynamics_degree.md` (code SHA
`66c4a99dba05aceaebe80276ffbffc607c3d2e40`), the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `run_dimension.py`), and the Audit time
normalisation that produced the `repair` row
(`metrics/dynamic-degree/src/dynamic_degree/backends/audit.py`). The review was
subsequently carried through to a fix and a re-measurement on the scoring host;
sections 3.1 and 6 carry the corrected numbers and the current state.

Verdict: the headline row `dynamics_degree | fps_resampling | 0.8333 | 0.8444 |
+0.0111` in `docs/counterfactual-reports/README.md` is **not** a measurement of
the contract and the `+0.0111` must not be quoted as a Repair win. Unlike
`subject_consistency`, the report's own `Score sensitivity` table already
contains the proof: the Official metric's dependence on the sampling interval is
real and measured, and the archived repair did **not** remove it — it mirrored it
(`p = -0.481` against `+0.491`). The composite CPA could not have reported this
even in principle. The fix and its verification are in section 6: with a fixed
exponent of 0.5 the test-split level profile goes from `+0.491 / -0.481` to
`-0.011`, i.e. the reported aggregate becomes frame-rate invariant, while
individual clips remain dispersed.

## 1. This family is a pure invariance family

`fps_resampling` (`transforms.py:34-58`) emits one variant per ladder rung, all
with `expected_rank = 1`:

| level | `expected_rank` | contract |
|---|---:|---|
| `fps8` | 1 | every rung scores the same |
| `fps6` | 1 | ” |
| `fps4` | 1 | ” |
| `fps2` | 1 | ” |

`family_pairs` (`cpa.py:58-72`) therefore expands **every** pair with expected
relation `0`: 4 levels → `C(4,2) = 6` pairs per base, 10 dev bases → 60 dev
pairs, 30 test bases → 180 test pairs. There is no sensitivity half to carry the
score, so every admissible statistic must be sensitive to the sign *and*
magnitude of the level effect. Both published CPA columns fail that requirement,
for three independent reasons.

## 2. The CPA cannot express invariance for a same-rank family

### 2.1 `zero-margin` requires exact floating-point equality

`prediction(delta, 0.0)` returns `0` only when `|delta| <= 0`. Four independently
encoded H.264 clips will never produce bit-identical continuous scores, so both
backends report `0.0000`, with a degenerate bootstrap CI `[0.0000, 0.0000]`. A
*perfectly* invariant metric also scores `0.0000` here (row `p = 0.00` in
section 3). The column contains no information and must be dropped from the
table rather than read as "the Official metric fails and the Repair passes".

### 2.2 The tie margin saturates at 0.90 by construction

`calibrate_margin` (`cpa.py:84-95`) searches `{0} ∪ quantile(|delta|_dev, 0.1..0.9)`
and keeps the margin that maximises **dev** CPA. For a same-rank family every dev
pair expects a tie, so

```text
dev_cpa(margin) = fraction of dev pairs with |delta| <= margin
```

which is monotone non-decreasing in `margin`. The search therefore always selects
the largest candidate, the 0.9 quantile, so the dev tie-aware CPA is ≈`0.90`
regardless of the score: it is fixed by the search, not measured. The published
test number is then only

```text
P_test(|delta| <= q90_dev(|delta|))
```

a near-scale-free statistic. It is invariant to any per-level linear rescaling of
the score — which is precisely the class of transformation that separates the two
backends here (section 4).

### 2.3 With every pair expecting a tie, the statistic is sign-blind — and inverts

Since `expected = 0` for all 180 pairs, `|delta| <= margin` cannot distinguish a
metric whose score *rises* 2x across the ladder from one whose score *falls* 2x.
`output/counterfactual/_scratch/dd_instrument_comparison.py` runs the
repository's own `cpa.py` on synthetic ladders whose only difference is the
exponent, with per-clip noise calibrated so the `p = +0.49` row reproduces the
observed within-base CV:

| synthetic lag law `p` | within-base CV | relative range | signed exponent | zero-margin CPA | tie-aware CPA |
|---|---:|---:|---:|---:|---:|
| +0.49 (Official raw flow) | 0.2841 | 0.7244 | +0.487 | 0.0000 | 0.7667 |
| -0.48 (Official / dt, the shipped repair) | 0.2566 | 0.6579 | -0.483 | 0.0000 | **0.7500** |
| 0.00 (perfectly lag-invariant) | 0.1226 | 0.3207 | -0.003 | 0.0000 | **0.6111** |
| +1.00 (pure ballistic) | 0.5605 | 1.4370 | +0.997 | 0.0000 | 0.7778 |

The composite tie-aware CPA **ranks the perfectly invariant metric below both
broken ones**. It is anti-correlated with the contract for this family.

Additional degeneracies visible in the published table:

- the Official margin is `20.14`, larger than the entire level effect
  (`16.51 → 32.87`); the tie bucket swallows the effect it is meant to test;
- the margin is calibrated on 10 dev bases / 60 pairs, so it is a noisy q90. Over
  30 resamples of the same 10/30 base budget (same script family,
  `blindness_seeds.py`) the headline CPA has sd ≈ 0.05-0.06 for both broken and
  ideal scores, so `+0.0111` is ~1/5 of one standard deviation and is expected to
  be ≈ 0 under the null.

### 2.4 The report's new `Invariance statistics` does not rescue the comparison

CV and relative range are *unsigned* dispersion measures; they cannot tell a +2x
drift from a -2x drift either. In the synthetic table the two mirrored laws give
CV `0.2841` vs `0.2566` and relative range `0.7244` vs `0.6579`, i.e. a ~10%
difference, against a sign flip in the exponent. The published values
(`0.3177 → 0.2360` CV, `0.8196 → 0.6107` relative range) are of exactly that
order. **The observed dispersion "improvement" is consistent with a pure sign
flip of the level effect, so it is not evidence that the repair restored
invariance.** The dispersion block is a genuine improvement over the CPA block,
but it still cannot adjudicate this family.

The statistic that does is the *signed* log-log slope of the score against the
sampling interval, per backend and per clip. Section 3 applies it to the numbers
the report already published.

## 3. The report's own level table proves the repair did not fix anything

`Score sensitivity` (test and dev pooled, 40 clips per level):

| backend | 8 fps | 6 fps | 4 fps | 2 fps | ratio to 8 fps | fitted `s ~ dt**p` | per-step `p` |
|---|---:|---:|---:|---:|---|---:|---|
| official (raw top-5% flow, px) | 16.5108 | 19.7201 | 24.2540 | 32.8744 | 1 / 1.19 / 1.47 / 1.99 | **+0.491** | +0.62, +0.51, +0.44 |
| repair (`d/dt`, diag/s) | 0.2484 | 0.2230 | 0.1864 | 0.1282 | 1 / 0.90 / 0.75 / 0.52 | **-0.481** | -0.38, -0.44, -0.54 |

The two rows are mirror images, to within 0.01 in the exponent, and the identity
is exact rather than approximate. Within one clip `dt` is constant and the
duration-weighted mean of `d/dt` equals the mean of `d` divided by `dt`, so

```text
repair  ==  official / dt  *  const
```

and indeed `official / (dt_ratio * repair)` is constant across the ladder:

| rung | `official / (dt_level/dt_8)` | repair | repair / that |
|---|---:|---:|---:|
| 8 fps | 16.5108 | 0.2484 | 0.015 |
| 6 fps | 14.7901 | 0.2230 | 0.015 |
| 4 fps | 12.1270 | 0.1864 | 0.015 |
| 2 fps | 8.2186 | 0.1282 | 0.016 |

The `repair` row is not an independent measurement of the dimension. It is the
Official metric's own flow magnitude divided by the sampling interval, times a
fixed constant (the image diagonal plus the residual-vs-apparent routing, which
changes little on these clips). Consequently:

- The criticism of the Official metric **stands and is quantified**: its raw
  top-5% flow grows sublinearly in `dt`, so the same trajectory scores ~2x higher
  at 2 fps than at 8 fps, and because `check_move` compares against a threshold
  fixed in pixels (`6*min(H,W)/256`), downsampling also pushes clips across the
  static/moving boundary. The metric is not comparable across frame rates.
- The claim that the archived repair fixes it is **false**. `p = -0.481` is the
  same violation with the opposite sign, and it is not an independent
  measurement that could have come out any other way.
- Target `p = 0`. Section 6 reports the fixed and re-measured value: `-0.011` on
  the test split.

### 3.1 `+0.491` is a heavy-tailed aggregate, not the per-clip law

The figure above is the slope of the *mean of 40 clips* at each rung, and this
benchmark's per-clip scores span three orders of magnitude (`0.30` to `153` px at
8 fps), so the mean at the coarse rungs is dominated by a few high-motion clips.
Measured directly on the 10 dev bases instead:

- the per-clip exponent fitted across that clip's own four rung scores averages
  **+0.706** (not +0.491), and it correlates **+0.925** with the within-clip RAFT
  exponent measured from the clip's own lag-1/2/4 displacements, and **+0.935**
  with `log(s_fps2 / s_fps8)`;
- within a single 8 fps clip, the top-5% displacement at lags 1/2/4 sampled frames
  (0.125/0.25/0.5 s) grows `1 : 1.73 : 2.86`, i.e. `p = +0.758` on the means and
  `+0.819` on the mean of per-clip fits.

So the level-mean slope understates the typical clip, and the exponent is
measurable from a clip's own frames. Both facts matter: the first means the
headline number must not be presented as "the" exponent, the second is the
premise any lag-calibrated normalisation relies on (section 6). The mirror
identity `repair == official / dt` is unaffected, because it is exact arithmetic
that holds clip by clip.

## 4. Why the repair code does that

`normalized_speed` divides the displacement by `dt` exactly once
(`backends/audit.py`, `normalized_speed`), which is the *ballistic* normalisation:
it is exact only when displacement grows linearly with the lag
(constant-velocity motion). A power law `d ~ dt**p` becomes `d/dt ~ dt**(p-1)`, so
the 1.0 exponent is off by exactly `p`. Measured `p = +0.49` therefore yields
`-0.51`, and the observed `-0.48`. The choice of exponent, not the estimator or
the pipeline, is the defect.

## 5. The clips are not the cause

Everything below was checked on this machine; the derived clips themselves live
on the scoring host (`/root/wenbiao_zhao/datasets/counterfactual-vbench`) and the
sources under `/home/msy625/projects/datasets/VBench/e0_public/`, neither of which
is reachable from this workspace, and this workspace has no CUDA device or RAFT
weight. Construction was therefore audited by exact arithmetic against
`resample_indices`, plus an end-to-end replay through the real
`transforms.fps_resample` → `common.encode_video` → `common.decode_video` path on
clips whose motion law is known
(`output/counterfactual/_scratch/data_ladder_check.py`).

**Verified sound.** Container fps equals the rung; frame counts equal
`resample_indices` exactly; the retained frames are a strictly increasing subset
of the source (no duplication, no drop, `-vsync 0` respected); the nominal
inter-frame times of the retained frames match their source times; and for the
8 fps sources the container duration is exactly 2.000 s at every rung.

**Two real defects, both smaller than the effect and neither able to cause it:**

| source geometry | 8 fps | 6 fps | 4 fps | 2 fps |
|---|---:|---:|---:|---:|
| mp4 16 frames / 8 fps: covered source span | 1.875 s (100%) | 1.875 s (100%) | 1.750 s (93.3%) | **1.500 s (80.0%)** |
| gif 33 frames / 10 fps: covered source span | 3.200 s (100%) | 3.200 s (100%) | 3.200 s (100%) | 3.000 s (93.8%) |
| gif 33 / 10: container duration | 3.375 s | 3.333 s | 3.500 s | 3.500 s |

1. **The coarse rungs sample a shorter trajectory.** At 2 fps the clip covers 80%
   of a 16-frame source and its mean rests on 3 transitions instead of 15, so the
   rungs do not average the same content. This biases the *coverage* of the mean,
   not the displacement-vs-lag law, and it works against the observed direction.
2. **The 10 fps GIF rungs do not hold the duration fixed** (up to +6.1% vs the
   3.300 s source), because `n_kept / target_fps` cannot equal the source
   duration when the source rate is not a multiple of the ladder rate. Inter-frame
   timing stays correct (misalignment <= 0.05 s); the drift is a stale final frame.

Neither explains the sublinear law, because the same pipeline reproduces the true
law for synthetic content:

| synthetic motion law | profile 8/6/4/2 fps | fitted `p` |
|---|---|---:|
| constant velocity (2 px/frame) | 1 / 1.365 / 2.005 / 4.013 | **+0.996** |
| independent per-frame jitter | 1 / 1.347 / 1.169 / 1.788 | +0.357 |
| static | 1 / 1 / 1 / 0.999 | 0.000 |

The ladder faithfully transmits a ballistic law at `p = +1.0`; it cannot
manufacture sublinearity. Three further checks on the real clips, run on the
scoring host, close the remaining alternative explanations:

1. **Per-rung re-encoding is not the cause.** The derived clips are re-encoded at
   each rung, so the level effect could have been a codec effect. On the identical
   0.5 s trajectory span the 2 fps rung's lag-1 displacement and the 8 fps rung's
   lag-4 displacement agree to within 3.5% (ratios 0.965-1.004 over 6 bases), and
   the stored source agrees with the 8 fps rung as well.
2. **It is not RAFT-specific.** Two independent estimators on the same frame pairs
   are both sublinear: RAFT `+0.819` and Farneback `+0.608` (mean of per-clip fits
   on the 10 dev bases). An estimator that saturated would be *more* linear than
   its cross-check, not less.
3. **The motion genuinely turns at these lags.** `straightness` (direct
   displacement at a lag divided by the sum of the lag-1 displacements over the
   same span) has median `0.793` at lag 2 and `0.574` at lag 4 across all 160
   clips, against `1.0` for constant-velocity motion and `k**-0.5 = 0.707 / 0.500`
   for a diffusive law. The chord is systematically much shorter than the path, so
   displacement grows sublinearly in the lag.

The two construction defects are still worth fixing before the family is used
again, because they make the rungs non-comparable; they are reported here rather
than patched because regenerating the dataset would invalidate the existing 160
scores.

## 6. The fix, and what it does and does not deliver

The normalisation is now `d / dt**alpha` with a fixed, pre-registered
`alpha = 0.5` (`AuditConfig.default_lag_exponent`, mode `LagExponentMode.FIXED`),
and the same exponent is used for the static/moving threshold and coverage so the
decision does not mix a pixel threshold with a per-frame displacement. What
remains configurable:

- `LagExponentMode.MEASURED` applies each clip's own fitted exponent. It is the
  diagnostic path, not the default: the fitted value is itself lag-window
  dependent (mean `0.618` at 8 fps falling to `0.480` at 2 fps), which is exactly
  why per-clip calibration under-corrects (`-0.138` against `-0.011`).
- `LagExponentMode.BALLISTIC` and `AuditAblation.WITHOUT_LAG_SCALING` restore the
  archived `d/dt` behaviour; `AuditConfig.lag_exponent` + `lag_exponent_source`
  apply any other fixed exponent with declared provenance.
- Every result carries `LagScalingEvidence`: the fitted exponent, the per-lag
  chord and path displacements, `straightness = chord/path`, pair counts and the
  fit residual.

Re-measured end to end on all 160 real clips (5 GPUs, `metrics/dynamic-degree` on
commit `a044ac9` + this patch), level profile and slope:

| backend | dev profile | dev `p` | test profile | test `p` |
|---|---|---:|---|---:|
| `official` (`alpha=0`) | 1 / 1.184 / 1.635 / 2.404 | +0.641 | 1 / 1.196 / 1.437 / 1.911 | +0.458 |
| `repair_v1` (`alpha=1`, archived) | 1 / 0.881 / 0.808 / 0.586 | -0.376 | 1 / 0.902 / 0.736 / 0.498 | -0.511 |
| `repair_v2` (`alpha=0.5`, shipped) | 1 / 1.017 / 1.143 / 1.172 | +0.124 | 1 / 1.042 / 1.040 / 0.996 | **-0.011** |

- The exponent that flattens the ladder is `+0.518` with a base-cluster bootstrap
  95% CI `[+0.417, +0.651]`; both `0` and `1` fall outside it. Slope magnitude on
  the test split drops from `0.511` to `0.011` (**~46x**).
- The recorded scores reproduce `mean_displacement / dt**0.5` to `1.7e-16`
  (160/160 clips), so the end-to-end path and the offline calibration agree.
- **Per-clip invariance is not achieved.** The per-base `fps2/fps8` ratio still
  has median `1.284`, IQR `[0.807, 1.537]`, and only 20% of bases are within
  ±20% of 1. The fix makes the *reported aggregate* frame-rate invariant, which is
  the quantity VBench's dynamic degree is defined as (a dataset-level mean); it
  does not make individual clips comparable across frame rates.
- **Calibrating the exponent on dev alone fails.** The dev split yields `+0.624`,
  which leaves a residual test slope of `-0.135`. The exponent must therefore be
  defended as an a-priori diffusive constant (`displacement ~ sqrt(dt)`) that the
  40-base CI supports, not as a tuned parameter. The dev split is too small for
  this constant.
- Coverage moves the same way: on the test split the static/moving fraction goes
  `0.501 -> 0.333` under `alpha=1` but `0.501 -> 0.533` under `alpha=0.5`, and at
  the reference lag the two thresholds coincide by construction.

## 7. What the report should say instead

For a same-rank family the primary table is the level profile and its signed
exponent; CPA belongs in an appendix, labelled non-diagnostic. The regenerated
`docs/counterfactual-reports/dynamics_degree.md` now does this, dev and test
separately:

1. **Sampling-interval response (primary).** Per backend: the per-level mean, the
   ratio profile relative to the reference rung, and the fitted
   `p = d log(score) / d log(dt)`. Target `p = 0`. Archived: Official `+0.491`,
   `repair_v1` `-0.481`. Shipped: `repair_v2` `-0.011` (test).
2. **Calibrated exponent.** The exponent that flattens the ladder, `+0.518` with
   a base-cluster bootstrap 95% CI `[+0.417, +0.651]`, plus the explicit warning
   that a dev-only calibration (`+0.624`) does not transfer.
3. **Per-clip dispersion and exponent.** Per-base `fps2/fps8` median/IQR/fraction
   within ±20%, within-base CV and relative range (marked unsigned), and the
   distribution of the within-clip measured exponent per rung. This is what keeps
   the aggregate result from being read as per-clip invariance.
4. **Threshold and coverage domain.** The static/moving fraction per rung under
   each exponent, so the threshold component is auditable too.
5. **Composite CPA (appendix, non-diagnostic).** Kept with the dev margin, the
   note that the dev tie-aware CPA is 0.90 by construction (section 2.2), and the
   note that zero-margin CPA requires exact float equality.

The honest headline sentence is: *the Official dynamic-degree score is
sampling-interval dependent (measured exponents `+0.491` on the 40-clip level
means and `+0.706` per clip, a ~2x inflation from 8 to 2 fps, plus a fixed-pixel
threshold that is easier to cross at low frame rates); the archived repair
mirrored that dependence (`-0.481`) instead of removing it; and normalising by
`dt**0.5` removes it from the reported aggregate (`-0.011` on the test split)
while leaving individual clips dispersed.*

## 8. Status of the numbers in this review

- Sections 2-4 are derived from the archived report and the committed CPA code.
  Nothing there needs a model run; the synthetic demonstrations use the
  repository's own `cpa.py` and its `calibrate_margin` / `evaluate_method`.
- Section 5 is derived from `resample_indices` arithmetic, the real
  transform/encode/decode path on synthetic clips, and — for the re-encoding,
  two-estimator and straightness checks — from the 160 real clips on the scoring
  host (`/root/wenbiao_zhao/datasets/counterfactual-vbench`, RAFT
  `/root/wenbiao_zhao/models/raft/raft-things.pth`, upstream
  `/root/wenbiao_zhao/VBench`). The two construction defects were verified by
  arithmetic against `resample_indices`; confirming them against the 160 stored
  files needs only `ffprobe` plus the recorded `kept_frame_indices`, which has not
  been run.
- **Section 6 is measured, not projected.** All 160 clips were scored end to end
  with the patched backend on the scoring host (5 physical GPUs, one logical
  `cuda:0` each), 160/160 succeeded, and the recorded scores reproduce the
  offline `mean_displacement / dt**0.5` calibration to `1.7e-16`. The
  `alpha = 0.5` default, the `alpha*` bootstrap and the per-clip residual are
  from that run; the coverage column comes from the same run plus the
  `WITHOUT_LAG_SCALING` ablation pass.
- The report at `docs/counterfactual-reports/dynamics_degree.md` has been
  regenerated from those raw score files by
  `scripts/counterfactual/regen_dynamics_degree_report.py` (no model needed) and
  now leads with the level profile, the calibrated exponent, the per-clip
  dispersion, the measured-exponent distribution and the threshold/coverage
  domain, with the composite CPA moved to a section labelled non-diagnostic.
- Reproduction: `scripts/counterfactual/score_lagcal.py` (one shard per physical
  GPU, `--mode fixed|measured|ballistic|raw`) writes the score files that
  `scripts/counterfactual/regen_dynamics_degree_report.py` turns into the report.
  The counterfactual harness itself still does not persist the new evidence —
  `scripts/counterfactual/score.py:_first` returns only `score/status/error` — so
  the default path should be extended or replaced by that driver.
- As with the sibling review, the archived reports name a `code SHA` that
  predates the runner that produced them: `66c4a99` is an ancestor of
  `7ffa6fa`, the commit that first adds `scripts/counterfactual/cpa.py`. The
  reports are not reproducible from the revision they cite and the field should
  be refreshed when they are regenerated.
- Scratch scripts behind the numbers above live in
  `output/counterfactual/_scratch/` (gitignored): `dd_instrument_comparison.py`
  and `blindness_sweep.py` / `blindness_seeds.py` for sections 2.3-2.4,
  `data_ladder_check.py` for section 5, `level_exponent.py` for section 3.
