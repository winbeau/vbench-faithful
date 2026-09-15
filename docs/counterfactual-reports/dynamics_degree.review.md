# Review: `dynamics_degree` counterfactual report

Scope: `docs/counterfactual-reports/dynamics_degree.md` (code SHA
`66c4a99dba05aceaebe80276ffbffc607c3d2e40`), the CPA instrument that produced it
(`scripts/counterfactual/cpa.py`, `run_dimension.py`), and the Audit time
normalisation that produced the `repair` row
(`metrics/dynamic-degree/src/dynamic_degree/backends/audit.py:normalized_speed`).

Verdict: the headline row `dynamics_degree | fps_resampling | 0.8333 | 0.8444 |
+0.0111` in `docs/counterfactual-reports/README.md` is **not** a measurement of
the contract and the `+0.0111` must not be quoted as a Repair win. Unlike
`subject_consistency`, the report's own `Score sensitivity` table already
contains the proof: the Official metric's dependence on the sampling interval is
real and measured (`p = +0.49`), and the shipped repair does **not** remove it —
it mirrors it (`p = -0.48`). The composite CPA could not have reported this even
in principle.

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

- The criticism of the Official metric **stands and is now quantified**: its raw
  top-5% flow grows like `dt**0.49`, so the same trajectory scores ~2x higher at
  2 fps than at 8 fps, and because `check_move` compares against a threshold fixed
  in pixels (`6*min(H,W)/256`), downsampling also pushes clips across the
  static/moving boundary. The metric is not comparable across frame rates.
- The claim that the repair fixes it is **false**. `p = -0.48` is the same
  violation with the opposite sign. Against a target of `p = 0`, the shipped
  repair is as wrong as the metric it audits, and no number in this report shows
  otherwise.

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

Neither explains `p = +0.49`, because the same pipeline reproduces the true law
for synthetic content:

| synthetic motion law | profile 8/6/4/2 fps | fitted `p` |
|---|---|---:|
| constant velocity (2 px/frame) | 1 / 1.365 / 2.005 / 4.013 | **+0.996** |
| independent per-frame jitter | 1 / 1.347 / 1.169 / 1.788 | +0.357 |
| static | 1 / 1 / 1 / 0.999 | 0.000 |

The ladder faithfully transmits a ballistic law at `p = +1.0`; it cannot
manufacture sublinearity. The measured exponent is a property of the content or
of RAFT, not of the dataset construction. (Both defects are still worth fixing
before the family is used again, because they make the rungs non-comparable; they
are reported here rather than patched because regenerating the dataset would
invalidate the existing 160 scores.)

## 6. The repair as it now stands

The normalisation has been replaced with a measured-exponent one in
`metrics/dynamic-degree` (see the new section in
`metrics/dynamic-degree/IMPLEMENTATION_REPORT.md` for the full account):

- `AuditConfig.lag_scaling_enabled` (default on) fits the clip's own
  displacement-vs-lag exponent from within-clip multi-lag flows (default lags
  1/2/4 sampled frames) and applies `d / dt**p` instead of `d / dt`; `p = 1`
  reproduces the previous behaviour bit for bit, so ballistic content is
  unchanged.
- `AuditAblation.WITHOUT_LAG_SCALING` restores the ballistic normalisation as a
  named ablation, `AuditConfig.lag_exponent` + `lag_exponent_source` apply a
  fixed (e.g. dev-calibrated) exponent with declared provenance, and every result
  carries `LagScalingEvidence`: the exponent, the per-lag chord and path
  displacements, `straightness = chord/path`, pair counts and fit residual.
- `tests/test_lag_scaling.py` (23 tests) verifies the invariance directly. With
  the law measured from the real run (`p = 0.49`):

| trajectory | previous `d/dt`, 2 fps / 8 fps | `d/dt**p`, 2 fps / 8 fps |
|---|---:|---:|
| ballistic (`p = 1`) | 1.0000 | 1.0000 |
| measured real law (`p = 0.49`) | **0.4931** | **1.0000** |

This is the fix for the *normalisation* defect identified here. It is not yet
evidence that the real clips become invariant, for the reason in section 8.

## 7. What the report should say instead

For a same-rank family the primary table is the level profile and its signed
exponent; CPA belongs in an appendix, labelled non-diagnostic:

1. **Sampling-interval response (primary).** Per backend: the per-level mean and
   median, the ratio profile relative to the reference rung, and the fitted
   `p = d log(score) / d log(dt)` with a cluster bootstrap over `base_id` and its
   per-step values. Target `p = 0`. Current: Official `+0.49`, previous repair
   `-0.48`.
2. **Per-clip exponent (the mechanism).** Distribution of the measured within-clip
   exponent per backend and per rung, plus `straightness` at lags 2 and 4. This is
   what distinguishes genuine trajectory curvature from estimator saturation, and
   it needs no ladder.
3. **Dispersion (secondary).** The existing within-base CV and relative range,
   clearly marked as unsigned and therefore unable to detect a sign flip.
4. **Composite CPA (appendix, non-diagnostic).** If kept, report alongside it the
   dev margin, the dev tie-aware CPA (which is 0.90 by construction, section 2.2),
   and a note that zero-margin CPA requires exact float equality.

The honest headline sentence is: *the Official dynamic-degree score is
sampling-interval dependent (`p = +0.49`, ~2x inflation from 8 to 2 fps, plus a
fixed-pixel threshold that is easier to cross at low frame rates), and the
previous repair mirrored that dependence (`p = -0.48`) instead of removing it.*

## 8. Status of the numbers in this review

- Sections 2-4 are derived from the published report and the committed CPA code.
  Nothing there needs a model run; the synthetic demonstrations use the
  repository's own `cpa.py` and its `calibrate_margin` / `evaluate_method`.
- Section 5 is derived from `resample_indices` arithmetic and the real
  transform/encode/decode path on synthetic clips. The derived-clip defects were
  **not** re-verified against the 160 real files, because neither the clips nor
  the sources are reachable from this workspace and there is no CUDA device here.
  A scoring host can confirm them with `ffprobe` (per-clip fps, frame count,
  duration) plus the recorded `kept_frame_indices`.
- Section 6's fix is unit-verified only. Whether the measured exponent is stable
  across rungs on real clips — i.e. whether `d / dt**p` actually closes the gap
  rather than moving it — requires re-running the Audit backend on the 40
  `dynamics_degree` bases with the real weights. Every result now emits the
  evidence needed to check that (`lag_scaling.lags_frames`, `lag_seconds`,
  `chord_displacement_diagonals`, `straightness`, `fit_rmse_log_log`), and the
  same re-run settles the content-vs-saturation question.
- The counterfactual harness does not yet persist that evidence:
  `scripts/counterfactual/score.py:_first` returns only `score/status/error`, so
  the `lag_scaling` block is currently dropped from the `repair` rows.
- As with the sibling review, the archived reports name a `code SHA` that
  predates the runner that produced them: `66c4a99` is an ancestor of
  `7ffa6fa`, the commit that first adds `scripts/counterfactual/cpa.py`. The
  reports are not reproducible from the revision they cite and the field should
  be refreshed when they are regenerated.
- Scratch scripts behind the numbers above live in
  `output/counterfactual/_scratch/` (gitignored): `dd_instrument_comparison.py`
  and `blindness_sweep.py` / `blindness_seeds.py` for sections 2.3-2.4,
  `data_ladder_check.py` for section 5, `level_exponent.py` for section 3.
