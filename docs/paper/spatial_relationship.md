# spatial_relationship — origin vs ours (paper material)

**Thesis.** VBench 1.0's Spatial Relationship score is the frame-weighted mean of a
**sign-free axis-dominance indicator**: it asks whether the two named objects are
separated along the correct *axis*, never which of them is on which *side*. The
nuisance factor this document names is therefore **geometric direction (sign)**.
Mirroring the frame leaves the metric *exactly* unchanged at the level of a fixed
detection list, so `score(original) > score(flip)` is unsatisfiable for Official by
construction; the counterfactual edge measures detector mirror-noise, not the
relation.

Source of record: `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
`vbench/spatial_relationship.py`, 150 lines,
`source_sha256 = 26f09a111dac904d1a92ae5bd1c332a029cebd22610d62376a09d7d634950270`
(`configs/upstream.toml`), verified byte-identical on the locked checkout; all upstream
line numbers are that revision. Ours: `metrics/spatial-relationship/src/spatial_relationship/`
and `scripts/counterfactual/`; frozen tree
`/root/wenbiao_zhao/datasets/counterfactual-vbench/scores/`. The §4 statistics were
recomputed from that tree on the host (no GPU) and the frame-level counts come from a
re-score that is **80/80 score-identical** to the frozen rows. Every "code fact" was
read in the files, not inferred.

---

## 1. Origin — the complete reduction, video → scalar

**Query.** `info['auxiliary_info']['spatial_relationship']` (`:123`) supplies
`object_a`, `object_b`, `relationship`, read out in `check_generate` (`:99-101`).

**Input and detection.** `load_video(video_path, num_frames=16)` (`:125`), resolution
untouched unless `min(h,w) > 768` (shorter side to 720, `:127-130`); then
`get_dect_from_grit` (`:84-96`) runs `run_caption_tensor(frame)` per frame (`:90`) and
reduces each entry to `[label, box]` (`:92-94`), so frame `t` yields
`D_t = [(l_i, b_i)]`.

**Matching and frame score.** `check_generate` (`:103-114`):

```text
for each frame:
  frame_obj_locats = []                                        :105
  cur_score = [0]                                              :106
  for item in frame_pred:
    if key_a == item[0] or key_b == item[0]:                   :108
      frame_obj_locats.append(item[1])   # box only, no role     :109
    for c_obj1 < c_obj2 over frame_obj_locats:                  :110-111
      cur_score.append(get_position_score(relation, ·, ·))      :112-113
  frame_score.append(max(cur_score))                            :114
```

**Geometry.** `get_position_score` (`:25-82`): centers (`:47-48`),
`x_distance = c2x − c1x` (`:51`), `y_distance = c2y − c1y` (`:52`), IoU (`:55-61`),
then

```text
if locality in 'on the right of' or locality in 'on the left of':      :68
    abs(x_distance) > abs(y_distance) and iou <  0.1  -> 1             :69-70
    abs(x_distance) > abs(y_distance) and iou >= 0.1  -> 0.1 / iou     :71-72
    otherwise                                         -> 0             :73-74
elif locality in 'on the bottom of' or locality in 'on the top of':    :75
    abs(y_distance) > abs(x_distance) ...            same three-way    :76-81
```

**Scalars.** Per-video mean `np.mean(frame_score)` (`:133`); dataset scalar
`np.mean(frame_score_overall)` (`:136`) — a **frame-weighted** mean over the whole
prompt list, not the mean of per-video means (the multi-rank path, `:147-149`,
recomputes it the other way, so the two differ in weighting).

Writing `A_t` for "at least one `{A,B}` pair exists in frame `t`" and `φ` for the
pair rule,

```text
s_{v,t} = max( {0} ∪ { φ(R, b_i, b_j) : i < j,  l_i, l_j ∈ {A, B} } )
S_official = ( Σ_v Σ_t s_{v,t} ) / ( Σ_v T_v )                        :136
φ(R, b1, b2) = 1[axis_dom(R)] · ( 1 if IoU < 0.1 else 0.1/IoU )
axis_dom(R) = 1[|Δx| > |Δy|] for R ∈ {left, right};  1[|Δy| > |Δx|] otherwise
```

Because `s_{v,t} = 1[A_t] · max φ`, the score is **bounded above by the co-detection
indicator**: `S_official ≤ (1/ΣT) Σ 1[A_t]`. (`max_width`/`max_height`, `:64-65`, are
computed and never used; the pair loops at `:110-113` re-enumerate prefixes for the
same `max`.)

---

## 2. Origin — what breaks, and why the nuisance factor is the sign

### F1 (code fact, `:68-81`) — the predicate never reads the sign

Only `abs(x_distance)` and `abs(y_distance)` are used; the file contains exactly six
mentions of the two distances and four of them are inside `abs()` (`:69`, `:71`,
`:76`, `:78`). Formally, `φ` depends on the pair only through
`(|Δx|, |Δy|, IoU)`. Hence `R = on the left of` and `R = on the right of` produce
**identical** frame scores, as do `on the top of` and `on the bottom of`.

### F2 (code fact, `:105-109`) — role identity is discarded

Boxes for `key_a` **or** `key_b` are appended to one list (`:108-109`); the role is
never recorded, and the pair enumeration is unordered (`c_obj1 < c_obj2`, `:110-111`).
The predicate at `:108` is symmetric in `key_a`/`key_b`, so swapping the query to
`(B, R, A)` (plan §9.4's role-swap family) is also an exact no-op for Official.

### F3 (code fact + proof, `:68-82` + `:110-114`) — mirror invariance is constructive

Let a frame's matched boxes be `M_t = (b_1, …, b_k)` and let the mirror be
`m(x, y) = (W − x, y)`. Under `b ↦ m(b)`: `Δx ↦ −Δx`, so `|Δx|` is unchanged; `Δy`
is unchanged; areas, intersections and therefore `IoU` are unchanged (reflection is
an isometry). The matched label multiset is unchanged because it is a property of
the detection list, not of the boxes. Every term of `φ` is therefore invariant for
every pair, the index set of pairs is unchanged, and `max` preserves the value:

```text
s_{v,t}(m(M_t)) = s_{v,t}(M_t)     for every frame and every clip
⇒ S_official(mirror) = S_official(original)     exactly
```

**This is the paper's clean statement of the premise failure: Official is
direction-blind by construction, not by noise.** The only reason the two observed
means differ at all is that GRiT is not exactly mirror-equivariant and the flipped
clip is re-encoded; those differences are upstream of the metric. Measured shadow
(§4.4): pooled level means `original` 0.3104, `horizontal_flip` 0.3108,
`vertical_flip` 0.3319, and an 11-versus-8 split of the 19 discordant test pairs
(exact two-sided sign test `p = 0.6476`).

### F4 (code fact, `:106` + `:114`) — detection failure is scored as a relation violation

`cur_score` is seeded with `0` (`:106`) and the frame takes the `max` (`:114`), so a
frame with fewer than two matched boxes and a frame with two boxes on the wrong axis
are both represented by **0**: abstention and violation are not separable downstream.
The score is therefore capped by co-detection, and any family whose intervention
removes detections is scored as though it changed the relation.

### F5 (code fact, `:68`, `:75`) — substring containment instead of equality

`locality in 'on the right of'` tests **containment**, not equality. Short forms such
as `"left"` are therefore silently accepted, and a relation outside the four supported
strings — e.g. `inside of` — matches neither branch, so every frame scores `0` and the
clip scores exactly `0.0` with no error raised. The fixture's exclusion of the 24
`inside of` prompts (`select_bases.py:78-81`, `docs/counterfactual-dataset.md:59`)
handles this by refusing to build the prompt rather than by scoring it.

### The named nuisance factor and its falsifiable predictions

The group generated by (i) mirroring the frame and (ii) swapping the two roles acts
trivially on `S_official`. The dimension is named for an **orientation**, and the
score is a function of the **unordered, unsigned** pair geometry — so the nuisance
factor is exactly the sign that the mapping drops.

- **P1 (F1/F3).** Mirroring cannot lower the score of a fixed detection list; any
  systematic drop must come from the detector. *Tested in §4.4; not observed.*
- **P2 (F2).** A role swap cannot change the score at all — the matched box multiset
  is identical. *A family built on it measures nothing.*
- **P3 (F4).** `S_official ≤` co-detection rate, so rare co-occurrence is a binding
  ceiling. *Tested on the repair side in §4.3.*
- **P4 (F5).** A relation outside the two literals scores exactly 0 on every frame.

---

## 3. Ours — contract, sign, and the directional gate

**Contract.** The query is ordered: `parse_query` (`metric.py:22-37`) reads
`object_a`, `object_b`, `relationship` from `dimension_metadata` (`:30-36`) and raises
on a missing field rather than inferring from the prompt, so `(A, R, B)` is carried
end to end.

**Signed frame rule.** `relation.ordered_position_score` (`relation.py:57-73`) keeps
F1's axis test and adds the sign. `center_distances` (`:26-29`) returns
`object_center − subject_center`, so `dx > 0` means the object lies right of the
subject; `direction_match` (`:63-68`) maps left `↦ dx > 0`, right `↦ dx < 0`, top
`↦ dy > 0`, bottom `↦ dy < 0`; `:69-72` returns `0` unless **axis dominance and
direction match** both hold, keeping the official partial credit `0.1/IoU` for parity
with `:71-72`. The ported official rule `official_position_score`
(`relation.py:44-54`) is retained for parity tests and deliberately has no sign term.

**Identity-preserving binding.** `evaluate_frame` (`models.py:75-158`) filters
candidates by exact label, reporting `missing_subject`/`missing_object` separately
(`:85-88`), assigns one instance per role (`:128-129`), rejects a shared detection
(`:134-135`), and labels the outcome `direction_mismatch` (`:143`) versus
`axis_mismatch` (`:145`) — the two cases F4 collapses into one zero. `AblationMode`
(`models.py:10-14`) adds `ordered_role`, which maximises the signed score over every
role pair instead of committing to one instance, and
`diagnostics.aggregate_video` (`diagnostics.py:27-56`) adds detection-conditioned
averaging, which removes F4's abstention term from the denominator.

**The family.** `transforms.directional_flip` (`transforms.py:240-264`) mirrors the
axis named by the relation (`flip_axis`, `:271-277`; horizontal is
`frames[:, :, ::-1, :]`, `:244`) and emits `original` (`expected_rank = 1`, `:252`)
and `<axis>_flip` (`expected_rank = 0`, `:260`); `build.py:442-447` turns the ranks
into the declared `expected_relation`, so the family asserts
**`score(original) > score(flip)`** with the prompt held fixed.

**The directional gate.** `pick_detectable.relation_frame_rate` (`:103-127`) counts,
over frames in which **both** targets were natively detected (`:120-123`), the
fraction where the signed rule scores above zero (`:125`); `detector_verdict`
(`:130-173`) then requires the co-detection fraction to reach
`--min-co-detected-frames` (`:159-166`) and the relation rate to reach
`--relation-min-frames` (`:167-172`). Kept bases carry `relation_oracle`,
`relation_frame_rate`, `relation_frames_scored` and `relation_frames_total`, copied
into every manifest row by `build.py:502-503`. This is plan §9.2's "original relation
confirmed by a validity check": the premise is asserted only for clips that satisfy
it.

**Two honest limits.** (i) The gate consumes the *same* GRiT boxes and geometry the
repair scores, so it is not an independent oracle: a gated family asks whether the
repair can rank a verified arrangement above its mirror, not whether it can discover
the arrangement. (ii) The gate cannot manufacture presence — the source clips
frequently lack one named object (§4.3), which is why running it on the published
bases empties the family rather than repairing it (§4.5).

---

## 4. Experiment

All figures: `scores/spatial_relationship__cpa.json`, the per-clip
`scores/spatial_relationship__{official,repair}.jsonl` and the report
`docs/counterfactual-reports/spatial_relationship.md` (code SHA `5c4a1309`),
recomputed on the host with `scripts/counterfactual/verify_spatial_review.py`
(`PASS: every quoted number reproduces`). Coverage `cpa.json["coverage"]`: dev 20/20,
test 60/60, both backends.

### 4.1 The row, decomposed (§3, claim: the row is not a Repair deficit)

`official 0.3667 → repair 0.0667`, paired `Δ −0.3000`, 95 % CI
`[−0.4667, −0.1333]` over the same 30 test clusters. Decomposing the 30 rank-gap-1
test pairs by outcome:

| backend | correct | tied 0:0 | inverted | match | tie |
|---|---:|---:|---:|---:|---:|
| official | 11 | 11 | 8 | 0.3667 | 0.3667 |
| repair | 2 | **25** | 3 | 0.0667 | **0.8333** |

Zero-margin CPA counts a 0:0 tie as an error, so `−0.3000` is 25 pairs the Repair
leaves *exactly tied* plus 5 it resolves (2 correctly, 3 not). **Supports §3** (the
signed rule resolves the arrangement or abstains, never half-resolves it) and
**rebuts** reading the row as "the repair is worse": the Repair has 3 inverted pairs
against Official's 8.

### 4.2 Where the zeros come from (§3, claim: the repair separates sign when it has evidence)

On the frozen tree the Repair is exactly `0.0` on **70 of 80** clips; only **8 of 40**
bases have any non-zero level and **5 of 40** a non-zero `original`. In the test split
only **5 of 30** bases have non-constant scores, splitting 2 correct / 3 inverted —
the entire content of `0.0667`. Sequence statistics agree: Repair median per-base
Spearman `−1.0000` (mean `−0.2000`, strict order 2/30) against a declared increasing
order, versus Official `+1.0000` / `+0.1579` / 11/30. **Supports §3**: the sign term
works where there is anything to score; there almost never is.

### 4.3 Why there is nothing to score (§3, claim: the limit is target presence)

Frame evidence from a re-score with the shipped code (80/80 score-identical to the
frozen rows), over the 40 `original` clips (640 frames):

| `frame_reason` | frames | share |
|---|---:|---:|
| `missing_object` | 278 | 43.4% |
| `missing_subject` | 230 | 35.9% |
| `direction_mismatch` | 72 | 11.2% |
| `axis_mismatch` | 22 | 3.4% |
| `relation_satisfied` (+ IoU penalty) | 38 | 5.9% |

**508/640 = 79.4 % of frames never reach the geometry check** (80.5 % over all 80
clips), and only 38 of the 132 resolved frames satisfy the relation. By base: **23 of
the 40 have no frame with both roles detected**, 12 partial, 5 complete. **Supports
§3's second limit**, and is the measured form of F4's ceiling: the abstention is
real, not an artefact of scoring zero.

### 4.4 Official's half is not a measurement either (§2, claims F1/F3)

Pooled level means, frozen tree: `original` 0.3104, `horizontal_flip` 0.3108,
`vertical_flip` 0.3319 — equal within noise, exactly as F3 requires. Of the 30 test
pairs 19 disagree, splitting 11 (original) versus 8 (flip); the exact two-sided sign
test gives `p = 0.6476`. **Supports §2**: Official's residual variation is detector
mirror-noise, and the premise is false for it by construction, not by sampling.

### 4.5 The gate, run for the first time (§3, claim: the gate enforces plan §9.2)

The gate had never been applied here — the published 40 bases came from plain
`select_bases.py`. Run now (`--relation-oracle detector --relation-min-frames 0.75
--min-co-detected-frames 0.25`): **scanned 84, rejected 78, kept 6** (dev 3, test 3),
exit 1, `only 3/10 eligible candidates`. Relaxing both thresholds to 0.50/0.10 gives
kept 8 (dev still 3/10); box presence alone (`--relation-oracle none`) fills dev but
exhausts the test pool at **22/30**. Rejections at the declared thresholds: 46 box
presence (a named object never detected in *any* frame), 17 co-detection floor, 15
directional. Probing the published 40: **20 pass box presence, 3 pass the gate**, 18
have ≥1 measurable frame, and 10 of those 18 satisfy the relation in **0.00** of them.
**Supports §3** (the gate is the §9.2 check and it surfaces the fixture's failure) and
**rebuts** any assumption that a gated family of the declared size exists. Sources:
`/root/wenbiao_zhao/spatial-verify/gate*.log`, `published_bases_probe.json`; reproduce
with `verify_spatial_review.py --probe-bases configs/counterfactual/bases_published.jsonl`.

---

## 5. Consistency with this dimension's review, and the redesign call

Re-checked against `docs/counterfactual-reports/spatial_relationship.review.md`:

- **Review §3 (the zero floor is the detector, not the sign).** *Still stands* and is
  re-verified in this round: 508/640, 70/80, 23/40 all reproduce. §2's F1–F5 are the
  mechanism statement the review inferred from the frame reasons.
- **Review §5 (plan §9.2 was never carried out).** *Still stands*; §4.5 is the same
  measurement. The review's "gate admits 2 of 40" is corrected to **3 of 40**.
- **Review §5.1 (the gate breaks the documented 205/205 reproduction).** *Still open,
  and this card depends on it*: with spatial inside
  `pick_detectable.DETECTOR_DIMENSIONS` (`:41`, oracle default `:281`) the published
  spatial subset cannot be regenerated. Fix before citing this fixture.
- **Review §5.2 (the published report's `Frame evidence` is empty).** *Still open*:
  §4.3 comes from a re-score outside the frozen tree.

**Redesign or delete.** Criteria: (1) does the paper need a *Repair-side* claim here?
No — §2 and §4.4 already give a complete, sample-size-independent statement about
Official. (2) Can the fixture be salvaged? Measured no: at 0.75/0.25 the gate keeps
6/84 and dev stays at 3/10 under every relaxation tried, and the binding constraint
(46/84 candidates never ground one object) is threshold-independent. (3) Does the
source have the property at all? Only 4 of the 18 measurable published bases satisfy
the relation in ≥80 % of frames, and selecting on them is circular.

Costs. **Delete** the row from the Repair comparison and keep §2 as the dimension's
result: ~1 hour of writing, no compute. **Rebuild** as a controlled composition from
VBench crops (real objects pasted at construction-known coordinates, registered as a
*new* family, not as `directional_flip`): ~1 day of engineering plus ~25 minutes of
GPU for 120 clips × 2 backends; it removes the circularity and the budget problem but
remains detector-conditioned. **Per-video tracking inside the Repair** is rejected: it
changes the metric contract without fixing the premise. Recommendation: delete now,
rebuild only if a Repair-side spatial claim is required.
