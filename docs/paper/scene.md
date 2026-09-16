# scene — origin vs ours (paper material)

**Thesis.** VBench 1.0's Scene score is a **lexical hit-rate over sampled frames**:
a frozen caption model writes one free-text sentence per frame, and the frame counts
as a success iff *every whitespace-separated token of the scene label occurs as a
substring* of that caption. The score is the fraction of sampled frames in which the
caption model's vocabulary **collided** with the label — not the fraction of the frame
in which the scene is present. The nuisance factor this dimension is audited against
is **partial coverage**: how much of the frame/volume the named environment occupies.
The official functional has no term that can see extent, so it is blind to partial
coverage by construction, and it is also *under-determined* about full coverage: it
returns exactly 0.0 on 9 of the 25 base videos it is handed. The repair makes coverage
explicit — frame score = global scene support × mean support over four canonical
quadrants — which injects the ordering the contract requires, but is by design
isomorphic to the injected transformation, so the counterfactual improvement is a
wiring check rather than evidence that the repaired metric is better.

Source of record: `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
`vbench/scene.py`, 79 lines,
`source_sha256 = f14c423710c812af2be953db619ed6e8b478acdc44f1df775815adfa30171965`
(`configs/upstream.toml`), verified on the locked scoring checkout; helper
`vbench/utils.py` (`tag2text_transform` `:63-66`, `get_frame_indices` `:68-88`,
`load_video` `:108-185`, metadata loader `load_dimension_info` `:202-227`). Ours:
`metrics/scene/src/scene/`, `scripts/counterfactual/`, report
`docs/counterfactual-reports/scene.md` at report code SHA `5c4a1309`.

**Provenance caveat before quoting the report's header.** `scene.md`'s
`repair variant: ordered_role_identity_assignment` line is wrong for this dimension:
that identifier is Spatial Relationship's role-assignment ablation, the field is
emitted for every dimension by the report template, and its default is read from
`VBENCH_AUDIT_SPATIAL_MODE` (`run_dimension.py:1119-1121`, `:524-531`), even though
the scoring driver documents the setting as spatial-only (`score.py:243-244`). The
scene repair's actual identity is `scorer=openclip`, `ViT-B-32`,
`mode=environment_grounded` (`score.py:335-341`). The report's `code SHA` is a
*render* revision as well, not the scoring revision (`66c4a99`); see
`scene.review.md` §7.2 and §3.

Every number in §4 was recomputed on `h100-server` from the frozen trees with
`scripts/counterfactual/verify_scene_review.py` (CPU only; no GPU, no re-scoring,
nothing written into `data/`, `results/`, `splits/`, `runs/` or the frozen dataset).

---

## 1. Origin — the complete reduction, video → scalar

**Input.** One video path and one scene label `key` from the prompt metadata
(`auxiliary_info.scene.scene.scene`, `scene.py:45`; mirrored at
`backends/vbench.py:32-41`).

**Frame sampling and resolution.** **16 frames**, indices = midpoint of each of 16
equal intervals (`get_frame_indices`, `utils.py:69-86`, `sample="middle"`), decoded
straight to **384×384** (`scene.py:47`) and normalised per frame before captioning
(`utils.py:63-66`, called at `scene.py:41,50`). The caption model never sees the
native-resolution frame.

**Caption model.** Tag2Text, Swin-B, 14M weights (`submodules_dict` =
`{"pretrained": "caption_model/tag2text_swin_14m.pth", "image_size": 384,
"vit": "swin_b"}`, our copy `backends/vbench.py:16-20`), called with
`tag_input=None`, `return_tag_predict=True`, keeping only the caption
(`scene.py:25-27`); all 16 frames in **one batch** (`scene.py:48-52`).

**Match rule.** (`scene.py:29-36`)

```text
key      = scene_label                        # free text, e.g. "train station platform"
tokens   = key.split(" ")                     # whitespace split, no normalisation
flag(w)  = every token q satisfies  q in caption_w        # raw Python substring test
success(w) = 1 if flag(w) else 0              # exactly the check above
```

**Aggregation.** (`scene.py:54`, `:62`)

```text
video_score   = (1/16) * Σ_{w=1..16} success(w)            # :54, per video
dataset_score = Σ_w success(w) / Σ_w 16                    # :62, frame-weighted pool
```

Under `world_size > 1` the pool is recomputed from the gathered per-video
`success_frame_count`/`frame_count` (`:74-78`), which is arithmetically the same
frame-weighted mean. **The weights are uniform over frames and over videos** — a
video contributes in proportion to its frame count (16 for all clips in this audit),
and no prompt-, scene- or model-level weight exists. The label enters only through
the token list.

**Consequences of the rule, read off the code.** The test is a *bag of substrings*:
it ignores word order (`platform` + `train` + `station` anywhere passes for
`train station platform`), syntax and negation, it is not lemmatised, and any token
that is a common caption word fires regardless of meaning. VBench scene prompts are
`"a"`/`"an"` + label (`a phone booth`), so the article is almost always present and
the *effective* test on this benchmark is the one content word.

## 2. Origin — what is actually broken

**(a) Code facts.** The successes are counted per frame and averaged (`scene.py:33-35`,
`:54`, `:62`); the label is matched token-wise as a substring (`scene.py:33`); the
caption is free text from a generative captioner (`scene.py:26`); frame size is
384×384 (`scene.py:47`). None of these lines references spatial extent, area,
proportion, mask, box, or the fraction of the frame occupied by the environment.

**(b) Formalisation.** For scene `c` with label tokens `T(c)`, caption `κ_w` emitted
for frame `w`, and `N = 16` sampled frames:

```text
S_official(v, c) = (1/N) Σ_w  ∏_{q ∈ T(c)}  1[ q ⊑ κ_w ]                    (1)
```

with `⊑` the substring relation. Two failure modes follow and are *different*:

```text
false negative (miss):      environment present in w, but q ⋢ κ_w  → success(w)=0
false positive (collision): environment absent, but q ⊑ κ_w        → success(w)=1
```

The nuisance factor here is **partial coverage**: let `a(w) ∈ [0,1]` be the fraction
of frame `w` occupied by the named environment. A contract "score is monotone in `a`"
requires `S` to depend on `a`. Eq. (1) depends only on the *token-membership set* of
the caption, so

```text
∂S_official / ∂a  ≡  0    except where a caption token appears or disappears:
                          S is piecewise constant in coverage and can move only by
                          a whole frame (1/16), never by an area fraction.
```

Shrinking the environment from 100 % to 40 % of the frame while the caption still says
"beach" leaves the score **exactly unchanged**; enlarging it cannot help once the
token is present. That is the formal statement of "cannot express partial coverage".
The repaired functional (§3, eq. 2) inserts an explicit spatial term, but over four
*equal-area* views, so its region average is only a coarse proxy for `a`: a scene on a
diagonal or in the frame centre is under-counted relative to one covering two
quadrants of equal area. That is a limitation of the repair, not of the fixture.

**(c) Which nuisance factor, and why the two readings differ.** The score answers
"did the caption model's vocabulary collide with the label?" and not "does the video
depict the scene?". They differ in both directions: a frame whose caption reads
"a boat on a lakehouse dock" fires `lake` by substring with no lake in the scene
(false positive, and the collision is *inside a longer word*); a frame that clearly
shows a bakery but is captioned "a person buying bread" scores 0 (false negative,
because the captioner prefers the action). A lexical metric measures the captioner's
vocabulary on the sample, so its noise is the captioner's, not the environment's.

**(d) Measurable predictions, with the frozen data.** From `verify_scene_review.py`
on the frozen score tree (125 clips):

| prediction | measured |
|---|---|
| every official score lies on the `k/16` lattice (eq. 1, `N = 16`) | **0 / 125 off-lattice**; histogram `0/16=91, 1/16=1, 3/16=6, 4/16=1, 6/16=1, 7/16=3, 9/16=1, 10/16=3, 12/16=2, 13/16=5, 15/16=2, 16/16=9` |
| multi-token labels fail more often (more chances to miss) | 1-token labels: mean **0.4485**, 5/17 exactly 0; 2-token: mean **0.1786**, 4/7 exactly 0; 3-token (`train station platform`): 1/1 at 1.0000 |
| the reference positive is not saturated (the metric is not "detecting the scene" reliably) | `coverage_100` = the untouched base video: mean **0.3950**, median **0.1875**, **9/25 exactly 0**, 16/25 below 0.5 |
| the all-donor level is a clean floor | `coverage_000`: **0.0000 for all 25** clips |

The 2-token row is the sharpest single piece of evidence for the collision reading:
"phone booth", "indoor library" and "science museum" are *easier* scenes than
"aquarium" or "cafeteria", yet they score worse, because the rule asks the captioner
to have produced two specific words in one sentence.

## 3. Ours — the environment_coverage fixture and the repaired score

**Fixture.** `transforms.environment_coverage` (`transforms.py:317-360`) builds five
clips per base from **two** videos: the *target* (the base clip, whose label is the
scene under test) and a *donor* (a different scene, same generator and split). Level
`k` copies the target's pixels into exactly `k` of four canonical quadrants of a
donor base frame (`transforms.py:303-314`, `:340-345`):

```text
x_k = donor,  x_k[Q_i] = target[Q_i]  for i < k,   k = 0..4
Q = {top_left, top_right, bottom_left, bottom_right}      :303
expected_rank = k,  target_fraction = k/4                 :351-354
```

so `coverage_000` is the pure donor scene, `coverage_100` is the base video
(re-encoded, never a byte copy — verified: 25/25 carry a fresh `output_sha256`), and
the declared relation is *monotone increasing in target coverage*
(`FAMILY_REFERENCE["environment_coverage"] = "coverage_100"`, `build.py:56`). The
contract this fixture asserts is a **coverage contract** — more of the frame is the
named environment, so the score must not go down — and it is deliberately a spatial
nuisance, orthogonal to the scene's identity.

**Repaired score.** A CLIP image-text scorer (`models.py:40-80`: ViT-B-32, cosine
mapped to `[0,1]` by `(cos+1)/2`) is queried once on the whole frame and once on each
of four non-overlapping quadrant crops (`backends/audit.py:9-26`), and the frame
score is their product (`backends/audit.py:38-49`):

```text
s_frame = s_global × (1/4) Σ_{i=1..4} s_quadrant,i      (2)
S_repair = (1/16) Σ_w s_frame(w)                        metric.py:73-79
```

run at the locked 384×384 / 16-frame sampling (`metric.py:39-55`) so that only the
functional differs, and recorded as `formula_version = global-times-regional-mean-v1`
(`cli.py:95`, variant `--audit-variant environment_grounded`, `cli.py:26`).

**What this does and does not fix.** Eq. (2) makes the score depend on the *spatial
distribution* of support, so a scene confined to one quadrant can no longer carry the
frame, and coverage becomes an explicit factor — the property eq. (1) lacks. It does
**not** fix: (i) the captioner/collision coupling, because it replaces the lexical
test rather than repairing it — the comparison in §4 therefore confounds *model*
(Tag2Text vs CLIP), *query form* (caption string vs label embedding) and *aggregation*
(presence vs `global × regional`); (ii) the absolute calibration, since the global
factor is not calibrated to reach 1 (see §4); (iii) the region term's own semantics:
the four quadrants are a fixed 2×2 partition, so "coverage" is quantised to five
levels and a scene occupying a diagonal or the frame centre is under-counted.

**Donor determinism and replay.** The donor is chosen by a seeded stable hash over
the eligible pool — same dimension, generator and split, different `prompt_id`
(`build.py:408-431`), with `stable_sample(..., f"{SEED}:donor:{base_id}")` and
`f"{SEED}:donorvideo:{base_id}"`. Re-running it reproduces the manifest for **25/25**
bases and is bit-identical across repeats (24 distinct donors; one shared by two
bases). Provenance is complete in all 125 rows (`video_uid`, `prompt_id`,
`relative_video_path`, `generator`, plus `quadrant_order`, `target_quadrants`,
`target_fraction`; 0 problems), and the committed verifier re-derives **all 125 clips
byte-exactly from the recorded parameters alone** (`validate.py`, `replayed: 125`,
`mismatches: 0`, 82 s) — no detector, no GPU. Gap: the donor's *content* is pinned by
`video_uid` + path, not by a hash (0 rows record a donor `sha256`), so an upstream
package change would surface as a replay byte-mismatch rather than as a provenance
failure.

## 4. Experiment — 0.3850 → 0.9300, and why the row is not a Repair win

Setting: 25 bases (5 dev / 20 test), 125 clips, both backends scored on all of them
(no failures, no incomplete shards). CPA is over all ordered level pairs at zero
margin; the dev-calibrated margin is **0 for both backends**, so the tie-aware and
zero-margin columns coincide — the result is not a margin artefact.

| quantity | official | repair | source |
|---|---:|---:|---|
| CPA, test (200 pairs) | **0.3850** | **0.9300** | `scene.md` `## CPA`; recomputed |
| paired delta over `base_id` clusters | — | **+0.5450 [+0.4249, +0.6800]** | `scene.md` `## Official vs Repair`; recomputed (2000 resamples, seed 2026) |
| pairs on which the backend **commits** a sign | **82 / 200** | 200 / 200 | recomputed |
| exact ties | **118** | 0 | recomputed |
| accuracy **on the committed pairs** | **0.9390** (77/82) | 0.9300 (186/200) | recomputed |
| CPA if an exact tie is credited | **0.9750** | 0.9300 | recomputed |
| strict order over the 5 levels | **0 / 20** bases | **14 / 20** bases | `## Sequence-level order statistics`; recomputed |
| mean `coverage_100 − coverage_000` | +0.4344 | **+0.0445** (1 base negative) | `## Score sensitivity`; recomputed |
| score span over all 125 clips | 0.0000 – 1.0000 | **0.3363 – 0.4317** | recomputed |

**Why the two headline numbers "contradict" each other.** They do not measure the
same event. The official metric resolves only **82 of 200** test pairs: 118 pairs
receive *identical* scores at two different coverage levels, because the score is an
integer count out of 16 and the ladder's per-frame caption outcomes do not change
enough to move that count — 91 of the 125 clips sit at exactly `0/16`. A strict-sign
CPA scores every abstention as a miss, so 0.3850 is `77/200` — a **tie-count**
statistic, not a ranking statistic. Restricted to the pairs official is willing to
decide, the two backends are indistinguishable: **both correct on 77, both wrong on
0, official-only 0, repair-only 5** (exact McNemar `p = 0.0625`), and official is
*perfect* at rank gaps 3 and 4 (20/20 and 14/14) — precisely the large coverage
differences the dimension exists to measure. So the +0.5450 delta is produced almost
entirely by the repair's refusal to tie, not by better ordering. (The 0.3850 and
0.9300 do not literally contradict; they are two different statistics over two
different pair sets. The contradiction is between the *reading* "official cannot rank
scene coverage" and the measured 93.9 % accuracy on everything official chose to
rank.)

*Inference, flagged as such:* why the count fails to move is not established here —
the frozen rows store scores only, so the per-frame captions (and therefore whether
the ladder shifts the *sampled* frames' captions at all) were never recorded. The
measured tie rate is a fact; attributing it to caption quantisation versus
sample-position luck requires the frame evidence the report says is missing.

**Why this row cannot be cited as a Repair win.** Three independent reasons, each
sufficient: (i) the delta is a tie-count effect, per the decomposition above;
(ii) the repair's functional (eq. 2) is built from the same quantity the fixture
varies (quadrant coverage), so the counterfactual ordering is the design's expected
output — a wiring check, not a discovery; (iii) the ablation that would let the delta
be attributed — scoring the same clips with `--audit-variant global` (`cli.py:26`,
`formula_version = global-image-text-score`, `cli.py:95`) — has **never been run**,
and cannot be reconstructed, because the frozen rows store only the composite and the
report's own `## Frame evidence` section records that no per-clip evidence exists.
The repaired score's absolute range reinforces (ii): 0.0955 wide, a 0.3586 floor on
clips with *no* target quadrant, and 0.4031 on the untouched base video — a ~0.045
tilt on a 0.34 floor.

**The one informative number is the Official one.** The official metric scores the
video it was handed — `coverage_100`, the base clip of the very prompt whose scene
label it is compared against — at **mean 0.3950, median 0.1875, with 9 of 25 clips at
exactly 0.0000** (test split: mean 0.4344, 6/20 at zero); on the all-donor level it
is 0.0000 for all 25. Recomputed from the frozen scores, that diagnosis holds and is
about *resolution*, not blindness: a metric that returns the floor on 36 % of its own
reference positives, and an identical score on 59 % of a five-rung ladder, is a 16-bin
frame hit-rate whose per-frame evidence is too sparse to resolve coverage. This is a
statement about the official functional, independent of any repair, and it is the
only claim this dimension can carry. The diagnostic also bounds the fixture: with
9/25 bases unresolvable at baseline and 6/20 test bases giving a constant official
profile, a ladder violation has two indistinguishable explanations (the metric cannot
rank coverage vs the metric cannot see the scene), which is exactly the premise gate
that `spatial_relationship` got and this family did not
(`pick_detectable.py:15-24`, `:54`).

**Which number supports which claim of §3.** The 82/200 commit rate and the 118 ties
*refute* the strong reading of the repair's win and support the "fixture is
isomorphic" caveat; the 0.9390 accuracy on committed pairs *supports* keeping the
official diagnostic and *refutes* "official is globally blind"; the 0.0445 mean slope
and the 14/20 strict-order rate *support* the narrower claim that the repaired
functional is monotone in coverage (eq. 2 does what it says); the 0.3950 base-video
mean with 9/25 zeros *supports* the coverage-vs-collision reading of §2 and the
resolution diagnosis; the 0 donor hashes *limits* the replay claim of §3 to
parameter-level, not content-level, provenance.

---

## Consistency with this dimension's review

`scene.review.md` §4 named two generator defects. Both are fixed in the current
report and code, verified by running the checks rather than reading the commit
message (`verify_scene_review.py` section 7):

1. **The dev+test-mixed `## CPA by contract half` section.** `contract_split_cpa` is
   now called per split — `run_dimension.py:1225-1230` builds
   `{split: contract_split_cpa(split_rows, ...) for split, split_rows in (("dev", dev_rows), ("test", test_rows))}` — and the frozen `cpa.json` shows disjoint
   `dev`/`test` halves (50 and 200 pairs). The section does not appear in `scene.md`
   at all, because the renderer gates the whole mixed-contract template on a shared
   rank (`run_dimension.py:576-579`, `mixed_family = len(ranks) > 1 and bool(dispersion_rows)`);
   `environment_coverage` has five distinct ranks, so `dispersion_rows` is empty.
2. **The borrowed `temporal_relocation` prose.** Same gate: `scene.md` contains no
   "relocated", "must tie" or "widening dev margin" text (grep returns nothing), and
   `score_profile` is computed per split (`run_dimension.py:1218-1219`), which is what
   removed the 25-clips-per-level rows the old prose referred to.

One defect the review raised is **not** fixed and is reported above rather than
assumed away: the ablation of §4(iii) is still absent, and the report's
`## Frame evidence` section states that no per-clip evidence was recorded, so the
attribution question cannot be answered from the cached run.
