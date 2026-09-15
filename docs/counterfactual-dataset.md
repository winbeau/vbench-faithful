# Counterfactual dataset construction (VBench-CF)

Code: `scripts/counterfactual/`.  Plan: [`plans/2026-09-14-experiment-plan-8d.md`](plans/2026-09-14-experiment-plan-8d.md).

`VBench-CF` is the metamorphic test set derived from the public VBench 1.0
human-preference videos.  Each family applies one deterministic transformation
whose effect on the score is known before any metric is run, so a violation is
attributable to the metric rather than to a judgement about which video is
better.

## Families

| Dimension | Family | Transformation | Expected relation |
|---|---|---|---|
| Dynamic Degree | `fps_resampling` | resample one trajectory to 8/6/4/2 fps, duration fixed | invariance (all rungs tie) |
| Subject Consistency | `temporal_relocation` | one fixed subject corruption at start / middle / end | clean > corrupted; three positions tie |
| Human Action | `filename_invariance` | byte-identical copies under correct / wrong / neutral names | invariance |
| Spatial Relationship | `directional_flip` | mirror the axis named by the relation | original > flip |
| Scene | `environment_coverage` | 2x2 grid mixing wrong-scene donor with target quadrants at 0-100% | monotone increasing |
| Multiple Objects | `weakest_object_visibility` | alpha-blend tracked target B towards its local mean at 0-100% | monotone decreasing |
| Motion Smoothness | `temporal_jerk` | duplicate / skip / reverse short local segments | monotone decreasing |

Overall Consistency (condition substitution) is deliberately out of scope for
this round: plan section 12.2 requires human-authored conditions, so it cannot
be generated from metadata alone.

## Decisions that shape the result

**The FPS ladder only downsamples.**  Every MP4 source is 8 fps / 16 frames /
2.0 s and every CogVideo GIF is 10 fps / 33 frames / 3.3 s, so no source exceeds
10 fps.  The plan's 8/12/16/24/30 ladder would therefore have to *invent* frames
for four of its five rungs.  Nearest-neighbour duplication injects temporal jerk
and would contaminate Motion Smoothness; interpolation needs a model and would
stop being deterministic.  The ladder is therefore 8/6/4/2 fps, and the retained
frames are an exact subset of the source trajectory.

**Every family has a re-encoded level 0.**  If the control were a byte copy, it
would differ from its siblings by an extra generation of compression and any
score gap could be blamed on the encoder.  The unmodified clip is still stored
under `<dimension>/source/` so the control can be checked against the original.

**Boxes are tracked per frame.**  Plan section 7.5 keeps a fixed ROI for an
ablation only, so the primary families use a per-frame box.  GRiT runs on every
frame; a detector drop-out inherits the nearest detection rather than punching a
hole in the suppression region, and a base whose target is never detected is
rejected instead of being silently generated.

**GRiT is queried the way VBench queries it.**  GRiT is caption-free: it emits
its own category names, so targets are matched as bare nouns taken from the
official annotations (`object_en` = "cat and dog", `subject_en` = "person") and
never from `prompt_en`, which matches nothing.  Frames are fed as **RGB** to
match `vbench.utils.load_video`; detectron2's own BGR convention yields different
detections.  `ObjectDet` mode is used because its short category names are the
ones the annotations use.

**Base selection never looks at a score.**  Bases come from the frozen E0
prompt split, one per eligible prompt, chosen by stable hash
(`select_bases.py`).  Prompts that cannot support their contract are rejected
rather than mistransformed — notably the 24 `inside of` Spatial prompts, which
the locked Official evaluator does not support.

**The directional family proves its own premise.**  `original > flip` is only the
right expectation when the source clip actually places A on the required side of B,
and plan section 9.2 requires that to be confirmed before the family is used.  The
first build never checked it, so a direction-sensitive repair was scored against a
coin flip.  `pick_detectable.py --dimension spatial_relationship` now gates Spatial
Relationship candidates on two things: at least `--min-co-detected-frames` (default
0.25) of frames must natively detect both targets, and at least
`--relation-min-frames` (default 0.75) of *those* frames must satisfy the ordered
relation.  A failed gate prints the observed rate distribution so the thresholds can
be tuned against real data instead of guessed.  The gate reuses the audited geometry
on the shared GRiT boxes, and every kept base records `relation_oracle`,
`relation_frame_rate`, `relation_frames_scored` and `relation_frames_total`, which
`build.py` copies into each manifest row.  It is therefore *not* an independent
oracle: a repair that consumes those same boxes is being asked to rank a verified
arrangement above its mirror, not to discover the arrangement.  Pass
`--relation-oracle human --relation-validity <file>` to use human verdicts instead,
or `--relation-oracle none` to reproduce the unverified family.

## Running it

```bash
# 1. choose bases (metadata only, no videos or models needed)
uv run --no-sync python -m scripts.counterfactual.select_bases

# 2. re-select the GRiT-dependent dimensions with oversampling.  This is also
#    where the Spatial Relationship directional gate runs, so Spatial bases are
#    replaced by ones whose original clip actually satisfies the relation.
uv run --no-sync python -m scripts.counterfactual.pick_detectable \
  --bases output/counterfactual/bases.jsonl \
  --dataset-root <vbench-1.0-human-preference> \
  --dimension spatial_relationship

# 3. generate (on a machine with the videos; --detector is required because the
#    two box-presence dimensions and the Spatial gate all ground targets)
uv run --no-sync python -m scripts.counterfactual.build \
  --bases output/counterfactual/bases.jsonl \
  --dataset-root <vbench-1.0-human-preference> \
  --output-root <counterfactual-vbench> \
  --detector default --dimension spatial_relationship

# 4. verify: structure, split isolation, hashes, duration/frame checks and a
#    byte-exact replay of every derived clip from its recorded parameters
uv run --no-sync python -m scripts.counterfactual.validate \
  --root <counterfactual-vbench> --dataset-root <vbench-1.0-human-preference>
```

The manifest records the plan's section 3.2 provenance fields plus the
transformation parameters needed to replay each clip.  Because the detector box
is stored as a parameter, validation never needs a GPU.

## Scoring variants

`directional_flip` cannot separate a direction-blind metric from a direction-sensitive
one unless the repair contract is stated.  The Spatial Relationship repair therefore
exposes plan section 9.6's two settings through the scoring driver:

```bash
# end-to-end (default): undetected frames count as relation violations
uv run --no-sync python -m scripts.counterfactual.run_dimension \
  --dimension spatial_relationship ... --repair-mode ordered_role_identity_assignment

# detection-conditioned ordered-role repair: the configuration that matches the
# gated, relation-verified base set above
VBENCH_AUDIT_SPATIAL_MODE=ordered_role \
VBENCH_AUDIT_SPATIAL_DETECTION_CONDITIONED=1 \
uv run --no-sync python -m scripts.counterfactual.run_dimension \
  --dimension spatial_relationship ...
```

- `ordered_role_identity_assignment` (default) commits to the highest-confidence instance
  per role; `ordered_role` maximises signed geometry over every `(subject, object)` pair.
- detection conditioning averages only over frames where both roles were detected, so a
  detector drop-out leaves the denominator instead of scoring as a wrong direction.

Each scored row records `repair_mode` and `detection_conditioned`, the generated report
header repeats them, and a shard file refuses to accept a second configuration: score a
different variant into a fresh `--scores` tree.  A variant change also needs a fresh
dataset root, because the manifest rows carry the eligibility provenance of the bases
they were built from.

## Budget

Plan section 4 recommends 50 dev + 180 test bases.  With Overall Consistency
deferred this is 45 dev + 160 test bases and roughly 950 derived clips.
