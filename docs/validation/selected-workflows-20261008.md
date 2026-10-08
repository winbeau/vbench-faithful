# Selected workflow cleanup validation — 2026-10-08

The active tree retains the 16-dimension evaluator and the selected paper
training/construction chains. The starting commit is `8939690`;
[training.md](../training.md) identifies retained versions and necessary parents.
Historical candidates remain available from that commit.

CPU validation: **1,127 passed, 3 skipped**, locked workspace check/sync, and
all 16 dimension CLI help entries. The final training input/provenance path
adjustments also passed **21 integration tests**, including help without
models, source-family exclusion, spatial direction conversion, scene source
components, manifest ordering and relocated K400 loading. These checks do not
constitute retraining.

A fresh H100 run evaluated the same **32 videos in all 16 dimensions**. Each
video is 2 seconds, 16 frames, 8 FPS. The nine repairs have identical per-record
scores and states to the pre-cleanup outputs. All **224/224** accelerated scores
pass the agreed official per-video error gate. Motion Smoothness has a maximum
absolute difference of **2.85e-8** from its previous accelerated run; the other
15 dimensions are exact. All **512 result states** are unchanged.

Coverage remains **459/512 scored**, with the same 53 null results: Subject 9,
Spatial 21, Color 23. Exit code 1 preserves that partial coverage. Fresh wall
time was **373.26 seconds**, including new environment preparation on a shared
H100. This is cleanup validation, not a new controlled speed comparison;
the official scores are the previous same-cohort reference. See the
[existing speed report](h100-same32-final-20261008.md).

The GPU snapshot is `f515570755cabd9270622d15a46eaf22c3f770b9`
(tree `67f60805410de47a5eb0c158e387b0724004ac35`). Subsequent edits concern
training/data paths, unused construction selectors, documentation and tests;
scoring source, model selection and media are unchanged. The
[machine-readable receipt](selected-workflows-20261008.json) records upstream,
GPU UUID, per-dimension checks, cohort hash and remote output locations.

No model was retrained and no frozen dataset or paper score table was rewritten.
