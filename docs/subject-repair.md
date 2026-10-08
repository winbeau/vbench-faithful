# Selected Subject Consistency repair

The paper uses **frozen hybrid-v5**, evaluated on the **official720 v9** cohort.
The unified evaluator selects this method through `--backend ours`; use
`--dimensions subject_consistency` to run it alone. The method is pinned in
[`paper-methods.json`](../configs/reproduction/paper-methods.json).

The independent scoring localizer combines Mask R-CNN and MobileSAM. Subject
isolation happens before DINO encoding (`preencode_crop`); missing localization
uses `exclude`. The crop margin is 0.1, crop side 224, and temporal aggregation
uses all pairs. Construction masks are never provided to scoring. No Subject
LoRA is selected or trained.

Construction retains the 720-candidate manifest, the official class map and
v9 quarter/full and single-frame protocols under `configs/subject-repair/`.
SegFormer construction and independent scoring use different models. All
construction rejects, missing scores and the predeclared semantic review
exclusion remain reported. The single-frame family is built from the verified
quarter/full parent without introducing new media selection.

See [training and construction](training.md) for entry points and
[the final cohort report](counterfactual-reports/subject_official_extension_20260920.md)
for denominators and limitations. The broader official cohort had previously
exposed prompts and is not an entirely unseen confirmation set. Predicted masks
are not segmentation ground truth.

Superseded Subject semantic training candidates and earlier construction
recipes are preserved in the
[pre-cleanup tree](https://github.com/winbeau/vbench-faithful/tree/8939690).
