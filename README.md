# VBench Faithful project page

Published at <https://winbeau.github.io/vbench-faithful/> by
[GitHub Actions](.github/workflows/pages.yml) from this `gh-pages` branch.

## Local development

This checkout is dedicated to the `gh-pages` branch. Run `pnpm dev` here to
preview the same root files at <http://localhost:5173/>. Edit the HTML, styles,
scripts or manifests directly; no production bundler is required.

After editing any served file, run `pnpm manifest` followed by `pnpm check`,
then commit and push to `origin gh-pages`. Actions verifies the updated file
manifest before publishing. Keep environments, downloads and temporary results
under ignored `output/`; this branch contains only the site and its small tools.

## Directory layout

- `index.html` and the styles/scripts at the root are the website entry point.
- `assets/videos/<case>/` contains five-second paired videos, exact prepared first frames,
  actual generation prompts and lightweight posters.
- `assets/hero/21-kayak-hd/` contains the separately generated 1024-square, 24 FPS
  five-second cover video and its generation receipt; it has no benchmark score.
- `assets/fonts/` and `assets/icons/` contain local fonts, licenses and official icons.
- `data/` contains case descriptions, full-precision measured scores and provenance.
- `data/deployment.json` identifies every served file by SHA-256. Actions checks it before deployment.

The page displays 30 pairs across nine dimensions. Thirty-two LTX-2.5 originals and
three exact static controls were produced on H200; three unselected cases remain
in `archived_cases` in `data/nuisance.json`. The displayed selection has 112
scores linked to evaluator receipts and three retest values supplied by the
author: both suitcase Color values and the jittered kayak Dynamic value.
Five Human Action scores use subsequent author-provided updates.
Earlier results remain in each case's score history.
Both Dynamic backends score the same initial two seconds of each five-second
clip. These are selected demonstrations, not a replay of the paper's frozen cohort.

The 2026-10-09 refinement updates eight selected pairs: six counterfactual
videos with their measured scores, plus two Multiple Objects target queries.
Multiple Objects explicitly labels the two VBench targets versus all three
Ours targets. Lakeside, terrace and museum use synthetic alternating-frame
foreground blur, documented in their case metadata. Selection details and
H200 receipts are preserved in `data/refinement-20261009.json`. Human Action's
earlier evaluator scores remain in case history alongside the author's updates;
these display updates do not change the scoring implementation.

A subsequent row-timing check updated all five Subject counterfactuals to
continuous background blur from 0.5 to 5 seconds. Dog and horse use tighter
construction masks. Their VBench scores are 0.8215 and 0.8461; all five are
below 0.9, with a maximum absolute Ours change of 0.01087. Slower Background
variants were rejected because their VBench scores remained at least 0.9,
so that check retained the previous Background videos and matching scores.
Kyoto and desert were still weak examples at that stage.
`data/row-timing-20261009.json` preserves all 142 fresh
scores and the selection/rollback decisions; previous Subject scores remain
in each case's history.

The cat, macaw and woman were then refined to use per-frame subject masks
with a small guard instead of a broad temporal union. Their new VBench
counterfactual scores are 0.846956, 0.818317 and 0.768834; absolute Ours
changes are at most 0.003779. The user chose to retain the woman's stronger
blur after seeing its score. Blur strength and the 0.5–5 second interval
remain unchanged. `data/subject-boundaries-20261009.json` records all 18
fresh scores, exact reproduction of the 12 previous scores, media hashes
and construction limits. The originals and other cases are preserved.

Kyoto and desert were subsequently replaced with newly inferred LTX-2.5 cat
and SUV videos, using built-in imagegen first frames. Their counterfactuals
use continuous foreground blur from 0.25 to 5 seconds. Fresh VBench scores
are 0.974012 → 0.873090 and 0.975272 → 0.850434; Ours scores are
0.977877 → 0.969630 and 0.986131 → 0.975265. Both originals and both
counterfactuals were replaced and rescored together; their former pairs
remain in case history and Git. `data/background-replacements-20261009.json`
retains all 33 candidates and 80 fresh score rows (74 unique scores), exact
generation prompts, media hashes and selection limits. The cat's scoring
foreground is absent in 109/120 blurred-video frames, so its score stability
does not establish successful foreground detection. The SUV retains detected
foreground on all frames. These are outcome-guided illustrative selections.

The arXiv button points to the official homepage while the paper is processing.
No model weights, caches, environments or raw worker logs are included in this branch.
