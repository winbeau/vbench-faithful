# VBench Faithful project page

Published at <https://winbeau.github.io/vbench-faithful/> by
[GitHub Actions](.github/workflows/pages.yml) from this `gh-pages` branch.

- `index.html` and the styles/scripts at the root are the website entry point.
- `assets/videos/<case>/` contains five-second paired videos, exact prepared first frames,
  actual generation prompts and lightweight posters.
- `assets/fonts/` and `assets/icons/` contain local fonts, licenses and official icons.
- `data/` contains case descriptions, full-precision measured scores and provenance.
- `data/deployment.json` identifies every served file by SHA-256. Actions checks it before deployment.

The page displays 30 pairs across nine dimensions. Thirty LTX-2.5 originals and
three exact static controls were produced on H200; three unselected cases remain
in `archived_cases` in `data/nuisance.json`. The displayed selection has 118
measured scores and two explicit N/A results from original VBench Color.
Both Dynamic backends score the same initial two seconds of each five-second
clip. These are selected demonstrations, not a replay of the paper's frozen cohort.

The arXiv button points to the official homepage while the paper is processing.
No model weights, caches, environments or raw worker logs are included in this branch.
