# Research project page

The English project page follows the accompanying paper in `../overleaf`.
Its main figure is the unchanged Figure 2 PDF and the repository's PNG render.
The page has three synchronized LTX video comparisons, measured VBench/Ours
scores, the frozen nine-dimension paper tables, a manuscript PDF and BibTeX.

## Preview and deploy

From the repository root:

```sh
python3 scripts/site/build.py
npx --yes http-server site -p 8766 -c-1
```

Open `http://localhost:8766`. Use an HTTP server that supports byte-range
requests for reliable video seeking; opening `index.html` as a file cannot
load its JSON data. No JavaScript framework or build dependency is required.
Fonts, figures, videos and data are served locally, with no runtime CDN.

The live site is published from the standalone public repository
`winbeau/vbench-repair-page`; the research repository stays private. The
current account cannot enable Pages for a private repository. The public
repository receives only this page's source, public assets, the packaging
validator and the Pages workflow; it does not receive the research checkout.

`.github/workflows/pages.yml` packages an explicit allowlist with
`scripts/site/build.py`, then uses the official GitHub Pages artifact/deploy
actions. Pushes to `winbeau` touching the page trigger deployment; manual
dispatch is also supported. Set the public repository's Pages source to
**GitHub Actions**. The private research copy runs build validation and skips
deployment; the public page repository deploys its own `winbeau` branch.
The workflow publishes only the 17.3 MB static site, excluding
research inputs, weights, private source, README files and development tools.
It does not change repository visibility. GitHub's plan must support Pages
for the chosen repository visibility.

All relative URLs work at a repository subpath such as `/vbench-repair/`.
The GitHub code link requires access while the source repository is private.

## Demonstration provenance

All clips derive from the existing LTX-2.3 22B Dev seaside-road clip, seed 45,
HQ 15 + 3 steps, BF16, 1920 × 1088, 121 frames at 24 FPS. The source SHA-256
is `0ededeee42ed1cc7b18a9b44ef994a6d02014d6e3f62843714b134eea0efb2c3`.
This is one source video, not three independent generated videos.

- Scene changes only the query: `a street` → `a road`. This illustrative
  pair was chosen after inspecting the initial captions. The first
  `an ocean` / `a sea` control is retained in `data/demo-scores.json`.
- Subject Consistency blurs rows 0–449 with sigma 35 during frames 0–29.
  The lower 638 rows and every later frame are byte-identical in decoded RGB.
  The supplied subject annotation is `car`. Scoring independently localizes
  each input; the construction region is never passed to a scorer.
- Spatial Relationship flips every frame horizontally and retains the query
  `a car on the left of a bicycle`. Both RGB re-encoding and reflection are
  checked pixel by pixel through per-frame hashes.

`scripts/site/prepare_demo.py` reproduces the lossless RGB masters and their
construction receipt in a **new** output directory. The masters were built
and evaluated on H100 with the pinned paper interfaces. Browser MP4s were
encoded on that same host from those scored masters (width 1280, H.264,
CRF 20, YUV420P, original frame cadence, no audio, faststart). Browser encodes
are viewing copies, not evaluation inputs. The source and master file hashes
are in `data/construction.json`; website asset hashes are in
`data/asset-manifest.json`.

The fresh evaluation used code `99a7e9fb9f76e5918ea539f7dbce6957e29b9cfa`
and upstream VBench `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`. The initial
six input records ran on physical H100 devices 0–2; the two-record Scene
follow-up used physical device 0. Each process saw logical `cuda:0`. All 16
per-query/backend records succeeded. Full run files are preserved under
`/root/wenbiao_zhao/vbench-site-showcase-20261008/`, with local evidence in
`output/site-showcase-20261008/evidence/`. `scripts/site/export_scores.py`
exports scores and result-file hashes without private runtime paths.

These examples are qualitative illustrations with measured scores, not a
failure-rate estimate or a new main-table result. The page retains both the
small nonzero Subject repair response and the initial Scene control. No
training, metric changes, frozen input/result edits or E0 parity rerun was
performed for the website.

The paper tables come from `overleaf/tables/invariance.tex` and
`sensitivity.tex`, with their exact hashes recorded in `data/paper-results.json`.
They remain separate from the LTX demonstration. The historical Object Class
and Color rows use the metadata-based repairs reported in those tables;
they must not be relabeled as selected LoRA reruns. Dynamic is the frozen
450-source aligned-v1 result, with two perturbation seeds averaged per source.

## Editing

Update `index.html`, `styles.css` and `app.js` for content or interactions.
Update measured data only from bound source evidence. After replacing an
asset, update its byte count and SHA-256 in `data/asset-manifest.json`.
The packaging validator rejects missing files, broken internal links,
root-relative URLs, score ranges outside [0, 1], incomplete table sets,
changed asset hashes and non-faststart videos.

The visual direction is inspired by
[Pyramid Forcing](https://if-lab-pku.github.io/Pyramid-Forcing/); the page
implementation is original. Inter and Source Serif 4 are self-hosted Google
Fonts with their OFL licenses in `assets/fonts/`.
