# Paper website

The public website is published by [GitHub Actions](pages.yml)
from the `gh-pages` branch at <https://winbeau.github.io/vbench-faithful/>. It displays 30 paired case studies selected from 33 generated pairs
across the nine audited dimensions. Every displayed clip is exactly five seconds.
Thirty originals were generated with LTX-2.5 from the prepared imagegen first
frames; three additional originals repeat a still frame to isolate jitter.

## Preview and build

From this directory, run `pnpm dev` and open <http://localhost:5173/>.
The development command uses pinned Vite 8.3.4. Production needs only Python:

```bash
python3 site/build.py
# Also verify every actual video with FFprobe:
python3 site/build.py --verify-media
```

The build checks media and first-frame hashes, paired score records, missing-score
states and coverage, then writes the static artifact to `output/paper-site/`.
All paths are relative, including media, so the same artifact works under the
repository's GitHub Pages prefix. Videos load near the viewport, play in pairs,
continue playing while scrolling and respect reduced-motion preferences. Only an explicit pause or a hidden browser tab pauses an activated pair.

Edit `index.html`, the styles and the case manifests in `data/`, then rebuild.
The arXiv button currently points to the arXiv homepage while the paper is
processing. Replace it and the manuscript BibTeX when an identifier is assigned.

## Inputs and measurements

- [Target Substitution](data/semantic.json): ten pairs, 38/40 backend scores.
- [Nuisance Entanglement](data/nuisance.json): fifteen displayed pairs, 60/60
  backend scores, plus twelve archived primary scores and sixteen scores for eight secondary jitter variants.
- [Evidence Collapse](data/spatial.json): five pairs, 20/20 backend scores.
- [Run provenance](data/provenance.json): source/model revisions, weight hashes,
  H200 run timings, coverage, cache state and construction checks.

Each case directory in `assets/videos/` contains the exact prepared first frame,
actual generation prompt, both videos and lightweight posters. The manifests
include full-precision measured scores, score-input hashes, generation receipts,
construction parameters and review decisions. The short italic captions on the
page are display excerpts; the recorded scoring queries can differ from them.

All generation and evaluation ran on H200-target-server with NVIDIA H200 NVL
GPUs. Evaluation used source commit
`d7a2423038f54d946c81307b0613e81398d3091c`, original VBench commit
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, and the unchanged paper-selected
methods. The LTX upstream commit is
`9ec55f9f22798a3198d9c923856824821bc3317e`; its model revision and every weight's
SHA-256 are recorded in the provenance manifest. Final scores are fresh inference
with cross-run reuse disabled. Within-run shared evidence is recorded separately.

Raw worker logs, complete evaluation outputs, native RGB masters, construction
masks and rejected generation attempts remain in the ignored production archive
`output/showcase-video-20261008/` and its corresponding H200 directory. Published
paper result tables and selected model identities were not changed.

## Interpretation

These are selected illustrations, not an unbiased benchmark cohort or a rerun of
the published paper cohort. Scores that do not support the intended illustration
remain present: both ballet scores change with wording, the moving horse and tram
show repair drift, and the static bench does not fool original VBench. The author
selected cyclist, horse, kayak, static gallery and static tram for the displayed
Dynamic row; moving tram, skater and static bench remain in `archived_cases` in
the manifest. The frozen tram control was added after observing the bench result.

Dynamic uses native 384 × 384 frames at 8 FPS. Both backends score the **same first
16 decoded frames (0–2 seconds)** of each five-second display clip. There is no
hidden temporal resampling. Display jitter seed 2904 passed the frozen construction
checks before scoring; seed 1701 failed the geometry gate and remains recorded as
a secondary, undisplayed variant. This is not a claim of per-video invariance.

Subject and Background blur only the declared region during frames 45–74. Spatial
left/right pairs use exact horizontal reflection before encoding; above/below
pairs move upright tracked objects between shelves. Vertical swaps do not guarantee
equal center distances, and the scorer recomputes its own detections. SAM2
construction masks and assistant review are not human ground truth. Pixel identity
checks refer to the lossless construction stage, before browser H.264 compression.

Official Color omitted both suitcase inputs. They are displayed as **N/A**, with
the missing reason retained, rather than being assigned zero. The first Dynamic
repair run failed source verification; an isolated clean checkout of the same
pinned V-JEPA source enabled the recorded retry. No verification was bypassed.

Fonts include their upstream OFL licenses in `assets/fonts/`. Official icon
sources are recorded in `assets/icons/sources.json`.
