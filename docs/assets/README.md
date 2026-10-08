# README figure provenance

`fig2.pdf` is an unchanged copy of the overview figure from the accompanying paper workspace. `fig2.png` is a raster rendering of its only page for GitHub Markdown display.

- Source: `../overleaf/figures/fig2.pdf`, relative to the repository root.
- Source workspace Git commit: `adcae00796a372a4c52cb57aa2bf47db41885938`.
- Source PDF working-tree status at copy time: clean.
- Paper caption: “Overview of target–evidence misalignment and repair strategies”.
- PDF size: 4,338,764 bytes; one page, 504 × 280.75 points.
- Source and copied PDF SHA-256: `ff007a6b586921a763914c0302c02c9dd5caf5eb68466d612a8fc1375d5cf195`.
- PNG SHA-256: `5021a4b620daef69bd8ba822d14da0db7573d590140c762f61c0153c30bee09f`.
- PNG size: 1,460,171 bytes; 2100 × 1170 pixels.
- Rendering: Poppler `pdftoppm` 22.02.0, first page, 300 DPI, PNG output.

From the repository root:

```bash
cp ../overleaf/figures/fig2.pdf docs/assets/fig2.pdf
pdftoppm -f 1 -l 1 -singlefile -r 300 -png \
  docs/assets/fig2.pdf docs/assets/fig2
sha256sum ../overleaf/figures/fig2.pdf docs/assets/fig2.pdf docs/assets/fig2.png
```

The PNG preserves the supplied figure's content and layout; no labels, illustrations, or scientific content were changed. Rendering with a different Poppler version may produce a different PNG hash while retaining the same source PDF.

Source figure note: panel (a) places the palette / `pairs` output under “Object compiler” and the car / `canonical` output under “Color compiler”. Those labels appear interchanged relative to the manuscript methods and implemented interfaces. The copied figure remains unchanged; the README method table and `paper-methods.json` describe the actual routes.
