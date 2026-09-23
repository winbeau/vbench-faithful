# VBench counterfactual case figure

This directory contains the reproducible two-panel case figure for the paper.

## Build

From the repository root:

~~~bash
python3 docs/figures/vbench_counterfactual_cases/make_figure.py
~~~

The script first creates the intermediate scene crop and spatial detection-box overlays, then writes:

- output/vbench_counterfactual_cases_2panel.png — 2400 × 1280 PNG at 300 dpi;
- output/vbench_counterfactual_cases_2panel.pdf — single-page PDF, 8 × 4.27 in at 300 dpi;
- output/provenance.json — source paths, hashes, commits, and numbers used in the figure.

## Evidence

- Panel (a) uses the real H100 LaVie / ocean-0 video frame. On the same cached 16 captions, the official Origin lexical score is 12/16 = 0.750 for an ocean and 0/16 = 0.000 for a sea. The corresponding 20-video dev means are 0.56250 -> 0.03125.
- Panel (b) uses the published H100 horizontal-mirror case. Relabeling A=traffic light and B=bus makes the semantic relation A left of B -> A right of B; the official Origin score is 1.000 -> 1.000. The unchanged score is the exact case-level result; it illustrates the broader deterministic finding that the locked geometry ignores direction signs.

The source gallery and scoring evidence are from /root/wenbiao_zhao/vbench-prompts-compile-git, matrix commit ea9d500, with final scoring code commit 4d2a78d. The figure is illustrative and should be captioned as a representative case, not as a replacement for the aggregate confidence intervals.
