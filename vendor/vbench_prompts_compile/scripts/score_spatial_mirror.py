#!/usr/bin/env python3
"""Spatial four-cell experiment, using the shared signed Repair implementation.

This entry point delegates to evaluate_spatial_repair. Supply --matrix,
--metadata, --cache, --transforms, --repair, --previous-paired and --out (a new
output directory). No upstream/GPU imports or detector reruns are required.
The historical unsigned implementation and its results remain in Git history.
"""
from evaluate_spatial_repair import main

if __name__ == '__main__':
    raise SystemExit(main())
