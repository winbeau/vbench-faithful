# H100 reset recovery and paper evaluation

User authorization: ensure the project survives the H100 container reset and the private
repository executes the nine paper-selected repairs and VBench evaluation. Preserve
research inputs/results; do not retrain or substitute a new candidate for paper methods.

- Preserve compiled visual runtime, exact Python distribution, upstream source and
  selected dependencies outside H100. Publish allowlisted dependency archives to the
  public HF dataset, as subsequently requested; retain project source in private GitHub.
  Only allowlisted files; no credentials, home directory or unrelated projects.
- Pin nine paper methods, preprocessing, selected adapters, Dynamic head/backbone
  and model hashes. Vendor the already published semantic source into this private repo.
- Provide one evaluation entry with explicit input metadata, input coverage, null/failure
  retention and per-dimension provenance. Repair selects the paper profile; official
  VBench remains available for all 16 dimensions. Do not silently route missing models
  to a deterministic or historical repair.
- Restore into an independent directory/environment; test actual video-to-score paths
  and compare retained fixtures with historical references, plus the full nine-row replay.
- Save restore manifests, required OS/CUDA constraints, tests, and the remaining exact
  acceptance scope. Commit/push and keep H100 checkout synchronized before handoff.
