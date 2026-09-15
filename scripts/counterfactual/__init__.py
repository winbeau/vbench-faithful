"""VBench-CF counterfactual dataset construction.

Modules:
    common      video IO, hashing and determinism helpers
    transforms  the seven deterministic counterfactual families
    select_bases  metadata-only base selection from the frozen E0 split
    build       orchestrator that materialises derived clips and the manifest
    validate    independent re-checks of the generated dataset
"""
