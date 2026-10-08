"""Locate restored construction inputs without putting data in package sources."""
from pathlib import Path
import os

from vbench_prompts_compile import sources

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get(
    "VBENCH_TRAINING_DATA", ROOT / "output/reproduction/model-code/data")).resolve()
RAW_ROOT = Path(os.environ.get("VBENCH_TRAINING_RAW", DATA_ROOT / "raw")).resolve()
FIXTURES_ROOT = Path(os.environ.get(
    "VBENCH_TRAINING_FIXTURES", ROOT / "output/reproduction/model-code/tests/fixtures")).resolve()
TRAINING_LOCK = Path(os.environ.get(
    "VBENCH_TRAINING_LOCK", ROOT / "output/reproduction/model-code/uv.lock")).resolve()

# The published loaders intentionally keep their parsing/selection implementation.
# Only construction processes importing this module relocate their raw input root.
sources.RAW_ROOT = RAW_ROOT
