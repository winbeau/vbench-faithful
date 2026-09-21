"""Official CLIP background consistency and independent foreground suppression."""

__version__ = "0.1.0"

from .metric import evaluate_batch

__all__ = ["__version__", "evaluate_batch"]
