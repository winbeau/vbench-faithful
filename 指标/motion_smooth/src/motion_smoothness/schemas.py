from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TimedFrame:
    frame: Any = field(repr=False, compare=False)
    timestamp_seconds: float
    source_frame_index: int


@dataclass(frozen=True)
class MotionField:
    flow: Any = field(repr=False, compare=False)
    dt: float
    source_frame_index: int
    valid_mask: Any | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class MotionSmoothnessConfig:
    eps: float = 1e-3
    min_motion_magnitude: float = 0.05
    tail_quantile: float = 0.90
    tail_weight: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(self.min_motion_magnitude) or self.min_motion_magnitude < 0:
            raise ValueError("min_motion_magnitude must be finite and non-negative")
        if not math.isfinite(self.tail_quantile) or not 0.0 <= self.tail_quantile <= 1.0:
            raise ValueError("tail_quantile must be in [0, 1]")
        if not math.isfinite(self.tail_weight) or not 0.0 <= self.tail_weight <= 1.0:
            raise ValueError("tail_weight must be in [0, 1]")


@dataclass(frozen=True)
class MotionSmoothnessResult:
    video: str
    score: float
    discontinuity: float
    diagnostics: dict[str, Any]
