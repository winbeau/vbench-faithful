from .metric import (
    aggregate_temporal_discontinuity,
    analyze_motion_fields,
    compute_acceleration,
    compute_jerk,
    compute_velocity,
    discontinuity_to_score,
)
from .schemas import MotionField, MotionSmoothnessConfig, TimedFrame

__version__ = "0.1.0"

__all__ = [
    "MotionField", "MotionSmoothnessConfig", "TimedFrame", "analyze_motion_fields",
    "aggregate_temporal_discontinuity", "compute_acceleration", "compute_jerk", "compute_velocity",
    "discontinuity_to_score",
]
