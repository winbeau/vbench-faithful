from __future__ import annotations

from typing import Any

from .schemas import MotionSmoothnessResult


def serialize_diagnostics(result: MotionSmoothnessResult) -> dict[str, Any]:
    return {"video_path": result.video, "score": result.score, "D_video": result.discontinuity, **result.diagnostics}
