from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VideoResult:
    video: str
    status: str
    score: float | None = None
    metric: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RunSummary:
    metric: str
    backend: str
    status: str
    total: int
    succeeded: int
    failed: int
    valid_samples: int
    aggregate: Any = None
    formula_version: str = "unavailable"
    errors: list[str] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    denominator_kind: str | None = None
    counts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
