from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def status_counts(statuses: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(statuses).items()))


def skeleton_diagnostic(**extra: Any) -> dict[str, Any]:
    """Legacy M1 fixture helper; implemented backends do not call this."""
    return {"implementation_status": "not_implemented", **extra}
