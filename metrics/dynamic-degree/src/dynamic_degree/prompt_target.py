from __future__ import annotations

import re
from enum import Enum

from .schemas import PromptTargetDecision


class MotionTarget(str, Enum):
    SUBJECT = "subject"
    CAMERA = "camera"
    GENERIC = "generic"
    BOTH = "both"
    UNKNOWN = "unknown"


_CAMERA_PATTERNS = (
    r"\bcamera\s+(?:pan|pans|panning|tilt|tilts|tilting|zoom|zooms|zooming|move|moves|moving|rotate|rotates|rotating|orbit|orbits|orbiting|shake|shakes|shaking|follow|follows|following)\b",
    r"\b(?:pan|panning|tilt|tilting|zoom|zooming|dolly|tracking shot|handheld camera|camera movement)\b",
)
_SUBJECT_PATTERNS = (
    r"\b(?:person|man|woman|child|animal|dog|cat|bird|horse|sheep|cow|elephant|bear|zebra|giraffe|car|vehicle|bicycle|bike|motorcycle|airplane|bus|train|truck|boat|object|subject)\b.{0,72}\b(?:walk|walks|walking|run|runs|running|swim|swims|swimming|dance|dances|dancing|jump|jumps|jumping|fly|flies|flying|drive|drives|driving|ride|rides|riding|move|moves|moving|rotate|rotates|rotating|accelerate|accelerates|accelerating|giving|washing|eating|drinking|playing|leaning|gliding|slowing|stuck|turning|cruising|soaring|taking|landing|speeding|crossing|anchored|sailing|building|grooming|bending|galloping|chewing|resting|spraying|catching|sniffing|climbing|hunting)\b",
    r"\b(?:walking|running|swimming|dancing|jumping|flying|driving|riding|accelerating|spinning|gliding|turning|cruising|soaring|landing|speeding|sailing|galloping|climbing)\b",
)
_GENERIC_PATTERNS = (r"\b(?:dynamic|motion|movement|moving scene|high action)\b",)


def _matches(prompt: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    found = []
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            found.append(match.group(0).lower())
    return tuple(found)


def parse_motion_target(prompt: str | None, override: str | MotionTarget | None = None) -> PromptTargetDecision:
    if override is not None:
        try:
            target = override if isinstance(override, MotionTarget) else MotionTarget(str(override).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unsupported motion_target override: {override!r}") from exc
        return PromptTargetDecision(target.value, "explicit_override")

    text = (prompt or "").strip()
    if not text:
        return PromptTargetDecision(MotionTarget.UNKNOWN.value, "empty_prompt")
    camera = _matches(text, _CAMERA_PATTERNS)
    subject = _matches(text, _SUBJECT_PATTERNS)
    generic = _matches(text, _GENERIC_PATTERNS)
    if camera and subject:
        target = MotionTarget.BOTH
    elif camera:
        target = MotionTarget.CAMERA
    elif subject:
        target = MotionTarget.SUBJECT
    elif generic:
        target = MotionTarget.GENERIC
    else:
        target = MotionTarget.UNKNOWN
    return PromptTargetDecision(target.value, "deterministic_rules", subject, camera, generic)
