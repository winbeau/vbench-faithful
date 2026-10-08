"""Numerical agreement gate over complete, identical evaluation populations."""
from __future__ import annotations

import math


def compare_scores(reference, candidate, *, rtol=.01, atol=1e-6):
    """Compare every identified video, never hiding errors behind an average.

    Inputs use the unified evaluator row contract. This validates numerical
    agreement with a reference implementation, not human/perceptual quality.
    """
    if rtol < 0 or atol < 0 or not math.isfinite(rtol + atol):
        raise ValueError("agreement tolerances must be finite and nonnegative")
    old = {r["id"]: r for r in reference}
    new = {r["id"]: r for r in candidate}
    if not old or len(old) != len(reference) or len(new) != len(candidate) or old.keys() != new.keys():
        raise ValueError("agreement requires identical, nonempty, unique input IDs")
    results = []
    for key, a in old.items():
        b = new[key]
        if any(a[field] != b[field] for field in ("video", "video_sha256", "prompt")):
            raise ValueError(f"input identity changed: {key}")
        if any(r["status"] != "succeeded" or not isinstance(r["score"], (int, float))
               or isinstance(r["score"], bool) or not math.isfinite(r["score"]) for r in (a, b)):
            raise ValueError(f"complete finite reference and candidate scores required: {key}")
        error = abs(b["score"] - a["score"])
        threshold = max(atol, rtol * abs(a["score"]))
        results.append({"id": key, "reference": a["score"], "candidate": b["score"],
                        "absolute_error": error, "tolerance": threshold, "passed": error <= threshold})
    return {"passed": all(r["passed"] for r in results), "count": len(results),
            "rtol": rtol, "atol": atol, "rule": "abs(candidate-reference) <= max(atol, rtol*abs(reference))",
            "max_absolute_error": max(r["absolute_error"] for r in results), "rows": results}
