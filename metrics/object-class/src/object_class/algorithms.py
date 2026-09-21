"""Object-presence predicates on retained GRiT evidence, with full denominator."""
from __future__ import annotations

from collections import Counter

from vbench_audit_models.labels import LabelVocabulary


def score_frames(frames: list[dict], target: str | None, vocabulary: LabelVocabulary,
                 *, lexical: bool = True) -> dict:
    canonical = vocabulary.object(target)
    if canonical is None:
        return {"status": "unsupported", "score": None, "reason": "null_or_outside_frozen_text_vocabulary",
                "frame_count": len(frames), "success_frame_count": None, "frames": [],
                "support_scope": "text vocabulary, not detector capability"}
    evidence = []
    for frame in frames:
        objects = frame.get("objects", [])
        raw = [item["text"] for item in objects]
        normalized = [vocabulary.object(name) for name in raw]
        exact = target in raw
        hit = canonical in normalized if lexical else exact
        if frame.get("status") != "succeeded":
            reason, hit = "runtime_failure", False
        elif exact:
            reason = "exact_hit"
        elif hit:
            reason = "exact_miss_alias_hit"
        elif not objects:
            reason = "no_detection"
        else:
            reason = "other_class_detected"
        evidence.append({"frame_index": frame.get("frame_index", len(evidence)),
                         "support": int(hit), "exact_hit": exact, "reason": reason,
                         "instances": objects, "normalized_labels": normalized})
    failures = sum(row["reason"] == "runtime_failure" for row in evidence)
    support = sum(row["support"] for row in evidence)
    # Failed inference is not an absent object. Retain every frame, provide a
    # conservative diagnostic lower bound, but do not label it a valid score.
    good = bool(frames) and failures == 0
    return {"status": "succeeded" if good else "failed", "score": support/len(frames) if good else None,
            "frame_count": len(frames), "success_frame_count": support,
            "support_lower_bound": support/len(frames) if frames else None,
            "runtime_failure_count": failures, "frames": evidence,
            "reason_counts": dict(Counter(row["reason"] for row in evidence)),
            "denominator_kind": "all_sampled_frames", "target": target, "canonical_target": canonical}


def aggregate(rows) -> float | None:
    good = [row for row in rows if row.status == "succeeded"]
    count = sum(row.metric["frame_count"] for row in good)
    return sum(row.metric["success_frame_count"] for row in good)/count if count else None
