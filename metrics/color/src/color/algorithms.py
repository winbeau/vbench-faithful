"""Instance-bound, whole-word color evidence with an explicit frame denominator."""
from __future__ import annotations

from collections import Counter
import re

from vbench_audit_models.labels import LabelVocabulary


def legacy_trace_score(frames: list[dict], prompt: str, color: str) -> dict:
    """Exact upstream tuple-consumption diagnostic, never labeled Official."""
    target = prompt.replace('a ', '').replace('an ', '').replace(color, '').strip()
    denominator = numerator = 0
    colors = ("white", "red", "pink", "blue", "silver", "purple", "orange", "green", "gray", "yellow", "black", "grey")
    for frame in frames:
        if frame.get("status") != "succeeded":
            return {"status": "failed", "score": None, "reason": "raw_model_failure"}
        descriptions = frame.get("postprocessed", frame.get("primary", []))
        labels = frame.get("legacy_labels", [x["text"] for x in frame.get("objects", [])])
        if descriptions and not labels:
            return {"status": "failed", "score": None, "reason": "upstream_would_index_empty_class_list"}
        hit, present = False, False
        for instance in descriptions:
            if labels[0] == target:
                present |= any(c in instance["text"] for c in colors)
                hit |= color in instance["text"]
        numerator += int(hit)
        denominator += int(present)
    return {"status": "succeeded" if denominator else "dropped_by_condition",
            "score": numerator/denominator if denominator else None,
            "conditional_frame_count": denominator, "success_frame_count": numerator,
            "frame_count": len(frames), "raw_object_query": target}


def color_support(caption: str, color: str, target: str, vocabulary: LabelVocabulary) -> dict:
    mentions = vocabulary.mentions(caption, colors=True)
    target_mentions = [m for m in vocabulary.mentions(caption) if m["label"] == target]
    spans, negated, unbound = [], [], []
    for item in mentions:
        if item["label"] != color:
            continue
        prefix = re.split(r"[,.!?;]|\bbut\b", caption[:item["start"]], flags=re.I)[-1]
        if re.search(r"\b(no|not|without|neither|nor)\b", prefix, re.I):
            negated.append(item)
            continue
        if target_mentions:
            # A caption can mention an unrelated red object. Binding the
            # instance class alone does not make that color its attribute.
            bound = False
            for obj in target_mentions:
                if item["end"] <= obj["start"]:
                    between = caption[item["end"]:obj["start"]]
                    bound |= len(between.split()) <= 4 and not bool(re.search(r"\b(on|in|near|with|by|and|beside|behind|under|above|of)\b|[,.!?;]", between, re.I))
                elif item["start"] >= obj["end"]:
                    between = caption[obj["end"]:item["start"]].strip().casefold()
                    bound |= between in {"is", "are", "looks", "appears"}
            if not bound:
                unbound.append(item)
                continue
        spans.append(item)
    return {"supported": bool(spans), "spans": spans, "negated_spans": negated,
            "unbound_spans": unbound,
            "reason": "supported" if spans else ("negated_color" if negated else "insufficient_color_evidence")}


def score_frames(frames: list[dict], target: str | None, color: str | None,
                 vocabulary: LabelVocabulary, *, variant: str = "repair") -> dict:
    if variant not in {"binding", "binding_lexical", "repair"}:
        raise ValueError("unknown color scoring variant")
    obj, col = vocabulary.object(target), vocabulary.color(color)
    if obj is None or col is None:
        return {"status": "unsupported", "score": None, "reason": "null_or_outside_vocabulary",
                "frame_count": len(frames), "frames": [], "success_frame_count": None,
                "support_scope": "text query vocabulary, not detector capability"}
    evidence = []
    whitelist = ("white", "red", "pink", "blue", "silver", "purple", "orange", "green", "gray", "yellow", "black", "grey")
    for frame in frames:
        hit, conditional, target_found = False, False, False
        pairs = (frame.get("binding") or {}).get("pairs", [])
        candidates = []
        ambiguous = sum(pair["method"] == "ambiguous" for pair in pairs)
        for pair in pairs:
            instance = pair["object"]
            if instance is None:
                continue
            matches = instance["text"] == target if variant == "binding" else vocabulary.object(instance["text"]) == obj
            if not matches:
                continue
            target_found = True
            caption = pair["caption"]["text"]
            if variant == "binding":
                support = {"supported": color in caption, "reason": "legacy_substring"}
                usable = any(key in caption for key in whitelist)
            else:
                support = color_support(caption, col, obj, vocabulary)
                mentioned = {m["label"] for m in vocabulary.mentions(caption, colors=True)}
                usable = any(color_support(caption, c, obj, vocabulary)["supported"] for c in mentioned)
            hit |= support["supported"]
            conditional |= usable
            candidates.append({"caption_index": pair["caption_index"], "object_index": pair["object_index"],
                               "binding_method": pair["method"], "caption": pair["caption"],
                               "object": instance, "color_evidence": support})
        if frame.get("status") != "succeeded":
            reason, hit, conditional = "runtime_failure", False, False
        elif hit:
            reason = "supported"
        elif ambiguous:
            reason = "ambiguous_binding"
        elif target_found:
            reason = "insufficient_color_evidence"
        else:
            reason = "no_target_detection"
        evidence.append({"frame_index": frame.get("frame_index", len(evidence)), "support": int(hit),
                         "conditional_denominator": int(conditional), "target_found": target_found,
                         "ambiguous_count": ambiguous, "instances": candidates, "reason": reason})
    failures = sum(item["reason"] == "runtime_failure" for item in evidence)
    successes = sum(item["support"] for item in evidence)
    conditional_count = sum(item["conditional_denominator"] for item in evidence)
    denominator = len(frames) if variant == "repair" else conditional_count
    status = "failed" if failures or not frames else ("succeeded" if denominator else "dropped_by_condition")
    return {"status": status, "score": successes/denominator if status == "succeeded" else None,
            "frame_count": len(frames), "success_frame_count": successes,
            "conditional_frame_count": conditional_count,
            "conditional_rate": successes/conditional_count if conditional_count else None,
            "support_lower_bound": successes/len(frames) if frames else None,
            "denominator_kind": "all_sampled_frames" if variant == "repair" else "target_with_color_evidence",
            "runtime_failure_count": failures, "reason_counts": dict(Counter(item["reason"] for item in evidence)),
            "frames": evidence, "target": target, "color": color, "variant": variant}


def aggregate(rows) -> float | None:
    good = [row for row in rows if row.status == "succeeded"]
    return sum(row.score for row in good)/len(good) if good else None
