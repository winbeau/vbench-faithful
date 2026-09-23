"""Pure metric functions for smoke evaluation (no torch, deterministic).

Failures are never dropped: every metric reports the denominator, the number of
unparseable outputs and the number of missing predictions, so a high score on a
tiny valid subset cannot masquerade as coverage.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .records import SCENE_LABELS, normalize_phrase


def _norm(value: str) -> str:
    return normalize_phrase(value)


def set_prf1(predicted: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    pred = {_norm(x) for x in predicted if _norm(x)}
    ref = {_norm(x) for x in gold if _norm(x)}
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "predicted": len(pred), "gold": len(ref)}


def triple_set(target: Mapping[str, Any] | None) -> set[tuple[str, str, str]]:
    if not target:
        return set()
    return {
        (_norm(row["subject"]), _norm(row["relation"]), _norm(row["object"]))
        for row in target.get("relationships", [])
    }


def string_set(target: Mapping[str, Any] | None, key: str) -> set[str]:
    if not target:
        return set()
    return {_norm(x) for x in target.get(key, [])}


def scene_confusion(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """``pairs`` are ``(gold, predicted)`` scene labels."""
    matrix: dict[str, Counter] = {label: Counter() for label in SCENE_LABELS}
    invalid = 0
    for gold, predicted in pairs:
        if predicted not in SCENE_LABELS:
            invalid += 1
            predicted = "<invalid>"
        matrix.setdefault(gold, Counter())[predicted] += 1
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in SCENE_LABELS:
        tp = matrix.get(label, Counter()).get(label, 0)
        fp = sum(matrix[g].get(label, 0) for g in matrix if g != label)
        fn = sum(count for p, count in matrix.get(label, Counter()).items() if p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return {
        "per_class": per_class,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "matrix": {gold: dict(counts) for gold, counts in matrix.items()},
        "invalid": invalid,
    }


def evaluate_task(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate parsed predictions.

    ``rows`` items: ``{"gold": target, "pred": target|None, "errors": [...]}``.
    """
    total = len(rows)
    failed = [row for row in rows if row.get("pred") is None]
    if task == "scene":
        pairs = [(str(row["gold"]), str(row["pred"])) for row in rows if row.get("pred") is not None]
        confusion = scene_confusion(pairs)
        exact = sum(1 for gold, pred in pairs if gold == pred)
        return {
            "task": task,
            "total": total,
            "scored": len(pairs),
            "unparseable": len(failed),
            "exact_match": exact / len(pairs) if pairs else 0.0,
            "macro_f1": confusion["macro_f1"],
            "per_class": confusion["per_class"],
            "matrix": confusion["matrix"],
        }
    tp = fp = fn = 0
    exact = 0
    for row in rows:
        gold = row["gold"]
        pred = row.get("pred")
        if pred is None:
            if task == "spatial":
                fn += len(triple_set(gold))
            else:
                key = "actions" if task == "action" else "entities"
                fn += len(string_set(gold, key))
            continue
        if task == "spatial":
            gold_set, pred_set = triple_set(gold), triple_set(pred)
        elif task == "action":
            gold_set, pred_set = string_set(gold, "actions"), string_set(pred, "actions")
        else:
            gold_set, pred_set = string_set(gold, "entities"), string_set(pred, "entities")
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        exact += int(gold_set == pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "task": task,
        "total": total,
        "scored": total - len(failed),
        "unparseable": len(failed),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / total if total else 0.0,
        "error_counts": dict(Counter(code for row in rows for code in row.get("errors", []))),
    }
