"""Integrity-checked intermediate inference shared only inside one evaluation.

The controller supplies a new run directory and a model/source/runtime namespace.
This is independent of persistent result reuse: no scores or prompts are stored.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .eval_cache import exclusive_lock, identity


class RunInferenceCache:
    def __init__(self, root, namespace):
        self.root = Path(root) / identity(namespace)
        self.hits = self.misses = 0

    def get_or_compute(self, specification, compute, *, valid):
        key = identity(specification)
        path = self.root / (key + ".json")
        with exclusive_lock(self.root / (key + ".lock")):
            try:
                record = json.loads(path.read_text())
                payload = record["payload"]
                if (record["key"] == key and record["sha256"] == identity(payload)
                        and valid(payload)):
                    self.hits += 1
                    return payload
            except (OSError, ValueError, KeyError, TypeError):
                pass
            self.misses += 1
            payload = compute()
            if valid(payload):
                record = {"key": key, "sha256": identity(payload), "payload": payload}
                temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                temp.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False))
                temp.replace(path)
            return payload
