"""Prompt-only compilation controls and validation of offline Qwen outputs.

The visual worker can consume precompiled Qwen outputs without loading another
backbone or changing its GRiT transformers environment. Sample lookup uses the
raw prompt alone, never a video identifier or detection evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

from vbench_audit_core.inputs import sha256_file
from .labels import LabelVocabulary, compile_prompt, validate_compilation


class PromptCompiler:
    def __init__(self, dimension: str, vocabulary: LabelVocabulary, config: dict):
        if dimension not in {"object_class", "color"}:
            raise ValueError("explicit supported dimension required")
        self.dimension, self.vocabulary = dimension, vocabulary
        self.kind = config.get("kind", "deterministic")
        self.provenance = {"kind": self.kind, "dimension": dimension,
                           "vocabulary": vocabulary.provenance}
        self.records = {}
        if self.kind in {"base", "lora"}:
            path = Path(config["path"])
            payload = json.loads(path.read_text())
            if payload["dimension"] != dimension or payload["mode"] != self.kind:
                raise ValueError("compiled prompt file route/mode mismatch")
            if payload["vocabulary_sha256"] != vocabulary.provenance.get("sha256"):
                raise ValueError("compiled prompt vocabulary hash mismatch")
            if payload.get("input_fields") != ["prompt"]:
                raise ValueError("compiled prompts must record prompt-only inputs")
            for item in payload["records"]:
                key = item["prompt"]
                if key in self.records:
                    raise ValueError("duplicate compiled prompt")
                self.records[key] = (validate_compilation(dimension, item["output"], vocabulary)
                                     if item.get("status", "succeeded") == "succeeded" else None)
            self.provenance.update({"path": str(path.resolve()), "sha256": sha256_file(path),
                                    "model": payload["model"]})
        elif self.kind != "deterministic":
            raise ValueError("compiler kind must be deterministic, base or lora")

    def __call__(self, prompt: str) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("raw prompt is required")
        if self.kind == "deterministic":
            return compile_prompt(self.dimension, prompt, self.vocabulary)
        if prompt not in self.records:
            raise ValueError("no prompt-only compilation for this exact raw prompt")
        if self.records[prompt] is None:
            raise ValueError("prompt compiler returned invalid output; no fallback or fabricated label")
        return dict(self.records[prompt])
