"""One prompt-only Qwen backbone, named independent adapters, hard routing.

Runs in the separately pinned Qwen environment; importing this module does not
load transformers, PEFT, weights or CUDA. Existing loaded model B can be passed
to the constructor so adding routes does not create another backbone.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

from vbench_audit_core.inputs import sha256_file
from .labels import compilation_request, validate_compilation

ROUTES = frozenset({"human_action", "multiple_objects", "spatial_relationship", "object_class", "color"})
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


class QwenPromptRouter:
    def __init__(self, model, tokenizer, adapter_routes: dict[str, str], provenance: dict):
        if set(adapter_routes) - ROUTES or len(set(adapter_routes.values())) != len(adapter_routes):
            raise ValueError("each known dimension must have its own named adapter")
        self.model, self.tokenizer = model, tokenizer
        self.routes, self.provenance = dict(adapter_routes), dict(provenance)
        self._lock = threading.Lock()

    @classmethod
    def from_local(cls, base: Path, adapters: dict[str, Path], *, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = Path(base).resolve()
        if not base.is_dir() or not (base / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("existing local complete Qwen3-8B snapshot required")
        model_files = sorted(base.glob("*.safetensors"))
        index = json.loads((base / "model.safetensors.index.json").read_text())
        if {p.name for p in model_files} != set(index["weight_map"].values()):
            raise ValueError("local base shards do not match their index")
        config = json.loads((base / "config.json").read_text())
        if config.get("model_type") != "qwen3" or config.get("hidden_size") != 4096:
            raise ValueError("expected Qwen3-8B architecture")
        files = [p for p in sorted(base.iterdir())
                 if p.is_file() and p.suffix in {".json", ".safetensors"}]
        # Hash independent shards concurrently; retain full byte verification
        # and identical provenance, precision, attention and generation options.
        with ThreadPoolExecutor(max_workers=4) as pool:
            hashes = dict(zip((p.name for p in files), pool.map(sha256_file, files)))
        provenance = {"base": str(base), "declared_revision": REVISION,
                      "base_file_sha256": hashes,
                      "input_fields": ["prompt"], "adapters": {}}
        model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
                    torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        for dimension, path in adapters.items():
            if dimension not in ROUTES:
                raise ValueError("unknown hard route")
            path = Path(path).resolve()
            if not (path / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"local adapter missing: {path}")
            if not provenance["adapters"]:
                model = PeftModel.from_pretrained(model, str(path), adapter_name=dimension,
                                                  is_trainable=False, local_files_only=True)
            else:
                model.load_adapter(str(path), adapter_name=dimension, is_trainable=False, local_files_only=True)
            provenance["adapters"][dimension] = {"path": str(path),
                "sha256": sha256_file(path / "adapter_model.safetensors"),
                "config_sha256": sha256_file(path / "adapter_config.json")}
        model.eval()
        return cls(model, tokenizer, {d: d for d in adapters}, provenance)

    @contextmanager
    def _route(self, dimension, mode):
        if dimension not in ROUTES or mode not in {"base", "lora"}:
            raise ValueError("explicit known dimension and base/lora mode required")
        if mode == "lora" and dimension not in self.routes:
            raise ValueError("no adapter for requested dimension; no cross-head fallback")
        with self._lock:
            previous = getattr(self.model, "active_adapter", None)
            if isinstance(previous, (list, tuple)):
                if len(previous) != 1:
                    raise ValueError("stacked adapters are not supported")
                previous = previous[0]
            context = self.model.disable_adapter() if mode == "base" and self.routes else nullcontext()
            try:
                if mode == "lora":
                    self.model.set_adapter(self.routes[dimension])
                with context:
                    yield
            finally:
                if previous is not None and self.routes:
                    self.model.set_adapter(previous)

    def _render(self, user_text):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def generate(self, dimension: str, user_text: str, *, mode: str, max_new_tokens=96) -> str:
        import torch
        with self._route(dimension, mode):
            inputs = self.tokenizer(self._render(user_text), return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] + max_new_tokens > 4096:
                raise ValueError("prompt exceeds frozen context bound; no silent truncation")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False,
                    max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_many(self, dimension, user_texts, *, mode, max_new_tokens=96, batch_size=32):
        """Independent left-padded requests; same templates, precision and decoding."""
        import torch
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        outputs = []
        with self._route(dimension, mode):
            padding, pad_token = self.tokenizer.padding_side, self.tokenizer.pad_token
            try:
                self.tokenizer.padding_side = "left"
                if pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                for start in range(0, len(user_texts), batch_size):
                    batch = user_texts[start:start + batch_size]
                    inputs = self.tokenizer([self._render(text) for text in batch],
                                            return_tensors="pt", truncation=False, padding=True)
                    width = inputs["input_ids"].shape[1]
                    if width + max_new_tokens > 4096:
                        raise ValueError("prompt exceeds frozen context bound; no silent truncation")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        generated = self.model.generate(**inputs, do_sample=False,
                            max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
                    if len(generated) != len(batch):
                        raise ValueError("Model changed prompt coverage")
                    outputs.extend(self.tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                                   for row in generated)
            finally:
                self.tokenizer.padding_side, self.tokenizer.pad_token = padding, pad_token
        return outputs

    def compile(self, dimension: str, prompt: str, vocabulary, *, mode: str) -> dict:
        request = compilation_request(dimension, prompt, vocabulary)
        raw = self.generate(dimension, request, mode=mode)
        return self._compiled(dimension, prompt, vocabulary, raw)

    def compile_many(self, dimension, prompts, vocabulary, *, mode, batch_size=32):
        requests = [compilation_request(dimension, prompt, vocabulary) for prompt in prompts]
        raws = self.generate_many(dimension, requests, mode=mode, batch_size=batch_size)
        return [self._compiled(dimension, prompt, vocabulary, raw) for prompt, raw in zip(prompts, raws)]

    @staticmethod
    def _compiled(dimension, prompt, vocabulary, raw):
        try:
            value = validate_compilation(dimension, json.loads(raw), vocabulary)
            return {"prompt": prompt, "status": "succeeded", "output": value, "raw": raw}
        except (ValueError, TypeError) as exc:
            return {"prompt": prompt, "status": "invalid_output", "output": None,
                    "raw": raw, "error": str(exc)}
