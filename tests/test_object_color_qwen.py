from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vbench_audit_models.qwen import QwenPromptRouter
from vbench_audit_models.labels import LabelVocabulary
from scripts.object_color_semantics import validate_silver


def test_local_router_keeps_full_hash_provenance_and_model_settings(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from threading import Barrier
    from vbench_audit_models import qwen

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 4096}))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "one": "one.safetensors", "two": "two.safetensors"
    }}))
    for name in ("one", "two"):
        (base / (name + ".safetensors")).write_bytes(name.encode())
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
    barrier = Barrier(4)
    original_hash = qwen.sha256_file

    def concurrent_hash(path):
        barrier.wait(timeout=5)
        return original_hash(path)

    monkeypatch.setattr(qwen, "sha256_file", concurrent_hash)
    calls = []
    model = SimpleNamespace(eval=lambda: None)

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: "tokenizer")))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=object))
    router = QwenPromptRouter.from_local(base, {}, device="cpu")
    assert router.provenance["base_file_sha256"] == expected
    assert router.provenance["input_fields"] == ["prompt"]
    assert calls == [(str(base), {"local_files_only": True, "torch_dtype": torch.bfloat16,
                                 "device_map": {"": "cpu"}, "attn_implementation": "sdpa"})]


def test_one_loaded_backbone_switches_one_head_and_restores_state():
    class Model:
        device = "cpu"
        active_adapter = "object_class"
        disabled = False
        events = []

        def set_adapter(self, name):
            assert isinstance(name, str)
            self.active_adapter = name
            self.events.append(("select", name))

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def generate(self, **kwargs):
            self.events.append(("generate", "base" if self.disabled else self.active_adapter))
            return torch.tensor([[1, 2, 3]])

    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kw):
            assert messages == [{"role": "user", "content": "original prompt only"}]
            assert kw["enable_thinking"] is False
            return "serialized"
        def __call__(self, text, **kw):
            assert kw["truncation"] is False
            return {"input_ids": torch.tensor([[1, 2]])}
        def decode(self, values, **kw):
            return '{"object":"car"}'

    model = Model()
    router = QwenPromptRouter(model, Tokenizer(), {"object_class": "object_class", "color": "color"}, {})
    router.generate("color", "original prompt only", mode="lora")
    assert model.active_adapter == "object_class"
    router.generate("color", "original prompt only", mode="base")
    assert [(e, v) for e, v in model.events if e == "generate"] == [("generate", "color"), ("generate", "base")]
    assert router.model is model
    with pytest.raises(ValueError, match="no adapter"):
        router.generate("human_action", "original prompt only", mode="lora")
    with pytest.raises(ValueError):
        QwenPromptRouter(model, Tokenizer(), {"object_class": "same", "color": "same"}, {})


def test_silver_labels_remain_silver_in_frozen_source_group():
    import json
    vocab = LabelVocabulary({"objects": ["car"], "colors": ["navy", "red"]})
    seed = {"dimension": "color", "prompt": "a red car", "source_group": "heldout", "split": "test"}
    raw = json.dumps({"records": [{"category": "expanded_color", "prompt": "a navy car",
                                  "output": {"object": "car", "color": "navy"}}]})
    rows = validate_silver(seed, raw, vocab, ["expanded_color"])
    assert rows[0]["target"]["color"] == "navy"
    assert rows[0]["split"] == "test" and rows[0]["source_group"] == "heldout"
    assert rows[0]["quality"] == "silver" and rows[0]["reviewed"] is False
    with pytest.raises(ValueError):
        validate_silver(seed, raw.replace('"navy"}', '"blue"}'), vocab, ["expanded_color"])


def test_batch_generation_keeps_order_decoding_contract_and_restores_route():
    class Tokenizer:
        padding_side, pad_token, eos_token, eos_token_id = "right", None, "eos", 0
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return messages[0]["content"]
        def __call__(self, texts, **kwargs):
            assert kwargs == dict(return_tensors="pt", truncation=False, padding=True)
            assert self.padding_side == "left" and self.pad_token == "eos"
            width = max(map(len, texts))
            return {"input_ids": torch.tensor([[0] * (width - len(t)) + list(t.encode()) for t in texts])}
        def decode(self, values, **kwargs):
            return bytes(v for v in values if v).decode()

    class Model:
        device, active_adapter = "cpu", "object_class"
        calls = []
        def set_adapter(self, route):
            self.active_adapter = route
        def generate(self, input_ids, **kwargs):
            assert self.active_adapter == "color"
            assert kwargs == dict(do_sample=False, max_new_tokens=96, pad_token_id=0)
            self.calls.append(len(input_ids))
            return torch.cat([input_ids, input_ids[:, -1:]], dim=1)

    model, tokenizer = Model(), Tokenizer()
    router = QwenPromptRouter(model, tokenizer, {"object_class": "object_class", "color": "color"}, {})
    assert router.generate_many("color", ["a", "longer", "tail"], mode="lora", batch_size=2) == ["a", "r", "l"]
    assert model.calls == [2, 1]
    assert model.active_adapter == "object_class" and tokenizer.padding_side == "right" and tokenizer.pad_token is None
    with pytest.raises(ValueError, match="no silent truncation"):
        router.generate_many("color", ["a" * 4001], mode="lora")
    assert model.active_adapter == "object_class" and tokenizer.padding_side == "right" and tokenizer.pad_token is None


def test_batched_compilation_preserves_invalid_raw_outputs(monkeypatch):
    vocabulary = LabelVocabulary({"objects": ["car"], "colors": ["red"]})
    router = QwenPromptRouter(None, None, {}, {})
    raws = ['{"object":"car","color":"red"}', 'not JSON', '{"object":"boat","color":"red"}']
    prompts = ["red car", "invalid", "out of vocabulary"]
    monkeypatch.setattr(router, "generate_many", lambda *a, **k: raws)
    actual = router.compile_many("color", prompts, vocabulary, mode="base")
    for i, prompt in enumerate(prompts):
        monkeypatch.setattr(router, "generate", lambda *a, **k: raws[i])
        assert actual[i] == router.compile("color", prompt, vocabulary, mode="base")
    assert [r["status"] for r in actual] == ["succeeded", "invalid_output", "invalid_output"]
