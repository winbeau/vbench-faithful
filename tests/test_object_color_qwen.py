from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vbench_audit_models.qwen import QwenPromptRouter
from vbench_audit_models.labels import LabelVocabulary
from scripts.object_color_semantics import validate_silver


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
