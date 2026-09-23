#!/usr/bin/env python3
"""Build a local random micro model + tokenizer for CPU smoke tests.

Nothing is downloaded: the tokenizer is trained on the project's own fixture and
contract text, and the causal LM is randomly initialised. The result validates
the training/saving/routing path only; it has no semantic ability and its
metrics must never be reported as model quality.

Usage::

    uv run --no-sync --extra train python scripts/make_tiny_model.py --output models/tiny-qwen-smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile.training import TASK_INSTRUCTIONS  # noqa: E402

SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def corpus_lines() -> list[str]:
    lines: list[str] = list(TASK_INSTRUCTIONS.values())
    lines += ['{"relationships":[{"subject":"cat","relation":"left","object":"dog"}]}',
              '{"relationships":[]}', '{"actions":["riding a bike"]}', '{"actions":[]}',
              '{"entities":["cat","dog","bicycle"]}', 'supported', 'contradicted', 'insufficient']
    fixtures = ROOT / "tests" / "fixtures"
    for path in sorted(fixtures.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" in row:
                lines.append(row["prompt"])
            if "caption" in row:
                lines.append(row["caption"])
            if "target" in row:
                lines.append(row["target"] if isinstance(row["target"], str) else R.canonical_json(row["target"]))
    small = ROOT / "data" / "smoke"
    if small.exists():
        for path in sorted(small.glob("*/*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                lines.append(row["input"]["prompt"])
                if "caption" in row["input"]:
                    lines.append(row["input"]["caption"])
                lines.append(row["target"] if isinstance(row["target"], str) else R.canonical_json(row["target"]))
    return lines


def build(output: Path, *, vocab_size: int, hidden_size: int, layers: int, heads: int, seed: int) -> dict[str, Any]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    lines = corpus_lines()
    tokenizer.train_from_iterator(lines, trainer=trainer, length=len(lines))

    output.mkdir(parents=True, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=1024,
        model_input_names=["input_ids", "attention_mask"],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(output)

    config = Qwen2Config(
        vocab_size=len(tokenizer.get_vocab()),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=max(1, heads // 2),
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(output)

    info = {
        "output": str(output),
        "vocab_size": len(tokenizer.get_vocab()),
        "corpus_lines": len(lines),
        "hidden_size": hidden_size,
        "layers": layers,
        "parameters": sum(p.numel() for p in model.parameters()),
        "note": "randomly initialised micro model for engineering smoke only; no semantic ability, no downloads",
    }
    (output / "tiny_model_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT / "models" / "tiny-qwen-smoke"))
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import random

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    info = build(Path(args.output), vocab_size=args.vocab_size, hidden_size=args.hidden_size, layers=args.layers, heads=args.heads, seed=args.seed)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
