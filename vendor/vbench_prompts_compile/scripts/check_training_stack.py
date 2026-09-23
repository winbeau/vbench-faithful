"""Check imports and optional tiny CUDA operation, never download models."""

import argparse
import importlib.metadata
import json
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Allocate a tiny tensor on logical cuda:0")
    args = parser.parse_args()
    import torch
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification

    assert SFTTrainer is not None and AutoModelForSequenceClassification is not None
    assert "completion_only_loss" in SFTConfig.__dataclass_fields__
    assert LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM").r == 16
    result = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "trl", "peft", "accelerate", "datasets"
        )},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_operation": "not_requested",
        "model_training": "not_run",
    }
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        x = torch.ones((2, 2), device="cuda:0", requires_grad=True)
        y = (x @ x).sum()
        y.backward()
        torch.cuda.synchronize()
        assert y.item() == 8 and torch.all(x.grad == 4).item()
        result["cuda_operation"] = "tiny_matmul_backward_passed"
        result["device"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
