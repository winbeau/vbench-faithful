from __future__ import annotations

import importlib.util
import platform
import sys


def main() -> int:
    print(f"python={sys.executable}")
    print(f"python_version={platform.python_version()}")
    for name in ("torch", "torchvision", "transformers", "decord", "cv2", "numpy"):
        spec = importlib.util.find_spec(name)
        print(f"{name}={'installed' if spec else 'missing'}")
    try:
        import torch
        print(f"torch_version={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
    except ImportError:
        print("torch_version=missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
