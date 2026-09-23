#!/usr/bin/env python3
"""Run the four-dimension system on a prompt with explicit hard routing.

Model B (parse): one frozen backbone + three separately trained LoRAs; the task
must be named by the caller, there is no automatic dimension selection and no
fallback adapter. Model A (scene): independent checkpoint taking prompt+caption.

Usage::

    uv run --no-sync --extra train python scripts/predict.py \
        --base-model /data1/wenbiao_zhao/models/Qwen3-8B \
        --adapter spatial=runs/formal/8b/spatial \
        --adapter action=runs/formal/8b/action \
        --adapter objects=runs/formal/8b/objects \
        --scene-model runs/formal/8b/scene \
        --task spatial --prompt "A cat sleeps to the left of a dog."

    # scene needs a caption as well
    ... --task scene --prompt "A beach at sunset." --caption "A cat on a sofa."
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import training as T  # noqa: E402
from vbench_prompts_compile.inference import AdapterRouter, SceneVerifier  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["spatial", "action", "objects", "scene"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--caption", default=None, help="required for the scene task")
    parser.add_argument("--base-model", required=True, help="frozen backbone (parse adapters and/or scene adapter base)")
    parser.add_argument("--adapter", action="append", default=[], help="task=path (repeatable)")
    parser.add_argument("--scene-model", default=None, help="Scene checkpoint directory (adapter or full model)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--scene-stop-at-newline", action="store_true", help="enforce the single-label contract at decode time")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument('--action-interface', choices=['repair-v2.1', 'repair-v2', 'legacy-model'], default='repair-v2.1',
                        help='Action v9 plus declared text contract, or the historical raw model output')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.task == "scene":
        if not args.caption:
            print("--caption is required for the scene task", file=sys.stderr)
            return 2
        if not args.scene_model:
            print("--scene-model is required for the scene task", file=sys.stderr)
            return 2
        scene_path = Path(args.scene_model)
        if not scene_path.is_absolute():
            scene_path = ROOT / scene_path
        if (scene_path / "adapter_config.json").exists():
            verifier = SceneVerifier(base_model_path=args.base_model, adapter_path=str(scene_path), local_files_only=not args.allow_download)
        else:
            verifier = SceneVerifier(model_path=str(scene_path), local_files_only=not args.allow_download)
        prediction = verifier.predict(args.prompt, args.caption, max_new_tokens=max(8, min(args.max_new_tokens, 32)), stop_at_newline=args.scene_stop_at_newline)
    else:
        adapters: dict[str, str] = {}
        for item in args.adapter:
            if "=" not in item:
                print(f"--adapter expects task=path, got {item!r}", file=sys.stderr)
                return 2
            task, path = item.split("=", 1)
            adapters[task] = path
        if args.task not in adapters:
            print(f"no adapter supplied for task {args.task!r}", file=sys.stderr)
            return 2
        router = AdapterRouter(base_model_path=args.base_model, adapters=adapters, local_files_only=not args.allow_download)
        record = {"task": args.task, "input": {"prompt": args.prompt}}
        prediction = router.predict(args.task, router.user_text(record), max_new_tokens=args.max_new_tokens,
                                    action_interface=args.action_interface, action_prompt=args.prompt)

    result = {
        "task": prediction.task,
        "value": prediction.value,
        "raw": prediction.raw,
        "errors": list(prediction.errors),
        "adapter": prediction.adapter,
    }
    if prediction.postprocess is not None:
        result['postprocess'] = prediction.postprocess
    print(json.dumps(result, ensure_ascii=False))
    return 0 if prediction.value is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
