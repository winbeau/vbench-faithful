#!/usr/bin/env python3
"""Frozen selected Qwen adapters plus the original paper scoring interfaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_common import ROOT, TASKS, configure_imports, digest, load_assets, result, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "assets", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    assets = load_assets(args.assets)
    configure_imports(assets)
    # Small sequential decode steps otherwise oversubscribe the host's cores.
    # This changes CPU scheduling only, not GPU precision or generation options.
    import torch
    torch.set_num_threads(3)
    rows = json.loads(args.input.read_text())
    adapter = Path(assets["adapters"]) / args.dimension
    expected = next(row for row in json.loads((ROOT / "configs/reproduction/release.json").read_text())["adapters"]
                    if row["dimension"] == args.dimension)
    if digest(adapter / "adapter_model.safetensors") != expected["sha256"]:
        raise ValueError("Adapter is not the selected published paper checkpoint")
    if args.dimension in {"object_class", "color"}:
        from vbench_audit_models.labels import LabelVocabulary
        from vbench_audit_models.qwen import QwenPromptRouter
        vocabulary = LabelVocabulary.from_file(ROOT / "configs/four_dimension/object_color_vocabulary.json")
        router = QwenPromptRouter.from_local(Path(assets["base_model"]), {args.dimension: adapter})
        records = [router.compile(args.dimension, prompt, vocabulary, mode="lora") for prompt in dict.fromkeys(row["prompt"] for row in rows)]
        write_json(args.output, {"dimension": args.dimension, "mode": "lora", "input_fields": ["prompt"],
                                "vocabulary_sha256": vocabulary.provenance["sha256"], "model": router.provenance,
                                "records": records})
        return
    from vbench_prompts_compile.inference import AdapterRouter, SceneVerifier
    from vbench_prompts_compile.experiments import text_key
    from vbench_prompts_compile.sources import load_k400
    from score_matrix import score_one, native_entity_codec
    task = TASKS[args.dimension]
    evidence_rows = json.loads(args.evidence.read_text())["rows"]
    evidence = {row["id"]: row for row in evidence_rows}
    if len(evidence) != len(evidence_rows) or evidence.keys() != {row["id"] for row in rows}:
        raise ValueError("Evidence coverage differs from input")
    vocabulary = load_k400(ROOT / "configs/reproduction/k400-labels.json")
    codec = native_entity_codec(json.loads((ROOT / "configs/reproduction/VBench_full_info.json").read_text()))
    predictions, records = {}, []
    if task == "scene":
        model = SceneVerifier(base_model_path=assets["base_model"], adapter_path=str(adapter), local_files_only=True)
        requests = {}
        for row in rows:
            for caption in evidence[row["id"]].get("frame_captions", []):
                if isinstance(caption, str):
                    requests[text_key(task, row["prompt"], caption)] = (row["prompt"], caption)
        items = list(requests.items())
        for start in range(0, len(items), 8):
            batch = items[start:start + 8]
            values = model.predict_batch([pair for _, pair in batch])
            if len(values) != len(batch):
                raise ValueError("Scene prediction count mismatch")
            for (key, pair), value in zip(batch, values):
                predictions[key] = value.value
                records.append({"request_id": key, "prompt": pair[0], "caption": pair[1], **value.as_dict()})
    else:
        model = AdapterRouter(base_model_path=assets["base_model"], adapters={task: str(adapter)}, local_files_only=True)
        # Use the committed exact K400 vocabulary, not a user-dependent cache path.
        model._action_vocabulary = vocabulary
        for prompt in dict.fromkeys(row["prompt"] for row in rows):
            key = text_key(task, prompt)
            value = model.predict(task, model.user_text({"task": task, "input": {"prompt": prompt}}),
                                  max_new_tokens=96, canonicalize_entity_output=task == "objects",
                                  action_interface="repair-v2.1", action_prompt=prompt)
            predictions[key] = value.value
            records.append({"request_id": key, "prompt": prompt, **value.as_dict()})
    results = []
    for row in rows:
        item = {"task": task, "prompt": row["prompt"], "eligible": True, "official_target": None}
        scored = score_one(item, "Repair-model", evidence[row["id"]], predictions, vocabulary, codec,
                           spatial_backend="repair-v2", action_interface="repair-v2.1", objects_backend="repair-v2")
        valid = evidence[row["id"]].get("status") == "ok" and scored["status"] == "ok" and scored.get("missing", 0) == 0
        results.append(result(row, scored["score"] if valid else None, "succeeded" if valid else "failed",
                              diagnostics=scored))
    write_json(args.output, {"rows": results, "predictions": records, "dimension": args.dimension,
                            "adapter_sha256": expected["sha256"], "fresh_semantic_inference": True,
                            "scorer_sha256": digest(ROOT / "vendor/vbench_prompts_compile/scripts/score_matrix.py")})


if __name__ == "__main__":
    main()
