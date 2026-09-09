from __future__ import annotations

import os
import re

from .errors import InputError


def parse_gpu(value: str | None) -> list[int]:
    if value is None:
        return [0]
    if value == "":
        raise InputError("--gpu 不能是空列表")
    parts = value.split(",")
    if any(not re.fullmatch(r"\d+", part) for part in parts):
        raise InputError("--gpu 只能是逗号分隔的非负整数，例如 0,2,4")
    ids = [int(part) for part in parts]
    if len(set(ids)) != len(ids):
        raise InputError("--gpu 不能包含重复编号")
    return ids


def check_cuda(gpu_ids: list[int]) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        import torch
    except ImportError as exc:
        raise InputError("未安装 PyTorch，无法验证 CUDA；第一版不自动切换 CPU") from exc
    if not torch.cuda.is_available():
        raise InputError("CUDA 不可用；第一版不自动切换 CPU")
    count = torch.cuda.device_count()
    if any(index >= count for index in gpu_ids):
        raise InputError(f"GPU 编号超出当前可见设备范围: {gpu_ids}, 可用数量={count}")
    identifiers = []
    for index in gpu_ids:
        properties = torch.cuda.get_device_properties(index)
        identifiers.append(
            {
                "visible_index": index,
                "name": properties.name,
                "uuid": str(getattr(properties, "uuid", "")) or None,
                "pci_bus_id": str(getattr(properties, "pci_bus_id", "")) or None,
            }
        )
    return {
        "cuda_visible_devices": visible,
        "device_count": count,
        "requested_gpu_ids": gpu_ids,
        "devices": [torch.cuda.get_device_name(index) for index in gpu_ids],
        "device_identifiers": identifiers,
    }


def round_robin_shards(items: list[object], gpu_ids: list[int]) -> dict[int, list[object]]:
    shards = {gpu: [] for gpu in gpu_ids}
    for index, item in enumerate(items):
        shards[gpu_ids[index % len(gpu_ids)]].append(item)
    return {gpu: values for gpu, values in shards.items() if values}
