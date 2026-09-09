from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InputError

_VIDEO_RE = re.compile(r"^video_(\d{3,})\.mp4$")


@dataclass(frozen=True)
class VideoInput:
    path: Path
    index: int | None
    metadata: dict[str, Any] | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_videos(video: str | None, video_dir: str | None) -> list[Path]:
    if bool(video) == bool(video_dir):
        raise InputError("必须且只能指定 --video 或 --video-dir")
    if video:
        path = Path(video).expanduser()
        if not path.is_file():
            raise InputError(f"视频不存在或不是文件: {path}")
        return [path]

    root = Path(video_dir).expanduser()
    if not root.is_dir():
        raise InputError(f"视频目录不存在或不是目录: {root}")
    matched: list[tuple[int, Path]] = []
    invalid_mp4: list[Path] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".mp4":
            continue
        match = _VIDEO_RE.fullmatch(path.name)
        if not match:
            invalid_mp4.append(path)
            continue
        matched.append((int(match.group(1)), path))
    if invalid_mp4:
        names = ", ".join(p.name for p in invalid_mp4)
        raise InputError(f"目录中存在不符合命名约定的 mp4: {names}")
    if not matched:
        raise InputError("视频目录为空，或没有符合 video_<至少三位数字>.mp4 的文件")
    matched.sort(key=lambda item: (item[0], item[1].name))
    duplicate_indexes = [idx for (idx, _), (next_idx, _) in zip(matched, matched[1:]) if idx == next_idx]
    if duplicate_indexes:
        raise InputError(f"视频数字索引重复: {sorted(set(duplicate_indexes))}")
    return [path for _, path in matched]


def find_metadata(video: str | None, video_dir: str | None, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise InputError(f"元数据文件不存在: {path}")
        return path
    root = Path(video_dir).expanduser() if video_dir else Path(video).expanduser().parent
    candidate = root / "metadata.json"
    return candidate if candidate.is_file() else None


def load_metadata(path: Path | None, videos: list[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"无法读取 metadata.json: {exc}") from exc
    entries = data.get("videos") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise InputError("metadata.json 必须包含 videos 列表")
    by_name: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("video"), str):
            raise InputError("metadata.videos 每项必须包含字符串 video")
        name = Path(item["video"]).as_posix()
        if name in by_name:
            raise InputError(f"metadata 对视频重复映射: {name}")
        by_name[name] = item
    result: dict[str, dict[str, Any]] = {}
    for video in videos:
        key = video.name
        if key in by_name:
            result[key] = by_name[key]
        elif video.as_posix() in by_name:
            result[key] = by_name[video.as_posix()]
    return result
