"""Bounded CPU prefetch and frame packing; no metric formulas or sampling."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def prefetch(items, load, *, workers=2):
    """Preserve order and retain at most ``workers`` decoded inputs ahead."""
    if workers < 1:
        raise ValueError("workers must be positive")
    source = iter(items)
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    try:
        for _ in range(workers):
            try:
                item = next(source)
            except StopIteration:
                break
            pending.append((item, pool.submit(load, item)))
        while pending:
            item, future = pending.popleft()
            value = future.result()
            try:
                following = next(source)
            except StopIteration:
                pass
            else:
                pending.append((following, pool.submit(load, following)))
            yield item, value
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def frame_batches(decoded, batch_size):
    """Pack equal-shaped frames across videos without dropping or reordering."""
    import torch
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    frames, owners, shape = [], [], None
    for owner, video in decoded:
        if not len(video):
            raise ValueError("empty video")
        for frame in video:
            current = (tuple(frame.shape), frame.dtype, frame.device)
            if frames and (len(frames) == batch_size or current != shape):
                yield owners, torch.stack(frames)
                frames, owners = [], []
            shape = current
            frames.append(frame)
            owners.append(owner)
    if frames:
        yield owners, torch.stack(frames)
