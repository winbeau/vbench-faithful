"""Batch native AMT frame pairs, retaining byte conversion and odd-frame errors."""
import os


def score_frames(motion, frames, *, batch_size=8):
    import numpy as np
    import torch
    from vbench.third_party.amt.utils.utils import img2tensor, tensor2img, check_dim_and_resize, InputPadder

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    inputs = [img2tensor(frame).to(motion.device) for frame in motion.fp.extract_frame(frames, start_from=0)]
    if len(inputs) < 2:
        raise ValueError("Motion Smoothness needs at least two even-indexed frames")
    inputs = check_dim_and_resize(inputs)
    h, w = inputs[0].shape[-2:]
    scale = motion.anchor_resolution / (h * w) * np.sqrt(
        (motion.vram_avail - motion.anchor_memory_bias) / motion.anchor_memory)
    scale = min(1, scale)
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    padder = InputPadder(inputs[0].shape, int(16 / scale))
    inputs = padder.pad(*inputs)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs) - 1, batch_size):
            count = min(batch_size, len(inputs) - 1 - start)
            left = torch.cat(inputs[start:start + count])
            right = torch.cat(inputs[start + 1:start + count + 1])
            predicted = motion.model(left, right, motion.embt.expand(count, -1, -1, -1),
                                     scale_factor=scale, eval=True)["imgt_pred"].cpu()
            predictions.extend(tensor2img(padder.unpad(frame[None])[0]) for frame in predicted)
    # Native VFI only compares reconstructed odd frames. With an even-length
    # video the last original odd frame has no right neighbour and is excluded.
    originals = motion.fp.extract_frame(frames, start_from=1)
    differences = [motion.get_diff(originals[i], frame) for i, frame in enumerate(predictions)]
    return (255.0 - np.mean(np.array(differences))) / 255.0


def compute(rows, device, submodules):
    import numpy as np
    from vbench.motion_smoothness import MotionSmoothness
    from vbench_audit_core.video_batches import prefetch

    motion = MotionSmoothness(submodules["config"], submodules["ckpt"], device)
    def decode(i):
        path = rows[i]["video"]
        if path.endswith(".mp4"):
            return motion.fp.get_frames(path)
        if os.path.isdir(path):
            return motion.fp.get_frames_from_img_folder(path)
        raise NotImplementedError("Native AMT accepts mp4 or frame directories")
    values = [score_frames(motion, frames) for _, frames in prefetch(range(len(rows)), decode)]
    return float(np.mean(values)), [{"video_path": r["video"], "video_results": v} for r, v in zip(rows, values)]
