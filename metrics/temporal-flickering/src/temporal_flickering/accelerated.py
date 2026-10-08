"""Stream every native BGR frame; no Torch, model or CUDA startup is needed."""


def score_video(path):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(path)
    previous = None
    differences = []
    try:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            current = np.asarray(frame, dtype=np.float32)
            if previous is not None:
                differences.append(np.mean(cv2.absdiff(previous, current)))
            previous = current
    finally:
        capture.release()
    if previous is None:
        return None  # Native implementation omits undecodable/empty videos.
    if not differences:
        raise ValueError("Temporal Flickering requires at least two decoded frames")
    return (255.0 - np.mean(np.array(differences)).item()) / 255.0


def compute(rows, device, submodules):
    import numpy as np
    from vbench_audit_core.video_batches import prefetch

    returned = [{"video_path": rows[i]["video"], "video_results": score}
                for i, score in prefetch(range(len(rows)), lambda i: score_video(rows[i]["video"]))
                if score is not None]
    return float(np.mean([r["video_results"] for r in returned])), returned
