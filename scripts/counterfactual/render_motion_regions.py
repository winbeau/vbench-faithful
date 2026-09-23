"""Display every automatic region; no mask labels or new inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamic_degree.trajectory import TrajectoryConfig, decode_video
from vbench_audit_models.sam_regions import unpack_masks
from .static_jitter import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "region-run", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    regions, output = Path(args.region_run), Path(args.output)
    identity = json.loads((regions / "provenance.json").read_text())
    runtime = json.loads((regions / "runtime.json").read_text())
    if (runtime["status"] != "finished" or runtime["failed"]
            or identity["manifest_sha256"] != digest(Path(args.manifest))):
        raise ValueError("completed bound region run required")
    rows = {r["candidate_id"]: r for r in map(json.loads, Path(args.manifest).read_text().splitlines())}
    output.mkdir(parents=True, exist_ok=False)
    displays = []
    for key in identity["cache_sha256"]:
        row = rows[key]
        if row["family"] == "encoding_control":
            continue
        path, video = regions / "evidence" / f"{key}.npz", Path(row["video"])
        if digest(path) != identity["cache_sha256"][key] or digest(video) != identity["input_sha256"][key]:
            raise ValueError("display cache/media identity changed")
        config = identity["config"]
        frames, t, _ = decode_video(video, TrajectoryConfig(sample_fps=config["sample_fps"], max_side=config["decode_max_side"]))
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(t, z["timestamps"]):
                raise ValueError("display timeline mismatch")
            # All first-frame proposals, in model order; no best-case selection.
            masks = unpack_masks(z['frame_0_masks_packed'], z['frame_0_image_shape'])
            count = len(masks) + 1
            fig, axes = plt.subplots((count + 5) // 6, 6, figsize=(15, 2.6 * ((count + 5) // 6)), squeeze=False)
            for i, ax in enumerate(axes.ravel()):
                ax.axis('off')
                if i == 0:
                    ax.imshow(frames[0]); ax.set_title('actual frame 0')
                elif i <= len(masks):
                    mask = masks[i - 1]
                    view = frames[0].astype(float) / 255 * np.where(mask[..., None], 1., .16)
                    ax.imshow(view); ax.set_title(f'region {i - 1} | area {mask.mean():.2%}', fontsize=9)
            fig.suptitle(f'{key}: every SAM proposal, not verified object labels', fontsize=12)
            fig.tight_layout()
            dest = output / f'{key}_regions.png'; fig.savefig(dest, dpi=110); plt.close(fig)
            displays.append({"candidate_id": key, "image": dest.name, "sha256": digest(dest)})
        print(key, flush=True)
    (output / 'provenance.json').write_text(json.dumps({"script_sha256": digest(Path(__file__)),
        "region_provenance_sha256": digest(regions / 'provenance.json'),
        "manifest_sha256": digest(Path(args.manifest)), "displays": displays, "scoring_input": False}, indent=2))


if __name__ == '__main__':
    main()
