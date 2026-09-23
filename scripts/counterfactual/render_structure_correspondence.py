"""Hash-bound displays of descriptor evidence; no score, inference or labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamic_degree.trajectory import TrajectoryConfig, decode_video
from .static_jitter import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-run", required=True)
    parser.add_argument("--match-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    feature_root, match_root, output = Path(args.feature_run), Path(args.match_run), Path(args.output)
    identity = json.loads((match_root / "provenance.json").read_text())
    if json.loads((match_root / "runtime.json").read_text())["status"] != "finished":
        raise ValueError("completed match run required")
    if (digest(feature_root / "provenance.json") != identity["source_provenance_sha256"]
            or digest(Path(args.manifest)) != identity["manifest_sha256"]):
        raise ValueError("bound feature source and manifest required")
    rows = [json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
    by_id = {r["candidate_id"]: r for r in rows}
    records = [json.loads(s) for s in (match_root / "diagnostics.jsonl").read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    displays = []
    for record in records:
        if record["family"] == "encoding_control":
            continue
        key = record["candidate_id"]
        row = by_id[key]
        video = Path(row["video"])
        feature_path = feature_root / "evidence" / f"{key}.npz"
        match_path = match_root / "evidence" / f"{key}.npz"
        if (digest(video) != identity["input_sha256"][key]
                or digest(feature_path) != identity["source_cache_sha256"][key]
                or digest(match_path) != identity["cache_sha256"][key]):
            raise ValueError("display input/cache changed")
        config = identity["config"]
        frames, t, _ = decode_video(video, TrajectoryConfig(sample_fps=config["sample_fps"], max_side=config["decode_max_side"]))
        with np.load(feature_path, allow_pickle=False) as f, np.load(match_path, allow_pickle=False) as m:
            if not np.array_equal(f["timestamps"], t) or not np.array_equal(m["timestamps"], t):
                raise ValueError("display timeline mismatch")
            q, shape = f["queries"], tuple(map(int, f["grid_shape"]))
            h, w = frames.shape[1:3]
            extent = (-.5, w - .5, h - .5, -.5)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
            for r, facet in enumerate(("key9", "token11")):
                # Same PCA basis/color limits for all native-time frames of one video.
                features = f[facet].astype(np.float32)
                flat = features.reshape(-1, features.shape[-1])
                mean = flat.mean(axis=0); centered = flat - mean
                values, basis = np.linalg.eigh(centered.T @ centered)
                colors = centered @ basis[:, -3:]
                low, high = np.percentile(colors, [1, 99], axis=0)
                colors = np.clip((colors - low) / np.maximum(high - low, 1e-9), 0, 1).reshape(len(t), *shape, 3)
                ax = axes[r, 0]
                ax.imshow(frames[0]); ax.imshow(f["attention"][0].reshape(shape), extent=extent, cmap="magma", alpha=.55)
                ax.set_title(f"{facet}: CLS attention (not a mask)", fontsize=10); ax.axis("off")
                for col, lag in ((1, 1), (2, 4)):
                    ax = axes[r, col]; ax.imshow(frames[0])
                    prefix = f"{facet}_0_{lag}_"
                    mutual = m[prefix + "mutual"]
                    resolved = m[prefix + "reciprocal_subpixel_resolved"]
                    xy = m[prefix + "target_xy"]
                    # Regular spatial subset, not chosen by motion magnitude/correctness.
                    indices = np.arange(len(q)).reshape(shape)[1::3, 1::3].ravel()
                    for i in indices:
                        if mutual[i]:
                            delta = xy[i] - q[i]
                            ax.arrow(*q[i], *delta, color="#00D5EA" if resolved[i] else "#FA8F25",
                                     head_width=max(h, w) / 100, length_includes_head=True, linewidth=.6)
                        else:
                            ax.scatter(*q[i], s=5, marker="x", color="#BBBBBB")
                    ax.set(xlim=(-.5, w - .5), ylim=(h - .5, -.5))
                    ax.set_title(f"0 -> {lag}: mutual {mutual.mean():.1%}; resolved {resolved.mean():.1%}", fontsize=10)
                    ax.axis("off")
                ax = axes[r, 3]
                ax.imshow(np.concatenate((colors[0], colors[4]), axis=1), interpolation="nearest")
                ax.set_title("PCA colors: frame 0 | 4 (not labels)", fontsize=10); ax.axis("off")
            fig.suptitle(f"{key}: raw structure correspondence, NOT Repair\n"
                         "cyan=mutual+reciprocal refinement; orange=coarse mutual; gray=nonmutual", fontsize=12)
            path = output / f"{key}.png"
            fig.savefig(path, dpi=135); plt.close(fig)
        displays.append({"candidate_id": key, "image": path.name, "sha256": digest(path)})
        print(key, flush=True)
    provenance = {"script_sha256": digest(Path(__file__)), "manifest_sha256": digest(Path(args.manifest)),
                  "match_provenance_sha256": digest(match_root / "provenance.json"),
                  "feature_provenance_sha256": digest(feature_root / "provenance.json"),
                  "displays": displays, "scoring_input": False}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
