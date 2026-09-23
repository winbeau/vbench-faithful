"""Deterministic, score-blind static-jitter construction and candidate ledger.

Run with ``python -m scripts.counterfactual.static_jitter --help``.
Never writes frozen input/result directories; output runs must be new.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class SourceRejected(ValueError):
    """Input-only image quality rejection, distinct from an execution failure."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_sources(manifest: Path, config: dict) -> list[dict]:
    """One hash-ranked sample per generator/prompt, never reading metric scores."""
    with manifest.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["dimension"] == "dynamics_degree"]
    selected = []
    for split in ("dev", "test"):
        prompts = sorted({r["prompt_id"] for r in rows if r["split"] == split},
                         key=lambda x: rank(x, config["selection_seed"]))[:config[f"{split}_prompts"]]
        for prompt in prompts:
            for generator in sorted({r["generator"] for r in rows}):
                pool = [r for r in rows if r["split"] == split and r["prompt_id"] == prompt and r["generator"] == generator]
                if not pool:
                    raise ValueError(f"missing generator {generator}/{prompt}")
                row = min(pool, key=lambda r: rank(r["video_uid"], config["selection_seed"]))
                selected.append({**row, "base_id": row["video_uid"],
                                 "historical_exposure": "VBench natural E0 inputs; new interventions, not new natural videos"})
    if {r["prompt_id"] for r in selected if r["split"] == "dev"} & {r["prompt_id"] for r in selected if r["split"] == "test"}:
        raise ValueError("source prompt leakage")
    return selected


def source_path(root: Path, row: dict) -> Path:
    relative = Path(row["relative_video_path"])
    if relative.parts[0] == "videocraft":
        relative = Path("videocrafter", *relative.parts[1:])
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def source_frame(path: Path, config: dict) -> tuple[np.ndarray, dict]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = max(0, frame_count // 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise ValueError("source midpoint does not decode")
    # Centre crop to the target aspect ratio; no stretching of the actual scene.
    h, w = bgr.shape[:2]
    ratio = config["width"] / config["height"]
    cw, ch = min(w, int(h * ratio)), min(h, int(w / ratio))
    image = bgr[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    image = cv2.cvtColor(cv2.resize(image, (config["width"], config["height"]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    std = float(image.std())
    clipped = float(np.mean((image <= 1) | (image >= 254)))
    if std < config["min_image_std"] or clipped > config["max_source_clipped_fraction"]:
        raise SourceRejected(f"source image quality rejected: std={std:.4f}, clipped={clipped:.4f}")
    # Predeclared, image-independent exposure headroom, applied to every base and
    # intervention. Prevents saturation from becoming the intended disturbance.
    headroom = float(config.get("headroom", 0))
    if not 0 <= headroom < 127:
        raise ValueError("headroom must be in [0, 127)")
    image = np.rint(headroom + image.astype(float) * (255 - 2 * headroom) / 255).astype(np.uint8)
    return image, {"source_frame_index": index, "source_frame_count": frame_count,
                   "source_image_std": std, "source_clipped_fraction": clipped,
                   "base_headroom": headroom,
                   "source_sha256": digest(path),
                   "source_image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


def motion_ladder(image: np.ndarray, speed: float, config: dict, *, oscillating=False) -> np.ndarray:
    """Translate a real-image textured card over an unchanged background.

    Constant object area, no boundary crossing; every speed uses the same image,
    background, starting point, and alpha mask. This is a controlled translation,
    not a claim of natural articulated-object motion.
    """
    h, w = image.shape[:2]
    n = config["frames"]
    size = min(h, w) // 2
    texture = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(image, (0, 0), 5).astype(np.float32)
    alpha = np.ones((size, size), np.float32)
    alpha[[0, -1], :] = 0
    alpha[:, [0, -1]] = 0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)[..., None]
    # Same starting location at every level, with sufficient room for fastest path.
    start_x = (w - size) // 2 - (n - 1) * max(config["motion_pixels_per_frame"]) / 2
    y = (h - size) // 2
    output = []
    for t in range(n):
        phase = min(t % 8, 8 - t % 8) if oscillating else t
        x = int(round(start_x + speed * phase))
        if x < 2 or x + size > w - 2:
            raise ValueError("motion path crosses image boundary")
        frame = background.copy()
        frame[y:y + size, x:x + size] = (1 - alpha) * frame[y:y + size, x:x + size] + alpha * texture
        output.append(np.rint(frame).clip(0, 255).astype(np.uint8))
    return np.stack(output)


def perturb(frames: np.ndarray, family: str, amplitude: float, seed: int) -> tuple[np.ndarray, dict]:
    """Pure appearance-only operation: no resampling or coordinate displacement."""
    rng = np.random.default_rng(seed)
    n, h, w, _ = frames.shape
    if family == "clean":
        noise = np.zeros((n, 1, 1, 1), np.float32)
    elif family == "pixel_rgb":
        noise = rng.uniform(-amplitude, amplitude, frames.shape).astype(np.float32)
    elif family == "correlated_rgb":
        small = rng.uniform(-1, 1, (n, 12, 16, 3)).astype(np.float32)
        noise = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC) for x in small])
        noise = np.clip(noise, -1, 1) * amplitude
    elif family in {"global_alternating", "global_random", "local_alternating", "local_random"}:
        values = ((-1.0) ** (np.arange(n) + seed % 2) if family.endswith("alternating")
                  else rng.uniform(-1, 1, n))
        mask = np.ones((h, w), np.float32)
        if family.startswith("local"):
            mask[:] = 0
            mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
            mask = cv2.GaussianBlur(mask, (0, 0), 2)
        noise = values[:, None, None, None] * mask[None, :, :, None] * amplitude
    else:
        raise ValueError(f"unknown family {family}")
    raw = frames.astype(np.float32) + noise
    result = np.rint(raw).clip(0, 255).astype(np.uint8)
    return result, {"preencode_difference": float(np.mean(abs(result.astype(float) - frames))),
                    "clipped_fraction": float(np.mean((raw < 0) | (raw > 255))),
                    "preencode_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
                    "coordinate_map": "identity"}


def encode(path: Path, frames: np.ndarray, config: dict, *, ffmpeg="ffmpeg") -> None:
    if path.exists():
        raise FileExistsError(path)
    n, h, w, c = frames.shape
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(config["fps"]), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-threads", "1", "-preset", "fast", "-crf", str(config["crf"]),
               "-pix_fmt", "yuv420p", "-frames:v", str(n), str(path)]
    try:
        subprocess.run(command, input=frames.tobytes(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg exit {exc.returncode}: {exc.stderr.decode(errors='replace')[-2000:]}") from exc


def decode(path: Path) -> tuple[np.ndarray, list[float], float]:
    capture = cv2.VideoCapture(str(path))
    frames, pts = [], []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pts.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
    finally:
        capture.release()
    if not frames:
        raise ValueError("encoded video has no decodable frames")
    return np.stack(frames), pts, fps


def validate(decoded: np.ndarray, clean: np.ndarray, pts: list, fps: float,
             config: dict, family: str, preencode: dict) -> dict:
    expected_pts = np.arange(config["frames"]) / config["fps"]
    shape_ok = decoded.shape == clean.shape
    time_ok = len(pts) == len(expected_pts) and np.allclose(pts, expected_pts, atol=0.001) and abs(fps - config["fps"]) < 1e-5
    if not shape_ok:
        return {"qualified": False, "reason": "shape_mismatch"}
    difference = float(np.mean(abs(decoded.astype(float) - clean.astype(float))))
    correlations = []
    for before, after in zip(clean, decoded):
        a = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), (0, 0), 3).astype(float).ravel()
        b = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_RGB2GRAY), (0, 0), 3).astype(float).ravel()
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-6 else 0)
    structure = min(correlations)
    reasons = []
    if not time_ok:
        reasons.append("timeline_mismatch")
    if structure < config["min_structure_correlation"]:
        reasons.append("structure_damaged")
    if preencode["clipped_fraction"] > config["max_clipped_fraction"]:
        reasons.append("excessive_clipping")
    if family != "clean" and difference < config["min_decoded_difference"]:
        reasons.append("perturbation_not_retained")
    return {"qualified": not reasons, "reason": ";".join(reasons) or None,
            "decoded_difference": difference, "min_structure_correlation": structure,
            "decoded_shape": list(decoded.shape), "pts": pts, "fps": fps,
            "mean_temporal_pixel_difference": float(np.mean(abs(np.diff(decoded.astype(float), axis=0))))}


def variants(config: dict, *, controls="all"):
    interventions = [("clean", 0, 0)] + [(f, a, s) for f in config["families"] for a in config["amplitudes"] for s in config["noise_seeds"]]
    for family, amplitude, seed in interventions:
        yield {"kind": "static", "motion": 0, "family": family, "amplitude": amplitude, "seed": seed}
    for speed in config["motion_pixels_per_frame"]:
        for family, amplitude, seed in interventions:
            if controls == "strong" and amplitude not in {0, max(config["amplitudes"])}:
                continue
            yield {"kind": "translation", "motion": speed, "family": family, "amplitude": amplitude, "seed": seed}
    for speed in config["motion_pixels_per_frame"][1:]:
        for family, amplitude, seed in [("clean", 0, 0), ("pixel_rgb", max(config["amplitudes"]), config["noise_seeds"][0])]:
            yield {"kind": "oscillation", "motion": speed, "family": family, "amplitude": amplitude, "seed": seed}


def build(args) -> None:
    cv2.setNumThreads(2)
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg}; no construction started")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    source_manifest = Path(args.sources).resolve()
    sources = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    sources = [s for s in sources if s["split"] == args.split]
    sources = sources[args.source_start:]
    if args.limit:
        sources = sources[:args.limit]
    root = Path(args.output).resolve()
    if root == ROOT or any(root.is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("output must not target the workspace root or frozen trees")
    if root.exists():
        raise FileExistsError("choose a new run output; existing candidates must be preserved")
    root.mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "previews").mkdir()
    (root / "construction.json").write_text(json.dumps({"config": config, "config_sha256": digest(config_path),
        "sources_sha256": digest(source_manifest), "split": args.split, "limit": args.limit,
        "source_start": args.source_start,
        "controls": args.controls, "source_count": len(sources), "prepared_root": args.prepared_root,
        "ffmpeg_executable": ffmpeg, "ffmpeg_sha256": digest(Path(ffmpeg)),
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "numpy_version": np.__version__, "opencv_version": cv2.__version__,
        "builder_sha256": digest(Path(__file__))}, indent=2))
    variant_list = list(variants(config, controls=args.controls))
    prepared = ({r["base_id"]: r for r in map(json.loads, (Path(args.prepared_root) / "sources.jsonl").read_text().splitlines())}
                if args.prepared_root else None)
    with (root / "candidates.jsonl").open("x") as ledger:
        ordinal = args.source_start * len(variant_list)
        totals = Counter()
        for source in sources:
            metadata = {}
            source_rejected = False
            try:
                if prepared is None:
                    path = source_path(Path(args.data_root), source)
                    image, metadata = source_frame(path, config)
                else:
                    metadata = prepared[source["base_id"]]
                    if metadata.get("preparation_error"):
                        raise ValueError(metadata["preparation_error"])
                    bgr = cv2.imread(str(Path(args.prepared_root) / "images" / f"{source['base_id']}.png"))
                    if bgr is None:
                        raise ValueError("prepared source image does not decode")
                    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if image.shape != (config["height"], config["width"], 3):
                        raise ValueError("prepared source has different image geometry")
                    if metadata["source_image_sha256"] != hashlib.sha256(image.tobytes()).hexdigest():
                        raise ValueError("prepared source pixel hash mismatch")
                    if metadata["base_headroom"] != config.get("headroom", 0):
                        raise ValueError("prepared source has different exposure headroom")
                cv2.imwrite(str(root / "previews" / f"{source['base_id']}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                source_error = None
            except Exception as exc:
                source_error = f"{type(exc).__name__}: {exc}"
                source_rejected = isinstance(exc, SourceRejected)
            decoded_clean = {}
            counts = Counter()
            for spec in variant_list:
                name = f"video_{ordinal:06d}.mp4"
                ordinal += 1
                record = {**source, **metadata, **spec, "video": str(root / "videos" / name), "candidate_id": name[:-4]}
                try:
                    if source_error:
                        raise ValueError(source_error)
                    key = (spec["kind"], spec["motion"])
                    base = (np.repeat(image[None], config["frames"], axis=0) if spec["kind"] == "static"
                            else motion_ladder(image, spec["motion"], config, oscillating=spec["kind"] == "oscillation"))
                    frames, pre = perturb(base, spec["family"], spec["amplitude"], spec["seed"])
                    record.update(pre)
                    dest = Path(record["video"])
                    encode(dest, frames, config, ffmpeg=ffmpeg)
                    decoded, pts, fps = decode(dest)
                    if spec["family"] == "clean":
                        decoded_clean[key] = decoded
                    record.update(validate(decoded, decoded_clean[key], pts, fps, config, spec["family"], pre))
                    record["sha256"] = digest(dest)
                    record["status"] = "qualified" if record["qualified"] else "rejected"
                except Exception as exc:
                    record.update(status="rejected" if source_rejected else "construction_failed",
                                  qualified=False, reason=f"{type(exc).__name__}: {exc}")
                counts[record["status"]] += 1
                totals[record["status"]] += 1
                ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
                ledger.flush()
            print(json.dumps({"base_id": source["base_id"], "candidates_written": ordinal,
                              "source_error": source_error, "counts": counts}), flush=True)
    (root / "completion.json").write_text(json.dumps({"status": "finished", "counts": totals,
        "candidate_count": sum(totals.values()), "manifest_sha256": digest(root / "candidates.jsonl")}, indent=2))
    if totals["construction_failed"]:
        raise SystemExit(1)


def merge_builds(manifests: list[Path], output: Path) -> None:
    """Merge disjoint construction shards without moving/re-encoding any videos."""
    rows, ids, sources = [], set(), set()
    config = None
    shards = []
    for manifest in manifests:
        metadata = json.loads(manifest.with_name("construction.json").read_text())
        signature = (metadata["config_sha256"], metadata["sources_sha256"], metadata["split"], metadata["controls"])
        if config is not None and signature != config:
            raise ValueError("construction shard protocols differ")
        config = signature
        local = [json.loads(line) for line in manifest.read_text().splitlines()]
        local_sources = {r["base_id"] for r in local}
        if sources & local_sources:
            raise ValueError("source duplicated across construction shards")
        sources.update(local_sources)
        for row in local:
            if row["candidate_id"] in ids:
                raise ValueError("candidate duplicated across construction shards")
            ids.add(row["candidate_id"])
            if row["status"] == "qualified" and digest(Path(row["video"])) != row["sha256"]:
                raise ValueError("qualified input changed after construction")
        if len(local_sources) != metadata["source_count"]:
            raise ValueError("construction shard is incomplete")
        per_source = len(list(variants(metadata["config"], controls=metadata["controls"])))
        if len(local) != len(local_sources) * per_source:
            raise ValueError("construction candidate ledger is incomplete")
        rows.extend(local)
        shards.append({"manifest": str(manifest), "sha256": digest(manifest), "sources": len(local_sources)})
    if not rows:
        raise ValueError("empty construction merge")
    output = output.resolve()
    if output == ROOT or any(output.is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("cannot merge into a frozen tree")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "candidates.jsonl").open("x") as handle:
        for row in sorted(rows, key=lambda r: r["candidate_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "construction.json").write_text(json.dumps({"config": metadata["config"],
        "config_sha256": config[0], "sources_sha256": config[1], "split": config[2],
        "controls": config[3], "source_count": len(sources), "shards": shards}, indent=2))
    print(json.dumps({"source_count": len(sources), "candidate_count": len(rows),
                      "manifest_sha256": digest(output / "candidates.jsonl")}))


def export_prepared(manifests: list[Path], output: Path) -> None:
    """Small lossless source-image export, not a score-dependent candidate filter."""
    if output.resolve() == ROOT or any(output.resolve().is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("cannot export into frozen trees")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    fields = ("base_id", "source_frame_index", "source_frame_count", "source_image_std",
              "source_clipped_fraction", "source_sha256", "source_image_sha256", "base_headroom")
    written = set()
    with (output / "sources.jsonl").open("x") as handle:
        for manifest in manifests:
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                if row["base_id"] in written:
                    continue
                written.add(row["base_id"])
                image = manifest.parent / "previews" / f"{row['base_id']}.png"
                item = {k: row[k] for k in fields if k in row}
                item["construction_manifest_sha256"] = digest(manifest)
                if image.is_file():
                    shutil.copyfile(image, output / "images" / image.name)
                else:
                    item["preparation_error"] = row.get("reason") or "source image missing"
                handle.write(json.dumps(item) + "\n")
    print(json.dumps({"prepared_sources": len(written), "index_sha256": digest(output / "sources.jsonl")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--manifest", default=str(ROOT / "data/processed/e0_scoring_manifest.csv"))
    select.add_argument("--config", required=True)
    select.add_argument("--output", required=True)
    construct = sub.add_parser("build")
    construct.add_argument("--sources", required=True)
    construct.add_argument("--config", required=True)
    construct.add_argument("--data-root", required=True)
    construct.add_argument("--prepared-root", help="optional lossless prepared sources; overrides source video decoding")
    construct.add_argument("--ffmpeg", default="ffmpeg", help="existing local ffmpeg executable (H.264/libx264 required)")
    construct.add_argument("--output", required=True)
    construct.add_argument("--split", choices=["dev", "test"], required=True)
    construct.add_argument("--limit", type=int)
    construct.add_argument("--source-start", type=int, default=0, help="source offset for disjoint construction shards")
    construct.add_argument("--controls", choices=["all", "strong"], default="all")
    merge = sub.add_parser("merge")
    merge.add_argument("--manifests", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    prepared = sub.add_parser("prepare-export")
    prepared.add_argument("--manifests", nargs="+", required=True)
    prepared.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "build" and (args.source_start < 0 or (args.limit is not None and args.limit <= 0)):
        parser.error("source-start must be nonnegative and limit must be positive")
    if args.command == "select":
        rows = select_sources(Path(args.manifest), json.loads(Path(args.config).read_text()))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"sources": len(rows), "sha256": digest(path)}))
    elif args.command == "merge":
        merge_builds([Path(p) for p in args.manifests], Path(args.output))
    elif args.command == "prepare-export":
        export_prepared([Path(p) for p in args.manifests], Path(args.output))
    else:
        build(args)


if __name__ == "__main__":
    main()
