"""Input, identity and process contracts for the fixed paper evaluator."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ROOT = ROOT / "vendor/vbench_prompts_compile"
PAPER = ("scene", "human_action", "object_class", "subject_consistency",
         "background_consistency", "dynamic_degree", "spatial_relationship", "multiple_objects", "color")
OFFICIAL = PAPER + ("motion_smoothness", "temporal_flickering", "aesthetic_quality", "imaging_quality",
                    "temporal_style", "overall_consistency", "appearance_style")
TASKS = {"scene": "scene", "human_action": "action", "spatial_relationship": "spatial", "multiple_objects": "objects"}
ALIASES = {"dynamics_degree": "dynamic_degree", "multiplt_object": "multiple_objects", "color_consistency": "color"}


def dimension(value):
    name = ALIASES.get(value.replace("-", "_"), value.replace("-", "_"))
    if name not in OFFICIAL:
        raise ValueError(f"Unknown dimension: {value}")
    return name


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def readl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def python_paths():
    return [ROOT, SEMANTIC_ROOT / "src", SEMANTIC_ROOT / "scripts", *sorted(ROOT.glob("packages/*/src")),
            *sorted(ROOT.glob("metrics/*/src"))]


def configure_imports(assets):
    sys.path[:0] = [str(p) for p in python_paths()]
    if assets.get("mobilesam_source"):
        sys.path.insert(0, assets["mobilesam_source"])
    os.environ["VBENCH_AUDIT_UPSTREAM"] = assets["vbench"]
    os.environ["VBENCH_CACHE_DIR"] = assets["vbench_cache"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if assets.get("hf_home"):
        os.environ["HF_HOME"] = assets["hf_home"]


def load_assets(path):
    raw = json.loads(Path(path).read_text())
    result = dict(raw)
    for key, value in raw.items():
        if isinstance(value, str) and key not in {"schema", "description"}:
            p = Path(value).expanduser()
            result[key] = str((Path(path).resolve().parent / p).resolve()) if not p.is_absolute() else str(p)
    return result


def load_inputs(path, video_root=None):
    path = Path(path)
    data = readl(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
    rows = data.get("videos", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty videos list, VBench full-info list, or JSONL")
    root = Path(video_root).resolve() if video_root else path.resolve().parent
    normalized, ids = [], set()
    for row in rows:
        paths = row.get("video_list", [row.get("video")])
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
            raise ValueError("Every input needs video or a nonempty video_list; use prepare_vbench_inputs.py for the standard suite")
        dims = row.get("dimensions", row.get("dimension", list(PAPER)))
        dims = [dims] if isinstance(dims, str) else dims
        dims = [dimension(d) for d in dims]
        if len(set(dims)) != len(dims):
            raise ValueError("Duplicate input dimension")
        prompt = row.get("prompt", row.get("prompt_en", ""))
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        for relative in paths:
            video = (root / relative).resolve()
            if not video.is_file():
                raise FileNotFoundError(video)
            uid = row.get("id") if len(paths) == 1 else None
            uid = uid or hashlib.sha256(json.dumps([str(video), prompt, dims], ensure_ascii=False).encode()).hexdigest()[:24]
            if uid in ids:
                raise ValueError("Duplicate input id: " + uid)
            ids.add(uid)
            video_sha256 = digest(video)
            if row.get("video_sha256") and row["video_sha256"] != video_sha256:
                raise ValueError("Input media hash differs from its manifest: " + str(video))
            normalized.append({"id": uid, "video": str(video), "prompt": prompt, "dimensions": dims,
                               "auxiliary_info": row.get("auxiliary_info", {}),
                               "subject_en": row.get("subject_en"), "video_sha256": video_sha256})
    return normalized


def summarize(rows):
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate result id")
    for row in rows:
        if row["status"] == "succeeded":
            if row.get("score") is None or not math.isfinite(row["score"]):
                raise ValueError("Successful result requires a finite score")
        elif row.get("score") is not None:
            raise ValueError("Failed/unsupported rows must retain a null score")
    valid = [row["score"] for row in rows if row["status"] == "succeeded"]
    mean = sum(valid) / len(valid) if valid else None
    return {"input_count": len(rows), "succeeded": len(valid), "missing": len(rows) - len(valid),
            "coverage": len(valid) / len(rows) if rows else 0,
            "complete": bool(rows) and len(valid) == len(rows),
            "score": mean if len(valid) == len(rows) else None,
            "observed_subset_mean": mean, "aggregation": "equal_video"}


def result(row, score, status="succeeded", **details):
    return {"id": row["id"], "video": row["video"], "prompt": row["prompt"],
            "video_sha256": row["video_sha256"], "score": score, "status": status, **details}


def process(python, script, arguments, assets, log):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in python_paths()),
               HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VBENCH_AUDIT_UPSTREAM=assets["vbench"], VBENCH_CACHE_DIR=assets["vbench_cache"])
    if assets.get("hf_home"):
        env["HF_HOME"] = assets["hf_home"]
    with Path(log).open("w") as stream:
        run = subprocess.run([python, str(ROOT / "scripts" / script), *map(str, arguments)],
                             env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    if run.returncode:
        raise RuntimeError(f"Worker failed ({run.returncode}); see {log}")


def verify_assets(assets, dimensions, backend):
    """Bind selected weights and vendored algorithms before starting inference."""
    checked = {}

    def check(path, expected):
        path = Path(path)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Asset differs from the frozen paper selection: {path}")
        checked[str(path)] = actual

    source = json.loads((ROOT / "configs/reproduction/semantic-source.json").read_text())
    for entry in source["files"]:
        relative = Path(entry["path"]).relative_to(source["prefix"])
        check(SEMANTIC_ROOT / relative, entry["sha256"])
    if backend != "repair" and dimensions:
        official = json.loads((ROOT / "configs/reproduction/official-assets.json").read_text())
        selected = {name for dim in dimensions for name in official["requirements"][dim]}
        for entry in official["files"]:
            if entry["path"] in selected:
                check(Path(assets["vbench_cache"]) / entry["path"], entry["sha256"])
    required = set()
    vision = {"scene": {"tag2text"}, "human_action": {"umt"},
              "spatial_relationship": {"grit"}, "multiple_objects": {"grit"},
              "object_class": {"grit"}, "color": {"grit"},
              "subject_consistency": {"dino", "mobilesam", "maskrcnn"},
              "background_consistency": {"clip", "mobilesam", "maskrcnn"}}
    if backend != "origin":
        for dim in dimensions:
            required.update(vision.get(dim, set()))
    pins = json.loads((ROOT / "docs/reproduction/external-assets.json").read_text())
    for key in sorted(required):
        check(assets[key], pins[key]["sha256"])
    if required.intersection({"maskrcnn", "mobilesam"}):
        sam_sources = json.loads((ROOT / "configs/background-repair/scoring_coco80_sam_dev_v1.json").read_text())["external_source_sha256"]
        for name, expected in sam_sources.items():
            if "/MobileSAM-f706ad9/" in name:
                check(Path(assets["mobilesam_source"]) / name.split("/MobileSAM-f706ad9/", 1)[1], expected)
    release = json.loads((ROOT / "configs/reproduction/release.json").read_text())
    semantic = set(TASKS) | {"object_class", "color"}
    if backend != "origin":
        for entry in release["adapters"]:
            if entry["dimension"] in dimensions:
                check(Path(assets["adapters"]) / entry["path"], entry["sha256"])
        if semantic.intersection(dimensions):
            base = json.loads((ROOT / "configs/reproduction/qwen-base.json").read_text())
            for entry in base["files"]:
                check(Path(assets["base_model"]) / entry["path"], entry["sha256"])
        if "dynamic_degree" in dimensions:
            for entry in release["dynamic_weights"] + release["model_metadata_files"]:
                check(Path(assets["dynamic_model"]) / Path(entry["path"]).relative_to("dynamic_degree"), entry["sha256"])
    return checked
