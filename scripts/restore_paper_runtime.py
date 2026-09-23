#!/usr/bin/env python3
"""Restore pinned dependency archives into a new directory, without the old container."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from paper_common import ROOT, digest, write_json


def extract_archive(archive, destination, roots):
    """Allow only declared archive roots and links confined to the new directory."""
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tf:
        for member in tf:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or not any(
                    path == PurePosixPath(root) or PurePosixPath(root) in path.parents for root in roots):
                raise ValueError("Unexpected recovery archive member: " + member.name)
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("Special file in recovery archive: " + member.name)
            tf.extract(member, destination, filter="data")


def relocate_environment(root, name, version):
    """Relocate standalone Python and venv launchers; no old source path is used."""
    root = Path(root).resolve()
    env = root / name
    python_root = root / ("python-" + version)
    interpreter = python_root / "bin" / ("python" + version)
    if not interpreter.is_file():
        raise FileNotFoundError(interpreter)
    for executable in ("python", "python3", "python" + version):
        path = env / "bin" / executable
        if path.is_symlink() or path.exists():
            path.unlink()
        path.symlink_to(os.path.relpath(interpreter, path.parent))
    (env / "pyvenv.cfg").write_text(
        f"home = {python_root / 'bin'}\ninclude-system-site-packages = false\n"
        f"executable = {interpreter}\n")
    for script in (env / "bin").iterdir():
        if not script.is_file() or script.is_symlink() or script.stat().st_size > 1 << 20:
            continue
        raw = script.read_bytes()
        if raw.startswith(b"#!") and b"python" in raw.split(b"\n", 1)[0]:
            script.write_bytes(("#!" + str(env / "bin/python") + "\n").encode() + raw.split(b"\n", 1)[1])
    return env / "bin/python"


def fetch(repo, revision, entry, directory, *, repo_type="dataset", allow_fallback=False):
    from huggingface_hub import hf_hub_download
    target = Path(directory) / entry["path"]
    if target.is_file() and digest(target) == entry["sha256"]:
        return target
    if entry.get("parts"):
        def part(item):
            return fetch(repo, entry["parts_revision"], item, directory, repo_type=repo_type,
                         allow_fallback=allow_fallback)
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(part, entry["parts"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".assembling")
        with temp.open("wb") as output:
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 << 20)
        if temp.stat().st_size != entry["bytes"] or digest(temp) != entry["sha256"]:
            raise ValueError("Assembled archive identity differs: " + entry["path"])
        temp.replace(target)
        return target
    endpoints = ["https://hf-mirror.com"] + (["https://huggingface.co"] if allow_fallback else [])
    for endpoint in endpoints:
        try:
            path = Path(hf_hub_download(repo, entry["path"], repo_type=repo_type,
                                       revision=revision, endpoint=endpoint, local_dir=directory))
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError("Downloaded asset identity differs: " + entry["path"])
            return path
        except Exception as exc:
            if endpoint == endpoints[-1]:
                raise
            print(json.dumps({"event": "mirror_fallback", "asset": entry["path"],
                              "reason": type(exc).__name__}), flush=True)


def configure_assets(root, semantic):
    root = Path(root).resolve()
    cache = root / "vbench-cache"
    (cache / "clip_model").mkdir(exist_ok=True)
    for name, source in [("ViT-B-32.pt", root / "clip"), ("ViT-L-14.pt", root / "clip-large")]:
        p = cache / "clip_model" / name
        if not p.exists():
            p.symlink_to(os.path.relpath(source, p.parent))
    raft = cache / "raft_model/models/raft-things.pth"
    raft.parent.mkdir(parents=True, exist_ok=True)
    if not raft.exists():
        raft.symlink_to(os.path.relpath(root / "raft/raft-things.pth", raft.parent))
    result = {"schema": "paper-assets/1", "visual_python": str(root / "visual-env/bin/python"),
              "vbench": str(root / "VBench"), "vbench_cache": str(cache), "hf_home": str(root / "hf-home"),
              "adapters": str(root / "selected/adapters"), "base_model": str(root / "base/Qwen3-8B"),
              "dynamic_model": str(root / "selected/models/dynamic_degree"), "vjepa_source": str(root / "vjepa-source"),
              "mobilesam_source": str(root / "subject-repair/MobileSAM-f706ad9")}
    if semantic:
        result["semantic_python"] = str(root / "semantic-env/bin/python")
    for key, relative in {"grit": "grit_model/grit_b_densecap_objectdet.pth",
                          "tag2text": "caption_model/tag2text_swin_14m.pth",
                          "umt": "umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth",
                          "dino": "dino_model/dino_vitbase16_pretrain.pth",
                          "dino_repo": "dino_model/facebookresearch_dino_main",
                          "clip": "clip_model/ViT-B-32.pt"}.items():
        result[key] = str(cache / relative)
    for key, name in [("maskrcnn", "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"), ("mobilesam", "mobile_sam.pt")]:
        result[key] = str(root / "subject-repair" / name)
    write_json(root / "assets.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "configs/reproduction/runtime-release.json")
    p.add_argument("--output", type=Path, required=True, help="new directory on an executable Linux filesystem")
    p.add_argument("--downloads", type=Path, required=True, help="resumable download cache, separate from output")
    p.add_argument("--allow-official-fallback", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue an interrupted restore created by this script, with the same manifest and options")
    p.add_argument("--visual-only", action="store_true", help="restore visual dependencies; omit Qwen environment and 16 GB base")
    p.add_argument("--archives-only", action="store_true", help="restore dependencies only, before separately preparing selected HF models")
    args = p.parse_args()
    root = args.output.resolve()
    downloads_root = args.downloads.resolve()
    if downloads_root == root or root in downloads_root.parents:
        raise ValueError("Download cache must be outside the restoration output directory")
    state_path = root / "restore-state.json"
    identity = {"manifest_sha256": digest(args.manifest), "visual_only": args.visual_only,
                "archives_only": args.archives_only}
    state = {**identity, "extracted": []}
    if root.exists():
        if not args.resume or not state_path.is_file():
            raise FileExistsError("Use a new directory, or --resume for this script's interrupted restore: " + str(root))
        state = json.loads(state_path.read_text())
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("Cannot resume a different manifest or restoration mode")
    manifest = json.loads(args.manifest.read_text())
    selected = [a for a in manifest["archives"] if not (args.visual_only and "semantic-env" in a["roots"])]
    downloads = [(a, fetch(manifest["repo_id"], a["revision"], a, args.downloads,
                           allow_fallback=args.allow_official_fallback)) for a in selected]
    root.mkdir(parents=True, exist_ok=args.resume)
    write_json(state_path, state)
    for entry, archive in downloads:
        if entry["sha256"] in state["extracted"]:
            continue
        extract_archive(archive, root, entry["roots"])
        state["extracted"].append(entry["sha256"])
        write_json(state_path, state)
    relocate_environment(root, "visual-env", "3.10")
    if not args.visual_only:
        relocate_environment(root, "semantic-env", "3.11")
    assets = configure_assets(root, not args.visual_only)
    if not args.archives_only:
        command = [os.sys.executable, str(ROOT / "scripts/prepare_reproduction.py"), "--output", str(root / "selected"),
                   "--include-models", "--base-model", str(root / "base/Qwen3-8B")]
        if args.allow_official_fallback:
            command.append("--allow-official-fallback")
        subprocess.run(command, check=True, env=dict(os.environ, HF_HUB_CACHE=str(args.downloads.resolve() / "hub-cache")))
        if not args.visual_only:
            base = json.loads((ROOT / "configs/reproduction/qwen-base.json").read_text())
            def base_file(entry):
                return fetch(base["repo_id"], base["revision"], entry, root / "base/Qwen3-8B", repo_type="model",
                             allow_fallback=args.allow_official_fallback)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(base_file, base["files"]))
    write_json(root / "restore-receipt.json", {"archives": selected, "manifest_sha256": digest(args.manifest),
               "old_container_required": False, "assets": assets, "selected_models_restored": not args.archives_only})
    print(json.dumps({"restored": str(root), "assets": str(root / "assets.json")}), flush=True)


if __name__ == "__main__":
    main()
