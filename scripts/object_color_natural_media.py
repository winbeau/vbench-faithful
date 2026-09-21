"""Materialize only the frozen official Object/Color cohort, preserving bytes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
import zipfile

REPO = "Vchitect/VBench_sampled_video"
REVISION = "8c89bb0218a12323d04821d11ec03c780d0dfb9c"
ARCHIVES = {"lavie": "lavie.zip", "modelscope": "modelscope.zip",
            "videocrafter": "videocrafter-09.zip"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    if "output" not in root.parts:
        raise ValueError("new media must remain under a task output directory")
    rows = [r for d in ("object_class", "color")
            for r in json.loads((root / (d + "-manifest.json")).read_text())["rows"]]
    wanted = {r["relative_video_path"] for r in rows}
    if len(wanted) != len(rows) or len({r["video_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate video identity in frozen cohort")
    sources, errors = {}, []
    started = time.time()
    for generator, name in ARCHIVES.items():
        archive = args.archives / name
        with zipfile.ZipFile(archive) as handle:
            available = set(handle.namelist())
            selected = sorted(p for p in wanted if p.startswith(generator + "/"))
            missing = set(selected) - available
            if missing:
                raise ValueError(f"official archive lacks exact-case paths: {sorted(missing)}")
            archive_sha = digest(archive)
            for relative in selected:
                content = handle.read(relative)  # ZipFile verifies the member CRC.
                destination = root / "media" / relative
                expected = hashlib.sha256(content).hexdigest()
                if destination.exists():
                    if digest(destination) != expected:
                        raise ValueError(f"existing media differs from official archive: {relative}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                sources[relative] = {"relative_path": relative, "bytes": len(content),
                    "sha256": expected, "source": "official_google_drive_archive",
                    "archive_name": name, "archive_sha256": archive_sha,
                    "member_crc32": handle.getinfo(relative).CRC, "byte_preserving": True}
        print(json.dumps({"archive": name, "materialized": len(selected)}), flush=True)
    import requests
    from huggingface_hub import get_token

    token = get_token()
    headers = {"Authorization": "Bearer " + token} if token else {}
    entries = {}
    for dimension in ("object_class", "color"):
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/cogvideo/{dimension}?recursive=false&expand=false&limit=1000"
        response = requests.get(url, headers=headers, timeout=40)
        response.raise_for_status()
        if response.links.get("next"):
            raise ValueError("unexpected pagination; do not silently truncate source inventory")
        payload = response.json()
        save(root / "sources" / ("hf-" + dimension + ".json"), payload)
        entries.update({r["path"]: r for r in payload if r["type"] == "file"})
    selected = sorted(p for p in wanted if p.startswith("cogvideo/"))
    if set(selected) - entries.keys():
        raise ValueError("official Hugging Face revision lacks required exact-case videos")

    def download(relative):
        info = entries[relative]
        expected = info["lfs"]["oid"]
        destination = root / "media" / relative
        if not (destination.is_file() and destination.stat().st_size == info["size"]
                and digest(destination) == expected):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/" + quote(relative, safe="/")
            error = None
            for attempt in range(5):
                try:
                    # requests strips Authorization on cross-host redirects.
                    with requests.get(url, headers=headers, timeout=(20, 50), stream=True) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                handle.write(chunk)
                    if temporary.stat().st_size != info["size"] or digest(temporary) != expected:
                        raise ValueError("download does not match official LFS size/SHA-256")
                    temporary.replace(destination)
                    break
                except Exception as exc:
                    # URLs may contain signed credentials; retain only error type/status.
                    error = {"type": type(exc).__name__,
                             "http_status": getattr(getattr(exc, "response", None), "status_code", None)}
                    time.sleep(min(2 ** attempt, 10))
            else:
                return {"relative_path": relative, "error": error}
        return {"relative_path": relative, "bytes": info["size"], "sha256": expected,
                "source": "official_huggingface", "repo": REPO, "revision": REVISION,
                "repo_path": relative, "git_blob_id": info["oid"], "byte_preserving": True}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, relative) for relative in selected]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if "error" in record:
                errors.append(record)
            else:
                sources[record["relative_path"]] = record
            if i % 20 == 0 or i == len(futures):
                progress = {"cogvideo_completed": i, "cogvideo_total": len(futures),
                            "failed": len(errors), "elapsed_s": time.time() - started}
                save(root / "media-progress.json", progress)
                print(json.dumps(progress), flush=True)
    ledger = {"expected_videos": len(rows), "materialized_videos": len(sources),
              "elapsed_s": time.time() - started, "errors": errors,
              "files": [sources[p] for p in sorted(sources)]}
    save(root / "media-ledger.json", ledger)
    if errors or set(sources) != wanted:
        raise ValueError("incomplete media coverage; failed inputs remain in the frozen cohort")


if __name__ == "__main__":
    main()
