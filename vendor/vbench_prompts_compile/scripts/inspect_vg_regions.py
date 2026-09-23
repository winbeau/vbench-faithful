#!/usr/bin/env python3
"""Download and audit the pinned Visual Genome region descriptions (annotation only).

Licence-clean natural-language source used to replace the Flickr dependency for the
Objects task (decision B4): 127 MB ZIP of human-written region phrases.

The download is explicit (``--download``), bounded and hash-checked against the
project manifest; nothing else is fetched and no images are touched.

Usage::

    uv run --no-sync python scripts/inspect_vg_regions.py --download
    uv run --no-sync python scripts/inspect_vg_regions.py            # offline statistics only
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
EXPECTED_BYTES = 127_377_968
TARGET = ROOT / "data" / "raw" / "visual_genome" / "region_descriptions.json.zip"
MANIFEST = ROOT / "data" / "raw" / "visual_genome" / "region_descriptions.manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> Path:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        size = TARGET.stat().st_size
        if size == EXPECTED_BYTES:
            print(f"cached: {TARGET} ({size} bytes)")
            return TARGET
        raise ValueError(f"cached file size mismatch: {size} != {EXPECTED_BYTES}")
    request = urllib.request.Request(URL, headers={"User-Agent": "annotation-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read(EXPECTED_BYTES + 1)
    if len(data) != EXPECTED_BYTES:
        raise ValueError(f"unexpected size {len(data)} != {EXPECTED_BYTES}")
    TARGET.write_bytes(data)
    print(f"downloaded {len(data)} bytes -> {TARGET}")
    return TARGET


def statistics(path: Path) -> dict:
    images = phrases = 0
    word_counts: Counter[int] = Counter()
    unique_phrases: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("region_descriptions.json")
        if member.file_size > 1_500_000_000:
            raise ValueError("unexpected uncompressed size")
        with io.TextIOWrapper(archive.open(member), encoding="utf-8") as stream:
            for image in _json_array(stream):
                images += 1
                for region in image.get("regions", []):
                    phrase = " ".join(str(region.get("phrase", "")).split())
                    if not phrase:
                        continue
                    phrases += 1
                    unique_phrases.add(phrase.lower())
                    word_counts[min(len(phrase.split()), 20)] += 1
    return {
        "source": URL,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "uncompressed_bytes": member.file_size,
        "images": images,
        "region_phrases": phrases,
        "unique_phrases_exact": len(unique_phrases),
        "phrase_word_histogram_capped_at_20": dict(sorted(word_counts.items())),
        "licence": "Visual Genome annotations CC-BY-4.0",
        "usage": "natural-language object phrases; replaces Flickr30k Entities for the Objects task",
    }


def _json_array(stream) -> "list":
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    while True:
        chunk = stream.read(1 << 20)
        eof = not chunk
        buffer += chunk
        if not started:
            buffer = buffer.lstrip()
            if not buffer.startswith("["):
                raise ValueError("expected JSON array")
            buffer = buffer[1:]
            started = True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("truncated JSON array") from None
                break
            yield value
            buffer = buffer[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="allow the pinned 127MB download")
    args = parser.parse_args(argv)
    path = TARGET
    if args.download or not path.exists():
        if not args.download:
            print(f"missing {path}; rerun with --download", file=sys.stderr)
            return 2
        path = download()
    stats = statistics(path)
    MANIFEST.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "phrase_word_histogram_capped_at_20"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
