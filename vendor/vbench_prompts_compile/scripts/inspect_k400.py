"""Fetch the exact VBench UMT vocabulary, never videos or weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
URL = f"https://raw.githubusercontent.com/Vchitect/VBench/{REVISION}/vbench/third_party/umt/kinetics_400_categories.txt"
EXPECTED_SHA256 = "903552632974b59d4d1beda96f7bb2829602bfeca1b3fdd5eb4bec3c5a12c1df"


def parse_labels(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        label, index = line.rsplit("\t", 1)
        rows.append({"id": int(index), "label": label})
    if sorted(row["id"] for row in rows) != list(range(400)):
        raise ValueError("Expected IDs 0..399 exactly once")
    if len({row["label"] for row in rows}) != 400:
        raise ValueError("Expected 400 distinct labels")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/raw/k400"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "kinetics_400_categories.txt"
    if target.exists():
        blob = target.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            blob = response.read(1024 * 1024 + 1)
        if len(blob) > 1024 * 1024:
            raise ValueError("Vocabulary exceeds 1 MiB cap")
        parse_labels(blob.decode())
        target.write_bytes(blob)
    if hashlib.sha256(blob).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Vocabulary SHA256 differs from inspected upstream file")
    rows = parse_labels(blob.decode())
    manifest = {
        "source_url": URL, "revision": REVISION,
        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
        "labels": len(rows), "natural_prompt_pairs": 0,
        "training_status": "vocabulary_only_not_annotated_prompts",
        "note": "Preserve upstream IDs: alphabetical ordering may not match UMT logits.",
    }
    (args.output / "labels.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
