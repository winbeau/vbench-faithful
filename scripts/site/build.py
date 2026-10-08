#!/usr/bin/env python3
"""Validate and package only the public project-page files for GitHub Pages."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths, self.anchors, self.ids = [], [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError(f"Duplicate element id: {attrs['id']}")
            self.ids.add(attrs["id"])
        for key in ("href", "src", "poster"):
            value = attrs.get(key, "")
            if value.startswith("#"):
                self.anchors.append(value[1:])
            elif value and not urlsplit(value).scheme:
                if value.startswith("/"):
                    raise ValueError(f"Root-relative URL breaks repository Pages: {value}")
                self.paths.append(unquote(urlsplit(value).path))


def validate() -> list[Path]:
    parser = Links()
    parser.feed((SITE / "index.html").read_text())
    for anchor in parser.anchors:
        assert anchor in parser.ids, f"Missing anchor: {anchor}"
    css = (SITE / "styles.css").read_text()
    paths = parser.paths + re.findall(r"url\(['\"]?([^'\")]+)", css)
    for path in paths:
        assert (SITE / path).is_file(), f"Missing page asset: {path}"
    scores = json.loads((SITE / "data/demo-scores.json").read_text())
    assert set(scores["cases"]) == {"scene", "subject", "spatial"}
    for key, case in scores["cases"].items():
        for backend in ("origin", "repair"):
            assert len(case[backend]) == 2
            assert all(v is None or (isinstance(v, (int, float)) and 0 <= v <= 1)
                       for v in case[backend]), (key, backend)
    paper = json.loads((SITE / "data/paper-results.json").read_text())
    assert len(paper["invariance"]["rows"]) == 6
    assert len(paper["sensitivity"]["rows"]) == 3
    assert all(json.loads((SITE / "data/construction.json").read_text())["checks"].values())
    assets = json.loads((SITE / "data/asset-manifest.json").read_text())
    assert {record["path"] for record in assets["files"]} == {
        path.relative_to(SITE).as_posix() for path in (SITE / "assets").rglob("*") if path.is_file()
    }, "Asset manifest must cover every published asset exactly"
    for record in assets["files"]:
        path = SITE / record["path"]
        assert path.is_file() and not path.is_symlink(), path
        assert path.stat().st_size == record["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    for name in ("coast-base", "coast-background-blur", "coast-mirror"):
        video = SITE / "assets" / f"{name}.mp4"
        content = video.read_bytes()
        assert 0 < content.find(b"moov") < content.find(b"mdat"), f"Missing faststart: {name}"
        assert (SITE / "assets" / f"{name}.webp").is_file()
    files = [SITE / name for name in ("index.html", "styles.css", "app.js", ".nojekyll")]
    files += sorted(p for folder in ("assets", "data") for p in (SITE / folder).rglob("*") if p.is_file())
    assert all(not p.is_symlink() for p in files), "Pages assets must be regular files"
    assert sum(p.stat().st_size for p in files) < 50_000_000, "Unexpected site size"
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = validate()
    if args.output:
        output = args.output.resolve()
        assert output != SITE and SITE not in output.parents, "Build outside source"
        assert not output.exists(), "Use a new build directory; existing artifacts are preserved"
        for source in files:
            target = output / source.relative_to(SITE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    print(f"Validated {len(files)} public files, {sum(p.stat().st_size for p in files) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
