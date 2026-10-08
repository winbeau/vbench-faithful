"""Read-only loaders for the pinned raw annotation snapshots.

Nothing here writes to ``data/raw`` and nothing downloads. Parsing mirrors the
audited inspection scripts (same whitelists, same bounds) so the cleaning stage
cannot silently widen a source. Standard library only; SNLI additionally
requires pyarrow and reports ``unavailable`` instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
import zipfile
from typing import Any, Iterable, Iterator, Sequence

from .records import normalize_phrase, normalize_space, sha256_text, word_count

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw"

# Same conservative exact whitelist as scripts/inspect_vg_snli.py.
VG_PREDICATES: dict[str, frozenset[str]] = {
    "left": frozenset({"left of", "to the left of", "on the left of"}),
    "right": frozenset({"right of", "to the right of", "on the right of"}),
    "above": frozenset({"above"}),
    "below": frozenset({"below", "under", "underneath", "beneath"}),
}

FLICKR_MENTION_RE = re.compile(r"\[/EN#([^/\s]+)/([^\s]+) ([^\]]+)\]")
# Types that are not objects we can target with the current protocol.
FLICKR_NON_OBJECT_TYPES = frozenset({"notvisual", "scene"})
# Types that need an explicit protocol decision before entering ``entities``.
FLICKR_FLAGGED_TYPES = frozenset({"bodypart", "clothing"})

ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


@dataclass(frozen=True)
class PromptRef:
    text: str
    source: str
    source_file: str
    line: int
    sha256: str
    word_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "source_file": self.source_file,
            "line": self.line,
            "sha256": self.sha256,
            "word_count": self.word_count,
        }


@dataclass(frozen=True)
class Mention:
    phrase: str
    chain: str
    types: tuple[str, ...]
    positioned: bool


@dataclass(frozen=True)
class FlickrCaption:
    image_id: str
    index: int
    split: str
    text: str
    mentions: tuple[Mention, ...]
    raw: str


@dataclass(frozen=True)
class VGEdge:
    image_id: int
    relationship_id: Any
    predicate: str
    direction: str
    subject: str
    object: str
    subject_id: Any
    object_id: Any

    @property
    def key(self) -> str:
        return f"vg-{self.image_id}-{self.relationship_id}"


class K400Vocabulary:
    """Locked Kinetics-400 names with the upstream non-alphabetical IDs."""

    def __init__(self, entries: Sequence[Mapping[str, Any]], aliases: Mapping[str, str] | None = None) -> None:
        self.entries = tuple({"id": int(e["id"]), "label": str(e["label"])} for e in entries)
        self._by_id = {e["id"]: e["label"] for e in self.entries}
        self._by_norm = {normalize_phrase(e["label"]): e["label"] for e in self.entries}
        # Explicit, reviewed aliases only; generated variants are handled below.
        self.aliases = {normalize_phrase(k): v for k, v in (aliases or {}).items()}
        self._alias_norm = {k: self._by_norm[normalize_phrase(v)] for k, v in self.aliases.items() if normalize_phrase(v) in self._by_norm}

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(e["label"] for e in self.entries)

    def by_id(self, identifier: int) -> str | None:
        return self._by_id.get(int(identifier))

    def resolve(self, name: str) -> str | None:
        """Return the canonical K400 label, or ``None`` when unsupported."""
        key = normalize_phrase(name)
        if not key:
            return None
        if key in self._by_norm:
            return self._by_norm[key]
        if key in self._alias_norm:
            return self._alias_norm[key]
        stripped = normalize_phrase(ARTICLE_RE.sub(" ", key))
        if stripped in self._by_norm:
            return self._by_norm[stripped]
        stripped_alias = {normalize_phrase(ARTICLE_RE.sub(" ", k)): v for k, v in self._by_norm.items()}
        if stripped in stripped_alias:
            return stripped_alias[stripped]
        # Conservative containment rule: a label is accepted when all of its content
        # tokens appear in the phrase and the phrase's head token is the label's head
        # ("climbing a tall tree" -> "climbing tree"), never when the head differs.
        from .records import phrase_tokens as _tokens, stem as _stem

        phrase_stems = [_stem(token) for token in _tokens(key)]
        if len(phrase_stems) < 2:
            return None
        head = phrase_stems[-1]
        best: tuple[int, str] | None = None
        for label_key, label in self._by_norm.items():
            label_stems = [_stem(token) for token in _tokens(label_key)]
            if len(label_stems) < 2 or label_stems[-1] != head:
                continue
            if all(token in phrase_stems for token in label_stems):
                overlap = len(label_stems)
                if best is None or overlap > best[0]:
                    best = (overlap, label)
        return best[1] if best else None


def raw_path(source: str, *parts: str) -> Path:
    return RAW_ROOT.joinpath(source, *parts)


def load_k400(path: Path | None = None) -> K400Vocabulary:
    target = path or raw_path("k400", "labels.json")
    entries = json.loads(Path(target).read_text(encoding="utf-8"))
    return K400Vocabulary(entries)


def load_moviegen_prompts(root: Path | None = None) -> list[PromptRef]:
    """Union of the video bench prompts and the audio ``video_prompt`` fields."""
    base = Path(root) if root else raw_path("moviegen")
    refs: list[PromptRef] = []
    video = base / "benchmark" / "MovieGenVideoBench.txt"
    for line_no, line in enumerate(video.read_text(encoding="utf-8").splitlines(), start=1):
        text = normalize_space(line)
        if not text:
            continue
        refs.append(PromptRef(text, "moviegen", "benchmark/MovieGenVideoBench.txt", line_no, sha256_text(text), word_count(text)))
    for name in ("MovieGenAudioBenchSfx.jsonl", "MovieGenAudioBenchSfxMusic.jsonl"):
        audio = base / "benchmark" / name
        for line_no, line in enumerate(audio.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = normalize_space(row.get("video_prompt", ""))
            if not text:
                continue
            refs.append(PromptRef(text, "moviegen", f"benchmark/{name}", line_no, sha256_text(text), word_count(text)))
    deduped: list[PromptRef] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.sha256 in seen:
            continue
        seen.add(ref.sha256)
        deduped.append(ref)
    return deduped


def load_flickr_exclusions(base: Path | None = None) -> set[tuple[str, int]]:
    directory = Path(base) if base else raw_path("flickr30k")
    excluded: set[tuple[str, int]] = set()
    path = directory / "UNRELATED_CAPTIONS"
    if not path.exists():
        return excluded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            excluded.add((parts[0], int(parts[1])))
    return excluded


def load_flickr_splits(base: Path | None = None) -> dict[str, list[str]]:
    directory = Path(base) if base else raw_path("flickr30k")
    return {
        split: [line.strip() for line in (directory / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        for split in ("train", "val", "test")
    }


def parse_flickr_caption(raw: str) -> tuple[str, tuple[Mention, ...]]:
    mentions: list[Mention] = []
    for match in FLICKR_MENTION_RE.finditer(raw):
        chain, types_blob, phrase = match.group(1), match.group(2), match.group(3)
        mentions.append(
            Mention(
                phrase=normalize_space(phrase),
                chain=chain,
                types=tuple(types_blob.split("/")),
                positioned=chain != "0",
            )
        )
    plain = normalize_space(FLICKR_MENTION_RE.sub(lambda m: m.group(3), raw))
    return plain, tuple(mentions)


def iter_flickr_captions(
    split: str = "train",
    *,
    base: Path | None = None,
    limit: int | None = None,
) -> Iterator[FlickrCaption]:
    directory = Path(base) if base else raw_path("flickr30k")
    splits = load_flickr_splits(directory)
    if split not in splits:
        raise KeyError(f"unknown flickr split {split}")
    excluded = load_flickr_exclusions(directory)
    wanted = set(splits[split])
    emitted = 0
    with zipfile.ZipFile(directory / "annotations.zip") as archive:
        names = sorted(n for n in archive.namelist() if n.startswith("Sentences/") and n.endswith(".txt"))
        for name in names:
            image_id = Path(name).stem
            if image_id not in wanted:
                continue
            if archive.getinfo(name).file_size > 1_000_000:
                raise ValueError(f"Oversized sentence member: {name}")
            lines = archive.read(name).decode("utf-8").splitlines()
            for index, raw in enumerate(lines, start=1):
                if (image_id, index) in excluded:
                    continue
                text, mentions = parse_flickr_caption(raw)
                if not text:
                    continue
                yield FlickrCaption(image_id=image_id, index=index, split=split, text=text, mentions=mentions, raw=normalize_space(raw))
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def json_array(stream: io.TextIOBase) -> Iterator[Any]:
    """Stream a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    ended = False
    while not ended:
        chunk = stream.read(1024 * 1024)
        eof = not chunk
        buffer += chunk
        if not started:
            buffer = buffer.lstrip()
            if not buffer.startswith("["):
                raise ValueError("Expected JSON array")
            buffer = buffer[1:]
            started = True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                ended = True
                break
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Truncated JSON array") from None
                break
            yield value
            buffer = buffer[end:]


def _entity_name(entity: Mapping[str, Any]) -> str:
    names = entity.get("names", [])
    return str(entity.get("name") or (names[0] if names else "")).strip()


def iter_vg_edges(*, base: Path | None = None, limit: int | None = None) -> Iterator[VGEdge]:
    """Yield whitelisted, de-duplicated Visual Genome relation edges."""
    directory = Path(base) if base else raw_path("visual_genome")
    seen: set[tuple[Any, Any, str, Any]] = set()
    emitted = 0
    with zipfile.ZipFile(directory / "relationships.json.zip") as archive:
        member = archive.getinfo("relationships.json")
        if member.file_size > 3_000_000_000:
            raise ValueError("Unexpected uncompressed size")
        with io.TextIOWrapper(archive.open(member), encoding="utf-8") as stream:
            for image in json_array(stream):
                image_id = image.get("image_id", image.get("id"))
                for rel in image.get("relationships", []):
                    predicate = normalize_phrase(rel.get("predicate", ""))
                    direction = next((k for k, values in VG_PREDICATES.items() if predicate in values), None)
                    if direction is None:
                        continue
                    subject, obj = rel.get("subject", {}), rel.get("object", {})
                    subject_name, object_name = _entity_name(subject), _entity_name(obj)
                    subject_id, object_id = subject.get("object_id"), obj.get("object_id")
                    if not subject_name or not object_name or subject_id is None or object_id is None:
                        continue
                    if subject_id == object_id:
                        continue
                    key = (image_id, subject_id, direction, object_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield VGEdge(
                        image_id=int(image_id),
                        relationship_id=rel.get("relationship_id"),
                        predicate=predicate,
                        direction=direction,
                        subject=subject_name,
                        object=object_name,
                        subject_id=subject_id,
                        object_id=object_id,
                    )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


@dataclass(frozen=True)
class VGRegionSet:
    image_id: int
    phrases: tuple[str, ...]


def iter_vg_regions(*, base: Path | None = None, limit: int | None = None) -> Iterator[VGRegionSet]:
    """Stream human-written Visual Genome region phrases (licence-clean)."""
    directory = Path(base) if base else raw_path("visual_genome")
    archive_path = directory / "region_descriptions.json.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"{archive_path}; run scripts/inspect_vg_regions.py --download")
    emitted = 0
    with zipfile.ZipFile(archive_path) as archive:
        member = archive.getinfo("region_descriptions.json")
        with io.TextIOWrapper(archive.open(member), encoding="utf-8") as stream:
            for image in json_array(stream):
                image_id = image.get("image_id", image.get("id"))
                phrases = tuple(
                    normalize_space(region.get("phrase", ""))
                    for region in image.get("regions", [])
                    if normalize_space(region.get("phrase", ""))
                )
                if not phrases:
                    continue
                yield VGRegionSet(image_id=int(image_id), phrases=phrases)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def load_snli_split(split: str, *, base: Path | None = None) -> tuple[list[dict[str, Any]] | None, str]:
    """Read an SNLI parquet split; returns ``(rows, status)``.

    ``status`` is ``ok`` or ``unavailable_no_pyarrow``; never silently empty.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return None, "unavailable_no_pyarrow"
    directory = Path(base) if base else raw_path("snli")
    path = directory / f"{split}-00000-of-00001.parquet"
    table = pq.read_table(path, columns=["premise", "hypothesis", "label"])
    rows = table.to_pylist()
    return rows, "ok"
