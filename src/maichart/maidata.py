"""Raw Maidata metadata parser.

This module only handles the first parser milestone: top-level metadata and raw
``inote_x`` difficulty blocks. It intentionally does not parse note tokens.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FIELD_RE = re.compile(r"^&(?P<key>[A-Za-z][A-Za-z0-9_]*)=")
DIFFICULTY_RANGE = range(1, 6)
DEFAULT_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "shift_jis", "cp932", "gbk")


@dataclass(slots=True)
class RawDifficultyBlock:
    """Raw data for one Maidata difficulty block."""

    index: int
    level: str | None = None
    designer: str | None = None
    inote: str | None = None
    raw_level: str | None = None
    raw_designer: str | None = None
    raw_inote: str | None = None


@dataclass(slots=True)
class RawMaidataChart:
    """Top-level Maidata metadata plus raw difficulty blocks."""

    title: str | None = None
    artist: str | None = None
    wholebpm: str | None = None
    first: str | None = None
    levels: dict[int, str] = field(default_factory=dict)
    designers: dict[int, str] = field(default_factory=dict)
    difficulties: list[RawDifficultyBlock] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    raw: str | None = None


def parse_maidata_metadata(text: str) -> RawMaidataChart:
    """Parse top-level Maidata metadata and raw ``inote_x`` blocks.

    The parser recognizes ``&key=value`` lines and treats ``&inote_1=`` through
    ``&inote_5=`` as raw multi-line blocks that continue until the next top-level
    Maidata field. Individual note tokens are not parsed in this milestone.
    """

    fields: dict[str, str] = {}
    raw_fields: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    index = 0

    while index < len(lines):
        line = lines[index]
        match = FIELD_RE.match(line)
        if match is None:
            index += 1
            continue

        key = match.group("key")
        value_start = match.end()

        if _is_inote_key(key):
            value_parts = [line[value_start:]]
            index += 1
            while index < len(lines) and FIELD_RE.match(lines[index]) is None:
                value_parts.append(lines[index])
                index += 1
            value = "".join(value_parts)
        else:
            value = line[value_start:].rstrip("\r\n")
            index += 1

        fields[key] = value
        raw_fields[key] = f"&{key}={value}"

    levels = _indexed_fields(fields, "lv")
    designers = _indexed_fields(fields, "des")
    difficulties = [
        RawDifficultyBlock(
            index=difficulty_index,
            level=levels.get(difficulty_index),
            designer=designers.get(difficulty_index),
            inote=fields.get(f"inote_{difficulty_index}"),
            raw_level=raw_fields.get(f"lv_{difficulty_index}"),
            raw_designer=raw_fields.get(f"des_{difficulty_index}"),
            raw_inote=raw_fields.get(f"inote_{difficulty_index}"),
        )
        for difficulty_index in DIFFICULTY_RANGE
        if (
            difficulty_index in levels
            or difficulty_index in designers
            or f"inote_{difficulty_index}" in fields
        )
    ]

    return RawMaidataChart(
        title=fields.get("title"),
        artist=fields.get("artist"),
        wholebpm=fields.get("wholebpm"),
        first=fields.get("first"),
        levels=levels,
        designers=designers,
        difficulties=difficulties,
        fields=fields,
        raw=text,
    )


def raw_maidata_to_dict(chart: RawMaidataChart) -> dict[str, Any]:
    """Convert raw Maidata metadata to JSON-compatible primitives."""

    return asdict(chart)


def raw_maidata_to_json(chart: RawMaidataChart, *, indent: int = 2) -> str:
    """Serialize raw Maidata metadata to JSON."""

    return json.dumps(raw_maidata_to_dict(chart), ensure_ascii=False, indent=indent)


def read_text_with_encoding(
    path: str | Path,
    encoding: str | None = None,
) -> tuple[str, str]:
    """Read text with explicit or auto-detected common Maidata encodings."""

    source = Path(path)
    if encoding is not None:
        return source.read_text(encoding=encoding), encoding

    errors: list[str] = []
    for candidate in DEFAULT_ENCODING_CANDIDATES:
        try:
            return source.read_text(encoding=candidate), candidate
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate}: {exc}")

    message = "; ".join(errors)
    raise UnicodeDecodeError(
        "maidata",
        source.read_bytes(),
        0,
        1,
        f"Unable to decode with candidate encodings: {message}",
    )


def parse_maidata_file(
    path: str | Path,
    encoding: str | None = None,
) -> RawMaidataChart:
    """Read and parse a Maidata text file."""

    text, _used_encoding = read_text_with_encoding(path, encoding=encoding)
    return parse_maidata_metadata(text)


def _is_inote_key(key: str) -> bool:
    return re.fullmatch(r"inote_[1-5]", key) is not None


def _indexed_fields(fields: dict[str, str], prefix: str) -> dict[int, str]:
    indexed: dict[int, str] = {}
    for difficulty_index in DIFFICULTY_RANGE:
        key = f"{prefix}_{difficulty_index}"
        if key in fields:
            indexed[difficulty_index] = fields[key]
    return indexed
