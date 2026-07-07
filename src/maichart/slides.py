"""Slide trajectory catalog for V1 Maidata/Simai-like parsing.

V1 treats slides as references to known maimai trajectory patterns. It does not
generate or sample arbitrary free-form geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PathArgCount = int | Literal["variable"]


@dataclass(frozen=True, slots=True)
class SlidePatternDefinition:
    """A known slide pattern accepted by the V1 parser."""

    pattern: str
    category: str
    path_arg_count: PathArgCount
    description: str
    supported: bool


SLIDE_PATTERN_DEFINITIONS: tuple[SlidePatternDefinition, ...] = (
    SlidePatternDefinition("-", "line", 1, "Straight slide to one button.", True),
    SlidePatternDefinition("<", "arc", 1, "Counter-clockwise arc-like slide.", True),
    SlidePatternDefinition(">", "arc", 1, "Clockwise arc-like slide.", True),
    SlidePatternDefinition("p", "curve", 1, "Known p-shaped maimai slide.", True),
    SlidePatternDefinition("q", "curve", 1, "Known q-shaped maimai slide.", True),
    SlidePatternDefinition("pp", "curve", 1, "Known pp-shaped maimai slide.", True),
    SlidePatternDefinition("qq", "curve", 1, "Known qq-shaped maimai slide.", True),
    SlidePatternDefinition("s", "zigzag", 1, "Known s-shaped maimai slide.", True),
    SlidePatternDefinition("z", "zigzag", 1, "Known z-shaped maimai slide.", True),
    SlidePatternDefinition("v", "composite", 1, "Known lowercase-v slide.", True),
    SlidePatternDefinition("V", "composite", 2, "V-shaped slide with two path arguments.", True),
    SlidePatternDefinition("w", "composite", 1, "Known w-shaped maimai slide.", True),
)

SLIDE_PATTERN_CATALOG: dict[str, SlidePatternDefinition] = {
    definition.pattern: definition for definition in SLIDE_PATTERN_DEFINITIONS
}


def get_slide_pattern_definition(pattern: str) -> SlidePatternDefinition | None:
    """Return the V1 catalog definition for a slide pattern."""

    return SLIDE_PATTERN_CATALOG.get(pattern)


def is_supported_slide_pattern(pattern: str) -> bool:
    """Return whether V1 accepts this slide pattern as a known trajectory."""

    definition = get_slide_pattern_definition(pattern)
    return bool(definition and definition.supported)
