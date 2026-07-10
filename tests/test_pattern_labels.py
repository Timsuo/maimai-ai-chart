from __future__ import annotations

from maichart.training.frame_label_codec import (
    START_PATTERN_IGNORE_INDEX,
    START_PATTERN_TO_ID,
    FrameLabelCodec,
    derive_frame_pattern,
)


def test_empty_frame_derives_none() -> None:
    derived = derive_frame_pattern(_labels(has_note=False))

    assert derived["activity_presence"] is False
    assert derived["note_start_presence"] is False
    assert derived["chord_size"] == 0
    assert derived["chord_size_class"] == 0
    assert derived["pattern_type"] == "none"


def test_active_hold_only_derives_active_hold() -> None:
    assert _pattern(_labels(hold_active_count=1)) == "active_hold"


def test_active_slide_only_derives_active_slide() -> None:
    assert _pattern(_labels(slide_active_count=1)) == "active_slide"


def test_single_tap_derives_single_tap() -> None:
    derived = derive_frame_pattern(_labels(note_count=1, tap_count=1))

    assert derived["note_start_presence"] is True
    assert derived["chord_size"] == 1
    assert derived["chord_size_class"] == 1
    assert derived["pattern_type"] == "single_tap"


def test_double_tap_derives_double_tap() -> None:
    derived = derive_frame_pattern(_labels(note_count=2, tap_count=2))

    assert derived["chord_size_class"] == 2
    assert derived["pattern_type"] == "double_tap"


def test_three_or_more_taps_derive_multi_tap() -> None:
    derived = derive_frame_pattern(_labels(note_count=3, tap_count=3))

    assert derived["chord_size_class"] == 3
    assert derived["pattern_type"] == "multi_tap"


def test_break_derives_break() -> None:
    assert _pattern(_labels(note_count=1, tap_count=1, break_count=1)) == "break"


def test_hold_start_derives_single_hold() -> None:
    assert _pattern(_labels(note_count=1, hold_start_count=1)) == "single_hold"


def test_hold_chord_derives_hold_chord() -> None:
    assert _pattern(_labels(note_count=2, hold_start_count=2)) == "hold_chord"


def test_slide_start_derives_single_slide() -> None:
    assert _pattern(_labels(note_count=1, slide_start_count=1)) == "single_slide"


def test_double_slide_derives_slide_chord() -> None:
    assert _pattern(_labels(note_count=2, slide_start_count=2)) == "slide_chord"


def test_tap_and_slide_derives_tap_slide_mix() -> None:
    assert _pattern(_labels(note_count=2, tap_count=1, slide_start_count=1)) == "tap_slide_mix"


def test_touch_derives_touch() -> None:
    assert _pattern(_labels(note_count=1, touch_count=1)) == "touch"


def test_touch_hold_derives_touch_hold() -> None:
    assert _pattern(_labels(note_count=1, touch_hold_start_count=1)) == "touch_hold"


def test_frame_label_codec_outputs_start_targets() -> None:
    codec = FrameLabelCodec()
    encoded = codec.encode_frames(
        [
            {"labels": _labels(has_note=False)},
            {"labels": _labels(note_count=1, tap_count=1)},
            {"labels": _labels(note_count=2, tap_count=2)},
            {"labels": _labels(note_count=3, tap_count=3)},
            {"labels": _labels(note_count=1, slide_start_count=1)},
        ]
    )

    assert encoded["note_start"].tolist() == [[0.0], [1.0], [1.0], [1.0], [1.0]]
    assert encoded["pattern_start"].tolist() == [
        START_PATTERN_IGNORE_INDEX,
        START_PATTERN_TO_ID["single_tap"],
        START_PATTERN_TO_ID["double_tap"],
        START_PATTERN_TO_ID["multi_tap"],
        START_PATTERN_TO_ID["single_slide"],
    ]
    assert encoded["chord_size_start"].tolist() == [START_PATTERN_IGNORE_INDEX, 0, 1, 2, 0]


def _pattern(labels: dict[str, int | bool]) -> str:
    return str(derive_frame_pattern(labels)["pattern_type"])


def _labels(**overrides: int | bool) -> dict[str, int | bool]:
    labels: dict[str, int | bool] = {
        "has_note": True,
        "note_count": 0,
        "tap_count": 0,
        "break_count": 0,
        "hold_start_count": 0,
        "hold_active_count": 0,
        "slide_start_count": 0,
        "slide_active_count": 0,
        "touch_count": 0,
        "touch_hold_start_count": 0,
    }
    labels.update(overrides)
    return labels
