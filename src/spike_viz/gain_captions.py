"""Glyph wrapping for gain-triptych captions."""

from __future__ import annotations

import math

from PIL import ImageFont

PANEL_MARGIN = 24


def _text_width(font: ImageFont.ImageFont, text: str) -> float:
    return float(font.getlength(text))


def _widest_glyph(font: ImageFont.ImageFont, texts: list[str]) -> int:
    widest = 1
    for text in texts:
        for char in text:
            widest = max(widest, math.ceil(_text_width(font, char)))
    return widest


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap ``text`` so each line's advance fits ``max_width`` pixels."""
    lines: list[str] = []
    current = ""
    for char in text:
        if char.isspace() and not current:
            continue
        trial = current + char
        if current and _text_width(font, trial) > max_width:
            lines.append(current.rstrip())
            current = "" if char.isspace() else char
            continue
        current = trial
    if current.strip():
        lines.append(current.rstrip())
    return lines or [""]


def display_captions(
    captions: list[list[str]], font: ImageFont.ImageFont, panel_w: int
) -> tuple[list[list[str]], int]:
    texts = [text for caption in captions for text in caption]
    content_w = max(panel_w - 2 * PANEL_MARGIN, _widest_glyph(font, texts))
    wrapped: list[list[str]] = []
    for caption in captions:
        lines: list[str] = []
        for text in caption:
            lines.extend(_wrap_text(text, font, content_w))
        wrapped.append(lines)
    return wrapped, content_w + 2 * PANEL_MARGIN
