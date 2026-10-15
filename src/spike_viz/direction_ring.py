"""CPU preferred-direction stills from explicit export geometry, never inferred."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

from spike_viz.events import SpikeEvents
from spike_viz.export import AxonExportCase

_ANGLES_ERROR = (
    "preferred_angles_radians must be a list of N finite numbers from the export"
)
_FINITE_ERROR = "preferred_angles_radians must contain finite numbers"
_WINDOW_ERROR = "window must be integer steps (start, stop) with 0 <= start < stop <= T"
_RADIUS_ERROR = "radius must be an integer >= 32 pixels"
_NEURON_ID_ERROR = "event neuron_id must be within [0, n_neurons)"
_COINCIDENT_SPACING = 24
_INK = (227, 236, 250)
_MUTED = (146, 162, 185)
_BACKGROUND = (8, 12, 23)
_RING = (43, 58, 78)
_COMPASS = (
    (0.0, "0 rad"),
    (math.pi / 2, "pi/2 rad"),
    (math.pi, "pi rad"),
    (3 * math.pi / 2, "3pi/2 rad"),
)


class _Frame(NamedTuple):
    radius: int
    peak: int
    event_count: int
    start: int
    stop: int


class _Canvas(NamedTuple):
    cx: int
    cy: int
    radius: int
    footer_y: int
    font: ImageFont.ImageFont


def render_direction_ring(
    case: AxonExportCase,
    window: tuple[int, int] | None = None,
    radius: int = 220,
    glow: bool = True,
    out_path: str | Path | None = None,
) -> Image.Image:
    """Paint exported preferred directions with per-neuron event counts.

    ``case.meta['preferred_angles_radians']`` must contain N finite numbers,
    indexed by neuron_id. Zero points east; positive angles run counterclockwise.
    No geometry is derived from neuron order or scalar tuning centers. Labels
    ``n0``, ``n1``, etc. identify neuron IDs, not spike counts.

    ``window`` is a half-open step interval [start, stop). None selects the last
    export step [T-1, T), even if silent. All event records count once regardless
    of signed/zero amplitude; brightness and dot size scale by the largest count
    in this window. Muted dots mark inactive neurons, not synthetic spikes.

    Returns an RGB Pillow image with visible provenance and optional PNG export.
    The provenance is also stored in the image/PNG's ``provenance`` text field.
    CPU only; no GPU or extra plotting dependency is needed.
    """
    meta = case.meta
    n_neurons, n_steps = meta["n_neurons"], meta["n_steps"]
    angles = _preferred_angles(meta.get("preferred_angles_radians"), n_neurons)
    start, stop = _step_window(window, n_steps)
    radius = _require_radius(radius)
    counts, peak, strength = _window_strength(case.events, start, stop, n_neurons)
    event_count = int(counts.sum())
    frame = _Frame(radius, peak, event_count, start, stop)
    provenance = _provenance(meta, n_neurons, n_steps, frame)
    image = _paint_ring(angles, strength, glow, provenance, frame)
    image.info["provenance"] = provenance
    if out_path is not None:
        _save_png(image, out_path, provenance)
    return image


def _is_real_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _is_integer(value: object) -> bool:
    """Python or NumPy integers. Booleans compare equal to 0 and 1, so they are not integers here."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def _preferred_angles(raw: object, n_neurons: int) -> np.ndarray:
    if (
        not isinstance(raw, list)
        or len(raw) != n_neurons
        or any(not _is_real_number(value) for value in raw)
    ):
        raise ValueError(_ANGLES_ERROR)
    try:
        angles = np.asarray(raw, dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise ValueError(_FINITE_ERROR) from exc
    if not np.all(np.isfinite(angles)):
        raise ValueError(_FINITE_ERROR)
    return angles


def _step_window(window: tuple[int, int] | None, n_steps: int) -> tuple[int, int]:
    if window is None:
        window = (n_steps - 1, n_steps)
    if (
        not isinstance(window, tuple)
        or len(window) != 2
        or any(not _is_integer(value) for value in window)
        or not 0 <= window[0] < window[1] <= n_steps
    ):
        raise ValueError(_WINDOW_ERROR)
    return int(window[0]), int(window[1])


def _require_radius(radius: object) -> int:
    if not _is_integer(radius) or radius < 32:
        raise ValueError(_RADIUS_ERROR)
    return int(radius)


def _window_strength(
    events: SpikeEvents, start: int, stop: int, n_neurons: int
) -> tuple[np.ndarray, int, np.ndarray]:
    ids = events.neuron_id
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= n_neurons):
        raise ValueError(_NEURON_ID_ERROR)
    selected = (events.t >= start) & (events.t < stop)
    counts = np.bincount(ids[selected], minlength=n_neurons)
    peak = int(counts.max())
    strength = counts / peak if peak else np.zeros(n_neurons)
    return counts, peak, strength


def _provenance_text(value: object) -> str:
    """JSON-quote strings so lone surrogates survive PNG text encoding."""
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def _provenance(meta: dict, n_neurons: int, n_steps: int, frame: _Frame) -> str:
    raw_commit = meta.get("axon_encoder_git_sha")
    commit = _provenance_text(raw_commit) if raw_commit else "unknown"
    synthetic = meta.get("synthetic", "unknown")
    synthetic_text = _provenance_text(synthetic) if isinstance(synthetic, str) else synthetic
    return (
        f"encoder={_provenance_text(meta['encoder'])} | dt={meta['dt_seconds']} s/step | "
        f"seed={meta['seed']} | N={n_neurons} | T={n_steps}\n"
        f"axon-encoder commit={commit}\n"
        f"synthetic={synthetic_text} | steps [{frame.start}, {frame.stop}) | "
        f"source_events={frame.event_count}"
    )


def _paint_ring(
    angles: np.ndarray,
    strength: np.ndarray,
    glow: bool,
    provenance: str,
    frame: _Frame,
) -> Image.Image:
    width = max(640, 2 * frame.radius + 160)
    canvas = _Canvas(
        width // 2,
        width // 2 + 40,
        frame.radius,
        width + 20,
        ImageFont.load_default(size=14),
    )
    caption = "\n".join(
        textwrap.fill(line, width=(width - 48) // 8) for line in provenance.splitlines()
    )
    image = Image.new(
        "RGB",
        (width, canvas.footer_y + 26 + len(caption.splitlines()) * 20),
        _BACKGROUND,
    )
    draw = ImageDraw.Draw(image)
    _draw_header(draw, canvas.font, frame)
    draw.ellipse(
        (
            canvas.cx - frame.radius,
            canvas.cy - frame.radius,
            canvas.cx + frame.radius,
            canvas.cy + frame.radius,
        ),
        outline=_RING,
        width=2,
    )
    _draw_compass(draw, canvas)
    positions = _spread_coincident(_positions(canvas, angles), strength, angles)
    if glow:
        image = _with_glow(image, positions, strength)
        draw = ImageDraw.Draw(image)
    _draw_markers(draw, positions, strength, angles, canvas)
    _draw_center_and_footer(draw, canvas, frame, caption)
    return image


def _draw_header(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, frame: _Frame) -> None:
    draw.text(
        (24, 22),
        "PREFERRED-DIRECTION POPULATION",
        font=ImageFont.load_default(size=22),
        fill=_INK,
    )
    draw.text(
        (24, 56),
        f"Steps [{frame.start}, {frame.stop})  |  {frame.event_count} source events",
        font=font,
        fill=_MUTED,
    )


def _draw_compass(draw: ImageDraw.ImageDraw, canvas: _Canvas) -> None:
    for angle, label in _COMPASS:
        x = canvas.cx + (canvas.radius + 30) * math.cos(angle)
        y = canvas.cy - (canvas.radius + 30) * math.sin(angle)
        draw.text((x, y), label, anchor="mm", font=canvas.font, fill=_MUTED)


def _positions(canvas: _Canvas, angles: np.ndarray) -> list[tuple[float, float]]:
    return [
        (
            canvas.cx + canvas.radius * math.cos(angle),
            canvas.cy - canvas.radius * math.sin(angle),
        )
        for angle in angles
    ]


def _with_glow(
    image: Image.Image, positions: list[tuple[float, float]], strength: np.ndarray
) -> Image.Image:
    halo = Image.new("RGB", image.size)
    halo_draw = ImageDraw.Draw(halo)
    for (x, y), scale in zip(positions, strength):
        if scale > 0:
            halo_draw.ellipse(
                (x - 12, y - 12, x + 12, y + 12),
                fill=(0, int(110 * scale), int(160 * scale)),
            )
    return ImageChops.add(image, halo.filter(ImageFilter.GaussianBlur(10)))


def _spread_coincident(
    positions: list[tuple[float, float]],
    strength: np.ndarray,
    angles: np.ndarray,
) -> list[tuple[float, float]]:
    """Offset neurons that share a pixel so a later, weaker dot cannot cover a stronger one.

    The strongest neuron in a coincident group stays on the exported angle. Weaker
    neurons move along the tangent, alternating sides, far enough that marker disks
    do not cover each other. Unique angles are unchanged.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for neuron, (x, y) in enumerate(positions):
        groups.setdefault((round(x), round(y)), []).append(neuron)
    placed = list(positions)
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda neuron: (float(strength[neuron]), neuron))
        angle = float(angles[ordered[-1]])
        tangent_x = -math.sin(angle)
        tangent_y = -math.cos(angle)
        for index, neuron in enumerate(ordered[:-1]):
            step = (index // 2 + 1) * _COINCIDENT_SPACING
            sign = -1 if index % 2 else 1
            x, y = positions[neuron]
            placed[neuron] = (
                x + sign * step * tangent_x,
                y + sign * step * tangent_y,
            )
    return placed


def _draw_markers(
    draw: ImageDraw.ImageDraw,
    positions: list[tuple[float, float]],
    strength: np.ndarray,
    angles: np.ndarray,
    canvas: _Canvas,
) -> None:
    # Inactive first, then weaker activity before stronger, so a later disk cannot hide a higher count.
    n_neurons = len(angles)
    order = sorted(
        range(n_neurons),
        key=lambda neuron: (float(strength[neuron]) > 0, float(strength[neuron]), neuron),
    )
    for neuron in order:
        x, y = positions[neuron]
        marker = (x, y, float(strength[neuron]), float(angles[neuron]), neuron)
        _draw_neuron(draw, marker, canvas, n_neurons)


def _draw_neuron(
    draw: ImageDraw.ImageDraw,
    marker: tuple[float, float, float, float, int],
    canvas: _Canvas,
    n_neurons: int,
) -> None:
    x, y, scale, angle, neuron = marker
    dot = 4 + 6 * math.sqrt(scale)
    color = (int(40 + 160 * scale), int(55 + 185 * scale), int(75 + 180 * scale))
    draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=color)
    if n_neurons <= 32:
        true_x = canvas.cx + canvas.radius * math.cos(angle)
        true_y = canvas.cy - canvas.radius * math.sin(angle)
        label_x = canvas.cx + (canvas.radius - 24) * math.cos(angle) + (x - true_x)
        label_y = canvas.cy - (canvas.radius - 24) * math.sin(angle) + (y - true_y)
        draw.text((label_x, label_y), f"n{neuron}", anchor="mm", font=canvas.font, fill=_MUTED)


def _draw_center_and_footer(
    draw: ImageDraw.ImageDraw, canvas: _Canvas, frame: _Frame, caption: str
) -> None:
    draw.text(
        (canvas.cx, canvas.cy),
        f"{frame.event_count} events",
        anchor="mm",
        font=ImageFont.load_default(size=24),
        fill=_INK,
    )
    draw.text(
        (24, canvas.footer_y - 28),
        f"Dot size / brightness: window count, peak={frame.peak}; muted = 0 events",
        font=canvas.font,
        fill=_MUTED,
    )
    draw.multiline_text(
        (24, canvas.footer_y), caption, font=canvas.font, spacing=6, fill=_MUTED
    )


def _save_png(image: Image.Image, out_path: str | Path, provenance: str) -> None:
    pnginfo = PngInfo()
    pnginfo.add_text("provenance", provenance)
    image.save(Path(out_path), format="PNG", pnginfo=pnginfo)
