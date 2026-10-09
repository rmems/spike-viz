"""CPU preferred-direction stills from explicit export geometry, never inferred."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

from spike_viz.export import AxonExportCase


def render_direction_ring(
    case: AxonExportCase,
    *,
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
    raw_angles = meta.get("preferred_angles_radians")
    if (
        not isinstance(raw_angles, list)
        or len(raw_angles) != n_neurons
        or any(
            isinstance(a, bool) or not isinstance(a, (int, float)) for a in raw_angles
        )
    ):
        raise ValueError(
            "preferred_angles_radians must be a list of N finite numbers from the export"
        )
    try:
        angles = np.asarray(raw_angles, dtype=np.float64)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "preferred_angles_radians must contain finite numbers"
        ) from exc
    if not np.all(np.isfinite(angles)):
        raise ValueError("preferred_angles_radians must contain finite numbers")
    if window is None:
        window = (n_steps - 1, n_steps)
    if (
        not isinstance(window, tuple)
        or len(window) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in window)
        or not 0 <= window[0] < window[1] <= n_steps
    ):
        raise ValueError(
            "window must be integer steps (start, stop) with 0 <= start < stop <= T"
        )
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 32:
        raise ValueError("radius must be an integer >= 32 pixels")

    start, stop = window
    events = case.events
    selected = (events.t >= start) & (events.t < stop)
    counts = np.bincount(events.neuron_id[selected], minlength=n_neurons)
    peak = int(counts.max())
    strength = counts / peak if peak else np.zeros(n_neurons)

    # Minimum width keeps full provenance readable even for a small ring.
    width = max(640, 2 * radius + 160)
    cx, cy = width // 2, width // 2 + 40
    provenance = (
        f"encoder={meta['encoder']} | dt={meta['dt_seconds']:g} s/step | seed={meta['seed']} | "
        f"N={n_neurons} | T={n_steps}\n"
        f"axon-encoder commit={meta.get('axon_encoder_git_sha') or 'unknown'}\n"
        f"synthetic={meta.get('synthetic', 'unknown')} | steps [{start}, {stop}) | "
        f"source_events={int(counts.sum())}"
    )
    caption = "\n".join(
        textwrap.fill(line, width=(width - 48) // 8) for line in provenance.splitlines()
    )
    font = ImageFont.load_default(size=14)
    footer_y = width + 20
    image = Image.new(
        "RGB", (width, footer_y + 26 + len(caption.splitlines()) * 20), (8, 12, 23)
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (24, 22),
        "PREFERRED-DIRECTION POPULATION",
        font=ImageFont.load_default(size=22),
        fill=(227, 236, 250),
    )
    draw.text(
        (24, 56),
        f"Steps [{start}, {stop})  |  {int(counts.sum())} source events",
        font=font,
        fill=(146, 162, 185),
    )
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=(43, 58, 78),
        width=2,
    )
    for angle, label in (
        (0, "0 rad"),
        (math.pi / 2, "pi/2 rad"),
        (math.pi, "pi rad"),
        (3 * math.pi / 2, "3pi/2 rad"),
    ):
        x, y = (
            cx + (radius + 30) * math.cos(angle),
            cy - (radius + 30) * math.sin(angle),
        )
        draw.text((x, y), label, anchor="mm", font=font, fill=(146, 162, 185))

    positions = [(cx + radius * math.cos(a), cy - radius * math.sin(a)) for a in angles]
    if glow:
        halo = Image.new("RGB", image.size)
        halo_draw = ImageDraw.Draw(halo)
        for (x, y), s in zip(positions, strength):
            if s > 0:
                halo_draw.ellipse(
                    (x - 12, y - 12, x + 12, y + 12),
                    fill=(0, int(110 * s), int(160 * s)),
                )
        image = ImageChops.add(image, halo.filter(ImageFilter.GaussianBlur(10)))
        draw = ImageDraw.Draw(image)
    # Inactive markers first so they cannot hide activity at coincident angles.
    for active in (False, True):
        for neuron, ((x, y), s) in enumerate(zip(positions, strength)):
            if bool(s > 0) != active:
                continue
            dot = 4 + 6 * math.sqrt(s)
            color = (int(40 + 160 * s), int(55 + 185 * s), int(75 + 180 * s))
            draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=color)
            if n_neurons <= 32:
                angle = angles[neuron]
                draw.text(
                    (
                        cx + (radius - 24) * math.cos(angle),
                        cy - (radius - 24) * math.sin(angle),
                    ),
                    f"n{neuron}",
                    anchor="mm",
                    font=font,
                    fill=(146, 162, 185),
                )
    draw.text(
        (cx, cy),
        f"{int(counts.sum())} events",
        anchor="mm",
        font=ImageFont.load_default(size=24),
        fill=(227, 236, 250),
    )
    draw.text(
        (24, footer_y - 28),
        f"Dot size / brightness: window count, peak={peak}; muted = 0 events",
        font=font,
        fill=(146, 162, 185),
    )
    draw.multiline_text(
        (24, footer_y), caption, font=font, spacing=6, fill=(146, 162, 185)
    )
    image.info["provenance"] = provenance
    if out_path is not None:
        pnginfo = PngInfo()
        pnginfo.add_text("provenance", provenance)
        image.save(Path(out_path), format="PNG", pnginfo=pnginfo)
    return image
