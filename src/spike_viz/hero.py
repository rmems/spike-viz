"""Provenance-pinned, same-stimulus rate gain comparisons (paint only)."""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from spike_viz.events import SpikeEvents
from spike_viz.export import AxonExportCase
from spike_viz.render import PathLike, render_raster


def _validate_gain_cases(cases: tuple[AxonExportCase, ...]) -> list[float]:
    """Reject comparisons that could imply a gain effect from unrelated data."""
    reference = cases[0].meta
    common = (
        "encoder",
        "encoder_config",
        "dt_seconds",
        "seed",
        "n_neurons",
        "n_steps",
        "axon_encoder_git_sha",
    )
    stimulus = None
    rates = []
    for case in cases:
        meta = case.meta
        if meta.get("synthetic") is not False or meta.get("encoder") != "rate":
            raise ValueError("gain triptych requires real rate encoder exports")
        sha = meta.get("axon_encoder_git_sha")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ValueError("gain triptych requires a full axon-encoder commit")
        for key in common:
            if key not in meta or meta[key] != reference.get(key):
                raise ValueError(f"gain cases must have identical {key}")
        for key in ("schema_version", "axon_encoder_version", "exporter_patch_sha256"):
            if meta.get(key) != reference.get(key):
                raise ValueError(f"gain cases must have identical {key}")
        gains = meta.get("encoding_gains")
        if not isinstance(gains, dict) or any(
            gains.get(key) != 1.0
            for key in ("threshold_scale", "sensitivity_scale", "latency_scale")
        ):
            raise ValueError("gain cases must document identity non-rate gains")
        rate = gains.get("firing_rate_scale")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate < 0
        ):
            raise ValueError("firing-rate gain must be finite and non-negative")
        rates.append(float(rate))
        if "modulators" not in meta or (
            meta["modulators"] is not None and not isinstance(meta["modulators"], dict)
        ):
            raise ValueError("modulators must document a summary object or null")
        if case.stimulus_path is None:
            raise ValueError("gain comparison requires exported stimulus.npy")
        current = np.load(case.stimulus_path, allow_pickle=False)
        if current.shape != (meta["n_steps"], meta["n_neurons"]):
            raise ValueError("rate stimulus must have shape [T, N]")
        if stimulus is not None and not np.array_equal(stimulus, current):
            raise ValueError("gain cases must have identical stimulus arrays")
        stimulus = current
    if rates[0] != 0 or rates[1] != 1 or rates[2] <= 1:
        raise ValueError("expected gains: silence=0, normal=1, elevated>1")
    if len(cases[0].events):
        raise ValueError("zero-gain silence export must contain no events")
    return rates


def render_gain_triptych(
    silence: AxonExportCase,
    normal: AxonExportCase,
    elevated: AxonExportCase,
    *,
    out_path: PathLike | None = None,
    scale: int = 2,
) -> Image.Image:
    """Render three real, same-stimulus rate exports and their provenance.

    Cases must share encoder config, dt, seed, N, T and a full upstream commit,
    include identical ``stimulus.npy`` arrays, and document ``encoding_gains``
    (identity except firing-rate scales 0, 1, >1) and ``modulators`` (null for
    direct gains, or a summary object). Invalid comparisons raise ValueError.

    Cyan bins show event presence, independent of amplitude/polarity; duplicate
    events share a bin. No gain scaling, resampling, jitter or bloom is applied
    to the spikes. Time is horizontal, neuron is vertical, with identical axes.
    Captions are visible and also stored in PNG ``Description`` metadata.
    ``scale`` is a positive integer nearest-neighbor enlargement of the figure.
    CPU only; fonts use Pillow's built-in default, with no system font dependency.
    """
    cases = (silence, normal, elevated)
    rates = _validate_gain_cases(cases)
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be an integer >= 1")

    meta = silence.meta
    rasters = [
        render_raster(
            SpikeEvents(case.events.t, case.events.neuron_id),
            n_steps=meta["n_steps"],
            n_neurons=meta["n_neurons"],
            scale=3,
        )
        for case in cases
    ]
    font = ImageFont.load_default()
    raster_w, raster_h = rasters[0].size
    panel_w = max(240, raster_w + 48)
    wrap_width = (panel_w - 48) // 6
    captions = []
    for label, rate, case in zip(("Silence", "Normal", "Elevated"), rates, cases):
        mods = case.meta["modulators"]
        summary = (
            "none (direct gains)" if mods is None else json.dumps(mods, sort_keys=True)
        )
        captions.append(
            [
                label,
                f"firing_rate_scale={rate:g}",
                f"modulators: {summary}",
                f"{len(case.events)} exported events",
            ]
        )
    display_captions = [
        [line for text in caption for line in textwrap.wrap(text, width=wrap_width)]
        for caption in captions
    ]
    raster_y = 30 + 16 * max(map(len, display_captions))
    footer_y = raster_y + raster_h + 42
    provenance = [
        (
            f"encoder={meta['encoder']} | dt={meta['dt_seconds']:g} s | seed={meta['seed']} | "
            f"N={meta['n_neurons']} | T={meta['n_steps']}"
        ),
        f"axon-encoder commit: {meta['axon_encoder_git_sha']}",
        "Same stimulus / config; binary event occupancy; no Python encoding.",
    ]
    image = Image.new("RGB", (3 * panel_w, footer_y + 64), (8, 12, 24))
    draw = ImageDraw.Draw(image)
    for index, (raster, caption) in enumerate(zip(rasters, display_captions)):
        x = index * panel_w + 24
        draw.multiline_text(
            (x, 16), "\n".join(caption), font=font, fill=(225, 235, 250), spacing=6
        )
        pixels = np.asarray(raster).copy()
        pixels[np.any(pixels != 0, axis=2)] = (0, 230, 255)
        image.paste(Image.fromarray(pixels), (x, raster_y))
        draw.rectangle(
            (x - 1, raster_y - 1, x + raster_w, raster_y + raster_h),
            outline=(60, 75, 100),
        )
        draw.text(
            (x, raster_y + raster_h + 10),
            "time (steps) ->  |  rows: neurons",
            font=font,
            fill=(150, 165, 190),
        )
    draw.multiline_text(
        (24, footer_y),
        "\n".join(provenance),
        font=font,
        fill=(180, 195, 215),
        spacing=6,
    )
    image = image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )
    description = "\n\n".join("\n".join(caption) for caption in captions)
    description += "\n\n" + "\n".join(provenance)
    image.info["Description"] = description
    if out_path is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Description", description)
        image.save(Path(out_path), format="PNG", pnginfo=pnginfo)
    return image
