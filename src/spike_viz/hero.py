"""Provenance-pinned hero stills (paint only)."""

from __future__ import annotations

import json
import math
import re
import textwrap
from decimal import Decimal
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from spike_viz.events import SpikeEvents
from spike_viz.export import AxonExportCase, load_axon_export
from spike_viz.io import SpikeIOError, load_dense
from spike_viz.render import PathLike, render_raster

def _format_number(value: float) -> str:
    """Shortest round-trip text, without a trailing ``.0`` on whole numbers.

    Default ``g`` formatting keeps six significant digits, so a precise
    ``dt_seconds`` would be painted as a different interval.
    """
    text = format(float(value), "")
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def _format_duration_ms(n_steps: int, dt_seconds: float) -> str:
    """Milliseconds as the decimal product of the round-trip interval and ``T``.

    ``n_steps * dt_seconds * 1000`` in binary float can paint noise such as
    ``1.9200000000000002`` for ``dt_seconds=0.00003``.
    """
    duration = Decimal(_format_number(dt_seconds)) * int(n_steps) * 1000
    return format(duration.normalize(), "f")


def _stimulus_for_encoder(case: AxonExportCase, encoder: str) -> np.ndarray:
    """Return the finite ``[T, N]`` stimulus for one real, pinned export."""
    if case.meta["encoder"] != encoder:
        raise SpikeIOError(f"{case.path}: expected {encoder} encoder")
    if case.meta.get("synthetic") is not False:
        raise SpikeIOError(f"{case.path}: hero requires synthetic=false")
    sha = case.meta.get("axon_encoder_git_sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise SpikeIOError(f"{case.path}: full axon_encoder_git_sha required")
    if case.stimulus_path is None:
        raise SpikeIOError(f"{case.path}: stimulus.npy required for comparison")
    stimulus = load_dense(case.stimulus_path)
    if stimulus.shape != (case.meta["n_steps"], case.meta["n_neurons"]):
        raise SpikeIOError(f"{case.path}: stimulus must have shape [T, N]")
    if not np.all(np.isfinite(stimulus)):
        raise SpikeIOError(f"{case.path}: stimulus must be finite")
    return stimulus


def render_rate_poisson_hero(
    rate_case_dir: PathLike,
    poisson_case_dir: PathLike,
    *,
    rate_mode: Literal["streaming", "batch"],
    out_path: PathLike | None = None,
) -> Image.Image:
    """Compose equal-scale Rate/Poisson rasters with visible provenance.

    Both cases must be real exports with full axon-encoder commit SHAs,
    identical ``stimulus.npy`` arrays, and equal N, T, and dt_seconds.
    Incompatible or missing exports raise ``SpikeIOError``; nothing is encoded
    or substituted. ``rate_mode`` is required because the export schema does
    not record it: callers must check the pinned exporter, not infer the mode
    from the encoder name. The checked-in ``shared_sine_v1`` Rate is streaming.

    Returns an RGB image with a textual ``Description`` in ``image.info``;
    if ``out_path`` is supplied, also writes a PNG with that metadata. The
    64-step, 8-neuron fixtures produce 2048 x 720; larger grids grow the canvas
    rather than downsampling away events.
    """
    if rate_mode not in ("streaming", "batch"):
        raise ValueError("rate_mode must be 'streaming' or 'batch'")

    cases = (load_axon_export(rate_case_dir), load_axon_export(poisson_case_dir))
    for key in ("dt_seconds", "n_steps", "n_neurons"):
        if cases[0].meta[key] != cases[1].meta[key]:
            raise SpikeIOError(f"hero requires equal {key} for shared scales")

    stimuli = []
    for case, encoder in zip(cases, ("rate", "poisson")):
        stimuli.append(_stimulus_for_encoder(case, encoder))
    if not np.array_equal(stimuli[0], stimuli[1]):
        raise SpikeIOError("hero requires the same stimulus in both exports")

    rate_behavior = (
        "deterministic streaming" if rate_mode == "streaming" else "stochastic batch"
    )
    titles = (f"RateEncoder ({rate_behavior})", "PoissonEncoder (stochastic)")
    captions = []
    width = max(440, cases[0].meta["n_steps"])
    height = max(144, cases[0].meta["n_neurons"])
    canvas = Image.new("RGB", (2 * width + 144, height + 216), "#0b111b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (48, 16),
        "RATE vs POISSON / same stimulus, shared axes",
        font=font,
        fill="#e4edf7",
    )
    draw.text(
        (48, 34), "axon-encoder exports / spike-viz paints", font=font, fill="#a5b6cc"
    )
    for case, title, x, accent in zip(
        cases, titles, (48, width + 96), ("#55ddeb", "#ffc174")
    ):
        meta = case.meta
        draw.text((x, 58), title, font=font, fill=accent)
        # Event presence, not signed polarity or within-panel normalization:
        # every occupied bin has the same brightness in both panels.
        events = case.events
        raster = np.zeros((meta["n_steps"], meta["n_neurons"]), dtype=bool)
        raster[events.t, events.neuron_id] = True
        panel = render_raster(raster).resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(panel, (x, 78))
        draw.rectangle((x - 1, 77, x + width, height + 78), outline="#41516a")
        draw.text((x - 28, 78), "0", font=font, fill="#a5b6cc")
        draw.text(
            (x - 28, height + 66), str(meta["n_neurons"] - 1), font=font, fill="#a5b6cc"
        )
        draw.text((x, height + 84), "0 ms", font=font, fill="#a5b6cc")
        end_label = f"{_format_duration_ms(meta['n_steps'], meta['dt_seconds'])} ms"
        end_width = draw.textbbox((0, 0), end_label, font=font)[2]
        draw.text(
            (x + width - end_width, height + 84), end_label, font=font, fill="#a5b6cc"
        )
        draw.text(
            (x + width // 2 - 24, height + 84), "time ->", font=font, fill="#a5b6cc"
        )
        dt_text = _format_number(meta["dt_seconds"])
        lines = (
            f"seed={meta['seed']}  dt_seconds={dt_text}  N={meta['n_neurons']}  T={meta['n_steps']}",
            "axon-encoder commit:",
            meta["axon_encoder_git_sha"],
        )
        captions.append("\n".join((title, *lines)))
        draw.multiline_text(
            (x, height + 104), "\n".join(lines), font=font, fill="#a5b6cc", spacing=4
        )
    draw.text(
        (48, height + 160),
        "Neuron ID: top to bottom / white = spike present (not amplitude)",
        font=font,
        fill="#e4edf7",
    )
    draw.text(
        (48, height + 178),
        "Shared input does not imply matched rate mappings; see fixture encoder_config.",
        font=font,
        fill="#a5b6cc",
    )
    if rate_mode == "streaming":
        draw.text(
            (48, height + 196),
            "Rate seed is recorded provenance only; streaming Rate uses no RNG.",
            font=font,
            fill="#a5b6cc",
        )

    image = canvas.resize(
        (canvas.width * 2, canvas.height * 2), Image.Resampling.NEAREST
    )
    description = "\n\n".join(captions)
    image.info["Description"] = description
    if out_path is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Description", description)
        image.save(Path(out_path), format="PNG", pnginfo=pnginfo)
    return image


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
