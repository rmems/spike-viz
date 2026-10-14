"""Provenance-pinned, same-stimulus rate gain comparisons (paint only)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from spike_viz.events import SpikeEvents
from spike_viz.export import AxonExportCase
from spike_viz.gain_captions import PANEL_MARGIN, display_captions
from spike_viz.render import PathLike, render_raster

_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_PATCH_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHARED_META_KEYS = (
    "encoder",
    "encoder_config",
    "dt_seconds",
    "seed",
    "n_neurons",
    "n_steps",
    "axon_encoder_git_sha",
    "schema_version",
    "axon_encoder_version",
    "exporter_patch_sha256",
)
_IDENTITY_GAIN_KEYS = ("threshold_scale", "sensitivity_scale", "latency_scale")
_BACKGROUND = (8, 12, 24)


def _require_hex(value: object, pattern: re.Pattern[str], message: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(message)


def _require_real_pinned_rate(meta: dict[str, Any]) -> None:
    if meta.get("synthetic") is not False or meta.get("encoder") != "rate":
        raise ValueError("gain triptych requires real rate encoder exports")
    _require_hex(
        meta.get("axon_encoder_git_sha"),
        _GIT_SHA,
        "gain triptych requires a full axon-encoder commit",
    )
    _require_hex(
        meta.get("exporter_patch_sha256"),
        _PATCH_SHA256,
        "gain triptych requires the exporter patch SHA-256",
    )


def _require_same_fields(meta: dict[str, Any], reference: dict[str, Any]) -> None:
    for key in _SHARED_META_KEYS:
        if key not in meta or meta[key] != reference.get(key):
            raise ValueError(f"gain cases must have identical {key}")


def _is_real_number(value: object) -> bool:
    """Python or NumPy reals.

    Booleans compare equal to 0 and 1, so they are not reals here.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _matches_python_float(rate: object, rate_f: float) -> bool:
    """True when ``rate`` is the same number as the narrowed float.

    NumPy integers compare equal to a float they do not round-trip to, because
    that comparison narrows the integer first. Python ``int`` does not.
    """
    if isinstance(rate, np.integer):
        return int(rate) == rate_f
    return rate == rate_f


def _as_finite_nonnegative(rate: object) -> float:
    if not _is_real_number(rate):
        raise ValueError("firing-rate gain must be finite and non-negative")
    try:
        rate_f = float(rate)
    except OverflowError as exc:
        raise ValueError("firing-rate gain must be finite and non-negative") from exc
    if (
        not math.isfinite(rate_f)
        or rate_f < 0
        or not _matches_python_float(rate, rate_f)
    ):
        raise ValueError("firing-rate gain must be finite and non-negative")
    return rate_f


def _require_gain(meta: dict[str, Any]) -> float:
    gains = meta.get("encoding_gains")
    if not isinstance(gains, dict):
        raise ValueError("gain cases must document identity non-rate gains")
    for key in _IDENTITY_GAIN_KEYS:
        value = gains.get(key)
        if not _is_real_number(value) or value != 1.0:
            raise ValueError("gain cases must document identity non-rate gains")
    return _as_finite_nonnegative(gains.get("firing_rate_scale"))


def _require_modulators(meta: dict[str, Any], reference: dict[str, Any]) -> None:
    if "modulators" not in meta:
        raise ValueError("modulators must document a summary object or null")
    mods = meta["modulators"]
    if mods is not None and not isinstance(mods, dict):
        raise ValueError("modulators must document a summary object or null")
    if mods != reference["modulators"]:
        raise ValueError("gain cases must have identical modulators")


def _require_stimulus(
    case: AxonExportCase, stimulus: np.ndarray | None
) -> np.ndarray:
    if case.stimulus_path is None:
        raise ValueError("gain comparison requires exported stimulus.npy")
    current = np.load(case.stimulus_path, allow_pickle=False)
    meta = case.meta
    if current.shape != (meta["n_steps"], meta["n_neurons"]):
        raise ValueError("rate stimulus must have shape [T, N]")
    if stimulus is not None and not np.array_equal(stimulus, current):
        raise ValueError("gain cases must have identical stimulus arrays")
    return current


def _require_gain_order(rates: list[float], silence: AxonExportCase) -> None:
    if rates[0] != 0 or rates[1] != 1 or rates[2] <= 1:
        raise ValueError("expected gains: silence=0, normal=1, elevated>1")
    if len(silence.events):
        raise ValueError("zero-gain silence export must contain no events")


def _validate_gain_cases(cases: tuple[AxonExportCase, ...]) -> list[float]:
    """Reject comparisons that could imply a gain effect from unrelated data."""
    reference = cases[0].meta
    stimulus = None
    rates = []
    for case in cases:
        meta = case.meta
        _require_real_pinned_rate(meta)
        _require_same_fields(meta, reference)
        rates.append(_require_gain(meta))
        _require_modulators(meta, reference)
        stimulus = _require_stimulus(case, stimulus)
    _require_gain_order(rates, cases[0])
    return rates


def _require_display_scale(scale: object) -> int:
    if (
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, np.integer))
        or scale < 1
    ):
        raise ValueError("scale must be an integer >= 1")
    return int(scale)


def render_gain_triptych(
    silence: AxonExportCase,
    normal: AxonExportCase,
    elevated: AxonExportCase,
    out_path: PathLike | None = None,
    scale: int = 2,
) -> Image.Image:
    """Render three real, same-stimulus rate exports and their provenance.

    Cases must share encoder config, dt, seed, N, T, a full upstream commit,
    and the exporter patch SHA-256. They must include identical ``stimulus.npy``
    arrays and identical ``modulators`` (null for direct gains, or one shared
    summary object), and document numeric ``encoding_gains`` (identity except
    firing-rate scales 0, 1, >1). Invalid comparisons raise ValueError.

    Cyan bins show event presence, independent of amplitude/polarity; duplicate
    events share a bin. No gain scaling, resampling, jitter or bloom is applied
    to the spikes. Time is horizontal, neuron is vertical, with identical axes.
    Captions are visible and also stored in PNG ``Description`` metadata.
    ``scale`` is a positive integer nearest-neighbor enlargement, including
    NumPy integers. CPU only; fonts use Pillow's built-in default, with no
    system font dependency.
    """
    cases = (silence, normal, elevated)
    rates = _validate_gain_cases(cases)
    scale = _require_display_scale(scale)

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
    captions = []
    for label, rate, case in zip(("Silence", "Normal", "Elevated"), rates, cases):
        mods = case.meta["modulators"]
        summary = "none (direct gains)" if mods is None else json.dumps(mods, sort_keys=True)
        captions.append(
            [
                label,
                f"firing_rate_scale={rate:g}",
                f"modulators: {summary}",
                f"{len(case.events)} exported events",
            ]
        )
    wrapped, panel_w = display_captions(
        captions, font, max(240, raster_w + 2 * PANEL_MARGIN)
    )
    raster_y = 30 + 16 * max(map(len, wrapped))
    footer_y = raster_y + raster_h + 42
    provenance = [
        (
            f"encoder={meta['encoder']} | dt={meta['dt_seconds']:g} s | seed={meta['seed']} | "
            f"N={meta['n_neurons']} | T={meta['n_steps']}"
        ),
        f"axon-encoder commit: {meta['axon_encoder_git_sha']}",
        "Same stimulus / config; binary event occupancy; no Python encoding.",
    ]
    image = Image.new("RGB", (3 * panel_w, footer_y + 64), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index, (raster, caption) in enumerate(zip(rasters, wrapped)):
        x = index * panel_w + PANEL_MARGIN
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
        (PANEL_MARGIN, footer_y),
        "\n".join(provenance),
        font=font,
        fill=(180, 195, 215),
        spacing=6,
    )
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    description = "\n\n".join("\n".join(caption) for caption in captions)
    description += "\n\n" + "\n".join(provenance)
    image.info["Description"] = description
    if out_path is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Description", description)
        image.save(Path(out_path), format="PNG", pnginfo=pnginfo)
    return image
