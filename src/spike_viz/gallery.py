"""Provenance-pinned seven-encoder gallery on one shared stimulus (paint only)."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from spike_viz.export import AxonExportCase, load_axon_export
from spike_viz.hero import _format_duration_ms, _format_number
from spike_viz.io import SpikeIOError, load_dense
from spike_viz.render import PathLike

GALLERY_ENCODERS = (
    "rate",
    "poisson",
    "latency",
    "population",
    "temporal",
    "predictive",
    "phase",
)

_TITLES = {
    "rate": "RateEncoder",
    "poisson": "PoissonEncoder",
    "latency": "LatencyEncoder",
    "population": "PopulationEncoder",
    "temporal": "TemporalEncoder",
    "predictive": "PredictiveEncoder",
    "phase": "PhaseEncoder",
}
_ACCENTS = {
    "rate": "#55ddeb",
    "poisson": "#ffc174",
    "latency": "#9fe870",
    "population": "#c4a7ff",
    "temporal": "#ff9cc2",
    "predictive": "#7fb8ff",
    "phase": "#f2e27a",
}
_BACKGROUND = "#0b111b"
_TEXT = "#e4edf7"
_MUTED = "#a5b6cc"
_OUTLINE = "#41516a"
# Reserved for event bins only; no text or decoration uses these colors.
_EVENT = (255, 255, 255)
_NEGATIVE = (255, 64, 208)
# Every T and N must divide the shared panel size so bins stay integer-aligned
# across panels; past these caps the gallery refuses rather than misalign.
_MAX_LCM_WIDTH = 2048
_MAX_LCM_HEIGHT = 1024
_MIN_RASTER_HEIGHT = 96
_MIN_RASTER_WIDTH = 512
_WINDOW_ENCODERS = frozenset({"latency"})
_CAPTION_WIDTH = 340


def _require_pinned(case: AxonExportCase, encoder: str) -> None:
    if case.meta["encoder"] != encoder:
        raise SpikeIOError(f"{case.path}: expected {encoder} encoder")
    if case.meta.get("synthetic") is not False:
        raise SpikeIOError(f"{case.path}: gallery requires synthetic=false")
    sha = case.meta.get("axon_encoder_git_sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise SpikeIOError(f"{case.path}: full axon_encoder_git_sha required")


def _load_stimulus(case: AxonExportCase) -> np.ndarray:
    """Finite ``[T, C]`` stimulus; C is input channels, not necessarily N."""
    if case.stimulus_path is None:
        raise SpikeIOError(f"{case.path}: stimulus.npy required for comparison")
    stimulus = load_dense(case.stimulus_path)
    if (
        stimulus.ndim != 2
        or stimulus.shape[0] != case.meta["n_steps"]
        or stimulus.shape[1] < 1
    ):
        raise SpikeIOError(f"{case.path}: stimulus must have shape [T, channels]")
    if not np.all(np.isfinite(stimulus)):
        raise SpikeIOError(f"{case.path}: stimulus must be finite")
    return stimulus


def _ticks_per_sample(
    case: AxonExportCase, stimulus: np.ndarray, shared: np.ndarray
) -> int:
    """Ticks each shared stimulus sample spans in this export.

    Instant encoders take one tick per sample. Window encoders (latency) hold
    each sample for the whole window, so their stimulus must be exactly the
    shared stimulus repeated row by row; anything else is a different input.
    """
    samples, channels = shared.shape
    if stimulus.shape[1] != channels or stimulus.shape[0] % samples:
        raise SpikeIOError(f"{case.path}: gallery requires the same stimulus")
    ticks = stimulus.shape[0] // samples
    if not np.array_equal(stimulus, np.repeat(shared, ticks, axis=0)):
        raise SpikeIOError(f"{case.path}: gallery requires the same stimulus")
    return ticks


def _mode_line(
    encoder: str, meta: dict, ticks: int, channels: int, rate_mode: str
) -> str:
    """Behavior that a reader needs to interpret the panel honestly."""
    if encoder == "rate":
        return (
            "deterministic streaming, no RNG"
            if rate_mode == "streaming"
            else "stochastic batch"
        )
    if encoder == "poisson":
        return "stochastic, seeded RNG"
    if encoder == "latency":
        return f"time-to-first-spike, {ticks}-tick window per sample"
    if encoder == "population":
        per_channel = meta["n_neurons"] // channels
        if meta["n_neurons"] % channels:
            return f"tuned population, N={meta['n_neurons']} over {channels} channels"
        return f"tuned population, {channels} channels x {per_channel} neurons"
    if encoder == "temporal":
        return "change-driven: input deltas"
    if encoder == "predictive":
        return "signed prediction error, +/-"
    return "phase-of-firing"


def _raster_pixels(case: AxonExportCase, width: int, height: int) -> Image.Image:
    """Event presence, magenta where every event in a bin is negative."""
    meta = case.meta
    events = case.events
    shape = (meta["n_neurons"], meta["n_steps"])
    present = np.zeros(shape, dtype=bool)
    present[events.neuron_id, events.t] = True
    non_negative = np.zeros(shape, dtype=bool)
    if events.amp is None:
        non_negative = present
    else:
        keep = events.amp >= 0
        non_negative[events.neuron_id[keep], events.t[keep]] = True
    pixels = np.zeros((*shape, 3), dtype=np.uint8)
    pixels[present & ~non_negative] = _NEGATIVE
    pixels[non_negative] = _EVENT
    # Nearest-neighbour upsampling only: width/height are never below T/N.
    return Image.fromarray(pixels, mode="RGB").resize(
        (width, height), Image.Resampling.NEAREST
    )


def render_encoder_gallery(
    case_dirs: Mapping[str, PathLike | None],
    *,
    rate_mode: Literal["streaming", "batch"],
    out_path: PathLike | None = None,
) -> Image.Image:
    """Stack one raster per axon-encoder strategy on a shared sample axis.

    ``case_dirs`` must name every encoder in :data:`GALLERY_ENCODERS`. A value
    of ``None`` declares that no export exists yet: that slot is painted as an
    explicit *pending* panel with no raster and no spikes, and the caption
    says so. Every supplied case must be a real export (``synthetic: false``)
    with a full axon-encoder commit and a ``stimulus.npy`` equal to the shared
    stimulus, row-repeated for window encoders such as latency. N and T may
    differ per encoder; every raster spans the same stimulus samples at the
    same width, and N is stretched to a common panel height. Phase exports
    with ``TimeModel::overlapping`` slack (T not a whole multiple of the
    sample count) are not supported yet and raise ``SpikeIOError``. Missing or
    incompatible exports raise ``SpikeIOError``; nothing is encoded.

    ``rate_mode`` is required for the same reason as in
    :func:`spike_viz.render_rate_poisson_hero`: the export does not record it.

    Returns an RGB image with a textual ``Description`` in ``image.info``;
    if ``out_path`` is supplied, also writes a PNG with that metadata.
    """
    if rate_mode not in ("streaming", "batch"):
        raise ValueError("rate_mode must be 'streaming' or 'batch'")
    missing = [e for e in GALLERY_ENCODERS if e not in case_dirs]
    extra = sorted(set(case_dirs) - set(GALLERY_ENCODERS))
    if missing or extra:
        raise ValueError(
            f"case_dirs must have exactly {GALLERY_ENCODERS}; "
            f"missing={missing} unexpected={extra}"
        )

    cases: dict[str, AxonExportCase | None] = {}
    stimuli: dict[str, np.ndarray] = {}
    for encoder in GALLERY_ENCODERS:
        if case_dirs[encoder] is None:
            cases[encoder] = None
            continue
        case = load_axon_export(case_dirs[encoder])
        _require_pinned(case, encoder)
        cases[encoder] = case
        stimuli[encoder] = _load_stimulus(case)
    if not stimuli:
        raise SpikeIOError("gallery requires at least one export")

    # Window encoders hold each sample for several ticks, so only the others
    # can define one tick per sample; a latency-only set would mislabel it.
    per_sample = [s for e, s in stimuli.items() if e not in _WINDOW_ENCODERS]
    if not per_sample:
        raise SpikeIOError(
            "gallery requires a non-window export to define the shared sample axis"
        )
    shared = min(per_sample, key=lambda s: s.shape[0])
    samples, channels = shared.shape
    ticks = {e: _ticks_per_sample(cases[e], s, shared) for e, s in stimuli.items()}

    real = [c for c in cases.values() if c is not None]
    steps = [c.meta["n_steps"] for c in real]
    neurons = [c.meta["n_neurons"] for c in real]
    width = math.lcm(*steps)
    if width > _MAX_LCM_WIDTH:
        raise SpikeIOError(
            f"gallery requires lcm of n_steps <= {_MAX_LCM_WIDTH} for a shared "
            f"sample axis; got {width} from {sorted(set(steps))}"
        )
    unit = math.lcm(*neurons)
    if unit > _MAX_LCM_HEIGHT:
        raise SpikeIOError(
            f"gallery requires lcm of n_neurons <= {_MAX_LCM_HEIGHT} for a shared "
            f"panel height; got {unit} from {sorted(set(neurons))}"
        )
    width *= max(1, math.ceil(_MIN_RASTER_WIDTH / width))
    height = unit * max(1, math.ceil(_MIN_RASTER_HEIGHT / unit))

    left, top, gap = 48, 64, 40
    row_height = 20 + height + 18
    caption_x = left + width + 24
    canvas = Image.new(
        "RGB",
        (
            caption_x + _CAPTION_WIDTH,
            top + len(GALLERY_ENCODERS) * (row_height + gap) + 64,
        ),
        _BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (left, 16),
        f"AXON-ENCODER GALLERY / same stimulus "
        f"({samples} samples x {channels} channels)",
        font=font,
        fill=_TEXT,
    )
    draw.text(
        (left, 34),
        "axon-encoder exports / spike-viz paints / shared sample axis",
        font=font,
        fill=_MUTED,
    )

    captions = []
    for index, encoder in enumerate(GALLERY_ENCODERS):
        case = cases[encoder]
        y = top + index * (row_height + gap)
        raster_y = y + 20
        title = _TITLES[encoder]
        if case is None:
            draw.text(
                (left, y),
                f"{title} (no export: pending)",
                font=font,
                fill=_ACCENTS[encoder],
            )
            # Outline only: no black raster interior and no event colors, so
            # the slot cannot be mistaken for an empty spike raster.
            draw.rectangle(
                (left, raster_y, left + width, raster_y + height), outline=_OUTLINE
            )
            message = "No export in this fixture set. Nothing is painted here."
            message_width = draw.textbbox((0, 0), message, font=font)[2]
            draw.text(
                (left + (width - message_width) // 2, raster_y + height // 2 - 6),
                message,
                font=font,
                fill=_MUTED,
            )
            lines = ("export pending", "no spikes, no provenance to report")
            captions.append(f"{title}: no export supplied (pending); nothing painted")
        else:
            meta = case.meta
            mode = _mode_line(encoder, meta, ticks[encoder], channels, rate_mode)
            draw.text((left, y), f"{title} ({mode})", font=font, fill=_ACCENTS[encoder])
            canvas.paste(_raster_pixels(case, width, height), (left, raster_y))
            draw.rectangle(
                (left - 1, raster_y - 1, left + width, raster_y + height),
                outline=_OUTLINE,
            )
            draw.text((left - 28, raster_y), "0", font=font, fill=_MUTED)
            draw.text(
                (left - 28, raster_y + height - 12),
                str(meta["n_neurons"] - 1),
                font=font,
                fill=_MUTED,
            )
            dt_text = _format_number(meta["dt_seconds"])
            duration = _format_duration_ms(meta["n_steps"], meta["dt_seconds"])
            lines = (
                f"seed={meta['seed']}  dt_seconds={dt_text}",
                f"N={meta['n_neurons']}  T={meta['n_steps']} ({duration} ms)",
                "axon-encoder commit:",
                meta["axon_encoder_git_sha"],
            )
            captions.append("\n".join((f"{title} ({mode})", *lines)))
        draw.text((left, raster_y + height + 4), "sample 0", font=font, fill=_MUTED)
        end_label = f"sample {samples}"
        end_width = draw.textbbox((0, 0), end_label, font=font)[2]
        draw.text(
            (left + width - end_width, raster_y + height + 4),
            end_label,
            font=font,
            fill=_MUTED,
        )
        draw.multiline_text(
            (caption_x, raster_y), "\n".join(lines), font=font, fill=_MUTED, spacing=4
        )

    footer_y = canvas.height - 56
    draw.text(
        (left, footer_y),
        "Neuron ID top to bottom per panel / white = spike present (not amplitude) "
        "/ magenta = negative polarity",
        font=font,
        fill=_TEXT,
    )
    draw.text(
        (left, footer_y + 18),
        "Shared input does not imply matched rates or N; dt_seconds is a display "
        "convention for dimensionless encoders.",
        font=font,
        fill=_MUTED,
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
