"""Provenance-pinned Rate vs Poisson hero still (paint only)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from spike_viz.export import load_axon_export
from spike_viz.io import SpikeIOError, load_dense
from spike_viz.render import PathLike, render_raster


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
        stimuli.append(stimulus)

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
        duration_ms = meta["n_steps"] * meta["dt_seconds"] * 1000
        end_label = f"{duration_ms:g} ms"
        end_width = draw.textbbox((0, 0), end_label, font=font)[2]
        draw.text(
            (x + width - end_width, height + 84), end_label, font=font, fill="#a5b6cc"
        )
        draw.text(
            (x + width // 2 - 24, height + 84), "time ->", font=font, fill="#a5b6cc"
        )
        lines = (
            f"seed={meta['seed']}  dt_seconds={meta['dt_seconds']:g}  N={meta['n_neurons']}  T={meta['n_steps']}",
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
