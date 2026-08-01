"""CPU raster rendering: sparse/dense spike grids -> PNG.

Axis convention: **time is horizontal** (columns), **neuron is vertical**
(rows). A ``[T, N]`` dense grid is transposed to an ``[N, T]`` pixel array
before rendering so image width == n_steps and image height == n_neurons.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from spike_viz.events import SpikeEvents
from spike_viz.io import sparse_to_dense

PathLike = str | os.PathLike[str]


def render_raster(
    data: SpikeEvents | npt.NDArray[np.floating | np.bool_ | np.integer],
    *,
    n_steps: int | None = None,
    n_neurons: int | None = None,
    out_path: PathLike | None = None,
    scale: int = 1,
) -> Image.Image:
    """Render a spike raster to a dark-theme grayscale PNG-ready image.

    Parameters
    ----------
    data:
        Either a :class:`SpikeEvents` (sparse COO; requires ``n_steps`` and
        ``n_neurons``) or a dense ``[T, N]`` array (from :func:`load_dense`
        or :func:`sparse_to_dense`).
    n_steps, n_neurons:
        Grid size, required only when ``data`` is a :class:`SpikeEvents`.
    out_path:
        If given, save the rendered image as a PNG at this path.
    scale:
        Integer upscale factor for each pixel (nearest-neighbor), useful
        since raw grids are often small. Default 1 (no upscaling).

    Returns
    -------
    PIL.Image.Image
        RGB image, width == ``T * scale`` and height == ``N * scale``,
        where ``T`` and ``N`` are the time and neuron grid dimensions
        (``n_steps`` and ``n_neurons`` for sparse input). Background is
        black; spike intensity is rendered as white, scaled so the
        brightest bin in the grid maps to full white.
    """
    if isinstance(data, SpikeEvents):
        if n_steps is None or n_neurons is None:
            raise ValueError(
                "n_steps and n_neurons are required when data is SpikeEvents"
            )
        grid = sparse_to_dense(data, n_steps, n_neurons, accumulate=True)
    else:
        grid = np.asarray(data)
        if grid.ndim != 2 or 0 in grid.shape:
            raise ValueError(
                f"dense data must be a non-empty 2-D [T, N] array, got shape {grid.shape}"
            )
        if grid.dtype.kind not in "fbiu":
            raise ValueError(
                f"dense data must be float, bool, or integer, got dtype {grid.dtype}"
            )

    if (
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, np.integer))
        or scale < 1
    ):
        raise ValueError(f"scale must be an integer >= 1, got {scale!r}")

    # Suppress the overflow warning so a too-large value becomes inf and is
    # caught below with a clear ValueError instead of a RuntimeWarning.
    with np.errstate(over="ignore"):
        frame = grid.T.astype(np.float32)  # [N, T]: rows=neuron, cols=time
    if not np.all(np.isfinite(frame)):
        raise ValueError("data contains non-finite or out-of-range values")

    peak = float(frame.max()) if frame.size else 0.0
    intensity = frame / peak if peak > 0 else frame
    pixels = np.clip(intensity, 0.0, 1.0)
    pixels = np.round(pixels * 255.0).astype(np.uint8)

    if scale != 1:
        pixels = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)

    image = Image.fromarray(pixels, mode="L").convert("RGB")

    if out_path is not None:
        image.save(Path(out_path), format="PNG")

    return image
