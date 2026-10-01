"""Neon bloom + tone-map: dense spike grids -> RGB arrays / PNG.

Axis convention matches :func:`spike_viz.render.render_raster`: input is a
``[T, N]`` dense grid, output image is ``[H=N, W=T]`` (time horizontal).
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image

from spike_viz.events import SpikeEvents
from spike_viz.io import sparse_to_dense

PathLike = str | os.PathLike[str]
Colormap = str | npt.NDArray[np.uint8 | np.floating] | Callable[..., npt.NDArray]
_MAX_SIGMA = 128.0


def _control_lut(stops: list[tuple[float, tuple[float, float, float]]]) -> npt.NDArray[np.uint8]:
    xs = np.array([s for s, _ in stops], dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.float64)
    for ch in range(3):
        ys = np.array([c[ch] for _, c in stops], dtype=np.float64)
        lut[:, ch] = np.interp(np.linspace(0.0, 1.0, 256), xs, ys)
    return np.round(np.clip(lut, 0.0, 1.0) * 255.0).astype(np.uint8)


_BUILTIN_COLORMAPS: dict[str, npt.NDArray[np.uint8]] = {
    "inferno": _control_lut(
        [
            (0.0, (0.0, 0.0, 0.0)),
            (0.2, (0.16, 0.04, 0.24)),
            (0.4, (0.39, 0.07, 0.55)),
            (0.6, (0.68, 0.18, 0.38)),
            (0.8, (0.89, 0.45, 0.18)),
            (0.9, (0.98, 0.75, 0.34)),
            (1.0, (0.99, 0.99, 0.64)),
        ]
    ),
    "magma": _control_lut(
        [
            (0.0, (0.0, 0.0, 0.0)),
            (0.25, (0.19, 0.07, 0.33)),
            (0.5, (0.46, 0.14, 0.53)),
            (0.75, (0.75, 0.32, 0.42)),
            (1.0, (0.99, 0.99, 0.79)),
        ]
    ),
    "plasma": _control_lut(
        [
            (0.0, (0.05, 0.03, 0.53)),
            (0.35, (0.49, 0.02, 0.65)),
            (0.65, (0.80, 0.28, 0.45)),
            (1.0, (0.94, 0.98, 0.60)),
        ]
    ),
    "viridis": _control_lut(
        [
            (0.0, (0.27, 0.00, 0.33)),
            (0.35, (0.13, 0.35, 0.52)),
            (0.65, (0.15, 0.69, 0.42)),
            (1.0, (0.99, 0.91, 0.14)),
        ]
    ),
    "hot": _control_lut(
        [
            (0.0, (0.0, 0.0, 0.0)),
            (0.35, (0.80, 0.00, 0.00)),
            (0.65, (1.00, 0.80, 0.00)),
            (1.0, (1.00, 1.00, 1.00)),
        ]
    ),
    "grayscale": _control_lut([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]),
}
_BUILTIN_COLORMAPS["gray"] = _BUILTIN_COLORMAPS["grayscale"]
_BUILTIN_COLORMAPS["grey"] = _BUILTIN_COLORMAPS["grayscale"]


def _resolve_lut(colormap: Colormap) -> npt.NDArray[np.uint8] | Callable[..., npt.NDArray]:
    if callable(colormap):
        return colormap
    if isinstance(colormap, str):
        key = colormap.lower()
        if key not in _BUILTIN_COLORMAPS:
            raise ValueError(
                f"unknown colormap {colormap!r}; "
                f"expected one of {sorted(_BUILTIN_COLORMAPS)} or a LUT array/callable"
            )
        return _BUILTIN_COLORMAPS[key]
    arr = np.asarray(colormap)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] not in (3, 4):
        raise ValueError(
            f"colormap LUT must have shape (M, 3) or (M, 4) with M >= 2, got {arr.shape}"
        )
    if arr.shape[1] == 4:
        arr = arr[:, :3]
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError("colormap LUT contains non-finite values")
        arr = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype.kind in "iu":
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"colormap LUT must be numeric, got dtype {arr.dtype}")
    if arr.shape[0] != 256:
        xs = np.linspace(0.0, 1.0, 256)
        src_x = np.linspace(0.0, 1.0, arr.shape[0])
        out = np.zeros((256, 3), dtype=np.float64)
        for ch in range(3):
            out[:, ch] = np.interp(xs, src_x, arr[:, ch].astype(np.float64))
        arr = np.round(out).astype(np.uint8)
    return arr


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        requested = device
    else:
        try:
            requested = torch.device(str(device))
        except Exception as exc:
            raise ValueError(f"invalid device {device!r}: {exc}") from exc
    if requested.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"CUDA requested ({requested}) but not available; falling back to CPU",
            UserWarning,
            stacklevel=3,
        )
        return torch.device("cpu")
    if requested.type not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    return requested


def _gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    k = 2 * math.ceil(3.0 * sigma) + 1
    half = k // 2
    x = torch.arange(k, dtype=torch.float32, device=device) - float(half)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _separable_blur(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0.0:
        return frame
    device = frame.device
    kernel = _gaussian_kernel1d(sigma, device)
    k = int(kernel.numel())
    pad = k // 2
    img = frame.unsqueeze(0).unsqueeze(0)
    kh = kernel.view(1, 1, 1, k)
    kv = kernel.view(1, 1, k, 1)
    img = F.conv2d(img, kh, padding=(0, pad))
    img = F.conv2d(img, kv, padding=(pad, 0))
    return img.squeeze(0).squeeze(0)


def bloom_raster(
    grid: SpikeEvents | npt.NDArray[np.floating | np.bool_ | np.integer] | torch.Tensor,
    *,
    n_steps: int | None = None,
    n_neurons: int | None = None,
    sigma: float = 2.0,
    intensity: float = 1.0,
    gamma: float = 2.2,
    colormap: Colormap = "inferno",
    device: str | torch.device = "cpu",
    out_path: PathLike | None = None,
) -> npt.NDArray[np.uint8]:
    """Apply neon bloom + tone-map to a dense ``[T, N]`` spike grid.

    Pipeline (identical on CPU/CUDA): ``[T, N]`` -> transpose to ``[N, T]``
    image frame -> separable Gaussian blur -> peak-normalize -> ``log1p``
    tone-map -> gamma -> colormap LUT -> ``uint8`` RGB.

    Parameters
    ----------
    grid:
        Dense ``[T, N]`` spike grid (e.g. from :func:`sparse_to_dense`) or
        sparse :class:`~spike_viz.events.SpikeEvents`. Sparse input requires
        ``n_steps`` and ``n_neurons`` and preserves zero-polarity events.
        Never synthesized here — pass loader output.
    n_steps, n_neurons:
        Grid dimensions required only for sparse ``SpikeEvents`` input.
    sigma:
        Gaussian blur std in pixels; ``0`` disables blur. Must be finite and
        between 0 and 128 inclusive.
    intensity:
        Tone-map strength; must be finite, > 0. Larger values lift mid-tones.
    gamma:
        Display gamma; must be finite, > 0. Applied as ``x ** (1 / gamma)``.
    colormap:
        Name (``inferno`` default, ``magma``, ``plasma``, ``viridis``,
        ``hot``, ``grayscale``/``gray``/``grey``), a ``(M, 3|4)`` LUT array,
        or a callable ``f(x: float [H, W] in [0, 1]) -> uint8/float [H, W, 3]``.
    device:
        ``'cpu'`` (default) or ``'cuda'``. CUDA without a GPU falls back to
        CPU with a warning — CPU always works.
    out_path:
        If given, save the RGB array as a PNG.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB array with shape ``(N, T, 3)``.
    """
    for name, value in (("sigma", sigma), ("intensity", intensity), ("gamma", gamma)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    sigma = float(sigma)
    intensity = float(intensity)
    gamma = float(gamma)
    if not 0.0 <= sigma <= _MAX_SIGMA:
        raise ValueError(f"sigma must be between 0 and {_MAX_SIGMA:g}, got {sigma!r}")
    if intensity <= 0.0:
        raise ValueError(f"intensity must be > 0, got {intensity!r}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}")

    compute = _resolve_device(device)
    lut = _resolve_lut(colormap)

    if isinstance(grid, SpikeEvents):
        if n_steps is None or n_neurons is None:
            raise ValueError("n_steps and n_neurons are required when grid is SpikeEvents")
        # axon-encoder exports polarity=False as amp=0.0. It is still an event,
        # so make it visible before dense conversion discards that information.
        if grid.amp is not None:
            amp = np.where(grid.amp == 0, np.float32(1.0), grid.amp)
            grid = SpikeEvents(t=grid.t, neuron_id=grid.neuron_id, amp=amp)
        with np.errstate(over="ignore"):
            grid = sparse_to_dense(grid, n_steps, n_neurons, accumulate=True)

    if isinstance(grid, torch.Tensor):
        if grid.ndim != 2 or 0 in grid.shape:
            raise ValueError(f"grid must be a non-empty 2-D [T, N] array, got shape {tuple(grid.shape)}")
        if grid.is_complex() or grid.is_quantized:
            raise ValueError(f"grid must be float, bool, or integer, got dtype {grid.dtype}")
        with torch.no_grad():
            if grid.is_floating_point():
                if not bool(torch.isfinite(grid).all()):
                    raise ValueError("grid contains non-finite values")
            if bool((grid < 0).any()):
                raise ValueError("grid must be non-negative")
            frame = grid.detach().to(device=compute, dtype=torch.float32).t().contiguous()
            if not bool(torch.isfinite(frame).all()):
                raise ValueError("grid contains non-finite or out-of-range values")
    else:
        arr = np.asarray(grid)
        if arr.ndim != 2 or 0 in arr.shape:
            raise ValueError(f"grid must be a non-empty 2-D [T, N] array, got shape {arr.shape}")
        if arr.dtype.kind not in "fbiu":
            raise ValueError(f"grid must be float, bool, or integer, got dtype {arr.dtype}")
        with np.errstate(over="ignore"):
            frame_np = arr.T.astype(np.float32)
        if not np.all(np.isfinite(frame_np)):
            raise ValueError("grid contains non-finite or out-of-range values")
        if bool((frame_np < 0).any()):
            raise ValueError("grid must be non-negative")
        frame = torch.from_numpy(np.ascontiguousarray(frame_np)).to(device=compute, dtype=torch.float32)

    with torch.no_grad():
        # A subnormal sigma rounds to zero in the float32 kernel. Its discrete
        # limiting blur is the identity, so avoid a 0/0 kernel center.
        if 0.0 < sigma < np.finfo(np.float32).tiny:
            sigma = 0.0
        blurred = _separable_blur(frame, sigma)
        peak = float(blurred.max().item()) if blurred.numel() else 0.0
        if peak > 0:
            normed = blurred / peak
        else:
            normed = blurred
        if intensity < np.finfo(np.float32).tiny:
            # lim_{a -> 0} log1p(a * x) / log1p(a) = x
            tonemapped = normed
        else:
            denom = math.log1p(intensity)
            tonemapped = torch.log1p(torch.clamp(normed * intensity, min=0.0)) / denom
        corrected = torch.pow(torch.clamp(tonemapped, 0.0, 1.0), 1.0 / gamma)
        heat = corrected.detach().to("cpu").numpy().astype(np.float64)

    heat = np.clip(heat, 0.0, 1.0)
    if callable(lut):
        rgb = np.asarray(lut(heat))
        if rgb.shape != (*heat.shape, 3):
            raise ValueError(f"colormap callable must return shape {(*heat.shape, 3)}, got {rgb.shape}")
        if rgb.dtype.kind == "f":
            if not np.all(np.isfinite(rgb)):
                raise ValueError("colormap callable returned non-finite values")
            rgb = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif rgb.dtype.kind in "iu":
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"colormap callable must return numeric array, got {rgb.dtype}")
    else:
        assert isinstance(lut, np.ndarray)
        indices = np.round(heat * 255.0).astype(np.int64)
        rgb = lut[indices]

    if out_path is not None:
        Image.fromarray(rgb, mode="RGB").save(Path(out_path), format="PNG")
    return rgb
