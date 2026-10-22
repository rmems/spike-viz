"""Neon bloom + tone-map tests (issues #11 and #15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from spike_viz import bloom_raster, load_axon_export
from spike_viz.events import SpikeEvents
from spike_viz.io import sparse_to_dense


def test_bloom_single_spike_glows_and_keeps_shape() -> None:
    grid = np.zeros((9, 7), dtype=np.float32)
    grid[4, 3] = 1.0

    rgb = bloom_raster(grid, sigma=1.5)

    assert rgb.shape == (7, 9, 3)
    assert rgb.dtype == np.uint8
    center = rgb[3, 4].astype(int)
    assert center.max() > 0
    assert rgb[3, 5].astype(int).max() > 0  # bloom spreads to neighbors
    assert rgb[0, 0].astype(int).max() < center.max()  # glow falls off with distance
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 2


def test_bloom_sigma_zero_is_sharp() -> None:
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[2, 2] = 1.0

    rgb = bloom_raster(grid, sigma=0.0)

    assert rgb.shape == (5, 5, 3)
    assert (rgb[2, 2] == rgb.max(axis=(0, 1))).all()
    assert np.count_nonzero(rgb.any(axis=-1)) == 1


def test_bloom_empty_grid_is_flat_background() -> None:
    grid = np.zeros((4, 3), dtype=np.float32)

    rgb = bloom_raster(grid)

    assert rgb.shape == (3, 4, 3)
    assert (rgb == rgb[0, 0]).all()


def test_bloom_accepts_torch_input_and_matches_numpy() -> None:
    grid = np.zeros((6, 4), dtype=np.float32)
    grid[1, 1] = 1.0
    grid[4, 2] = 2.0

    from_numpy = bloom_raster(grid, sigma=1.0)
    from_torch = bloom_raster(torch.from_numpy(grid), sigma=1.0)

    np.testing.assert_array_equal(from_numpy, from_torch)


def test_bloom_rejects_non_finite_and_negative() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        bloom_raster(np.array([[1.0, np.inf]], dtype=np.float64))
    with pytest.raises(ValueError, match="non-finite"):
        bloom_raster(np.array([[np.nan]], dtype=np.float64))
    with pytest.raises(ValueError, match="non-negative"):
        bloom_raster(np.array([[-1.0]], dtype=np.float32))


def test_bloom_rejects_bad_params_and_colormap() -> None:
    grid = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="sigma"):
        bloom_raster(grid, sigma=-1.0)
    with pytest.raises(ValueError, match="sigma"):
        bloom_raster(grid, sigma=129.0)
    with pytest.raises(ValueError, match="intensity"):
        bloom_raster(grid, intensity=0.0)
    with pytest.raises(ValueError, match="gamma"):
        bloom_raster(grid, gamma=0.0)
    with pytest.raises(ValueError, match="colormap"):
        bloom_raster(grid, colormap="not-a-map")
    with pytest.raises(ValueError, match=r"\[T, N\]"):
        bloom_raster(np.zeros(4, dtype=np.float32))


def test_bloom_handles_torch_cast_overflow_and_bfloat16() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        bloom_raster(torch.tensor([[1e300]], dtype=torch.float64))

    grid = torch.tensor([[0, 1]], dtype=torch.bfloat16)
    assert bloom_raster(grid, sigma=0.0).shape == (2, 1, 3)


def test_bloom_tiny_positive_params_use_stable_limits() -> None:
    grid = np.array([[0.0, 1.0]], dtype=np.float32)
    rgb = bloom_raster(grid, sigma=1e-50, intensity=1e-50)
    assert np.isfinite(rgb).all()
    assert rgb[1, 0].max() > rgb[0, 0].max()


def test_bloom_sparse_preserves_zero_amp_events() -> None:
    events = SpikeEvents(
        t=np.array([0, 1], dtype=np.int64),
        neuron_id=np.array([0, 1], dtype=np.int64),
        amp=np.array([1.0, 0.0], dtype=np.float32),
    )
    rgb = bloom_raster(events, n_steps=2, n_neurons=2, sigma=0.0)
    assert rgb[0, 0].max() > 0
    assert rgb[1, 1].max() > 0


def test_bloom_colormap_variants_and_custom_lut() -> None:
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[2, 2] = 1.0

    inferno = bloom_raster(grid, sigma=1.0, colormap="inferno")
    hot = bloom_raster(grid, sigma=1.0, colormap="hot")
    gray = bloom_raster(grid, sigma=1.0, colormap="grayscale")
    assert inferno.shape == hot.shape == gray.shape == (5, 5, 3)
    assert not np.array_equal(inferno, hot)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 1] = np.arange(256, dtype=np.uint8)
    custom = bloom_raster(grid, sigma=1.0, colormap=lut)
    assert custom[..., 0].max() == 0 and custom[..., 2].max() == 0
    assert custom[..., 1].max() > 0


def test_bloom_cuda_falls_back_with_warning_when_unavailable() -> None:
    grid = np.zeros((5, 4), dtype=np.float32)
    grid[2, 1] = 1.0
    if torch.cuda.is_available():
        pytest.skip("CUDA available; fallback path not exercised")
    with pytest.warns(UserWarning, match="falling back to CPU"):
        rgb = bloom_raster(grid, device="cuda")
    assert rgb.shape == (4, 5, 3)
    assert rgb.max() > 0


# ---------------------------------------------------------------------------
# CPU/CUDA parity (issue #15): CUDA is acceleration, never a different
# aesthetic. The GPU pipeline must reproduce the CPU reference for identical
# inputs and parameters.
#
# float32 convolutions accumulate in a different order on cuDNN/cuBLAS than
# on CPU MKL, so exact cross-device equality is not expected even for a
# correct implementation. Documented tolerance (standard convention for
# cross-device conv/blur parity in PyTorch projects): atol=1e-4, rtol=1e-3 —
# loose enough to absorb accumulation-order drift, tight enough to fail a
# genuinely different algorithm. torch.testing.assert_close reports the max
# absolute difference plus shapes/dtypes on failure, which is the repro
# context the acceptance criteria require.

_PARITY_ATOL = 1e-4
_PARITY_RTOL = 1e-3
# uint8 output after LUT rounding: 1 LSB from heat rounding plus 1 LSB drift.
_UINT8_MAX_DIFF = 2


def _parity_grid() -> np.ndarray:
    """Deterministic small [T, N] grid shared by every CPU/CUDA parity test."""
    grid = np.zeros((24, 12), dtype=np.float32)
    grid[3, 2] = 1.0
    grid[5, 8] = 0.5
    grid[11, 4] = 3.0
    grid[17, 10] = 2.0
    grid[21, 0] = 0.25
    return grid


def _capture_heat_lut(captured: list[np.ndarray]):
    """Colormap callable that records the pre-LUT tone-mapped heat frame."""

    def lut(heat: np.ndarray) -> np.ndarray:
        captured.append(np.array(heat, dtype=np.float64, copy=True))
        return np.zeros((*heat.shape, 3), dtype=np.uint8)

    return lut


def _bloom_heat_on_devices(grid: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:
    """Run bloom on CPU and CUDA with identical params; return (cpu, cuda) heat."""
    cpu_captured: list[np.ndarray] = []
    cuda_captured: list[np.ndarray] = []
    bloom_raster(grid, device="cpu", colormap=_capture_heat_lut(cpu_captured), **params)
    bloom_raster(grid, device="cuda", colormap=_capture_heat_lut(cuda_captured), **params)
    assert len(cpu_captured) == 1 and len(cuda_captured) == 1
    return cpu_captured[0], cuda_captured[0]


def _assert_heat_parity(cpu_heat: np.ndarray, cuda_heat: np.ndarray) -> None:
    torch.testing.assert_close(
        torch.from_numpy(cuda_heat),
        torch.from_numpy(cpu_heat),
        atol=_PARITY_ATOL,
        rtol=_PARITY_RTOL,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("sigma", "intensity", "gamma"),
    [
        (2.0, 1.0, 2.2),  # defaults
        (0.0, 1.0, 2.2),  # blur disabled
        (0.5, 1.0, 2.2),
        (8.0, 1.0, 2.2),  # wide kernel
        (1.5, 2.0, 1.8),  # strong tone-map lift
        (3.0, 0.5, 3.0),  # dim tone-map
    ],
)
def test_bloom_cuda_heat_matches_cpu(sigma: float, intensity: float, gamma: float) -> None:
    grid = _parity_grid()

    cpu_heat, cuda_heat = _bloom_heat_on_devices(
        grid, sigma=sigma, intensity=intensity, gamma=gamma
    )

    _assert_heat_parity(cpu_heat, cuda_heat)


@pytest.mark.cuda
def test_bloom_cuda_uint8_matches_cpu() -> None:
    grid = _parity_grid()
    params: dict = {"sigma": 1.5, "intensity": 2.0, "gamma": 1.8}

    cpu = bloom_raster(grid, device="cpu", **params)
    cuda = bloom_raster(grid, device="cuda", **params)

    assert cpu.dtype == cuda.dtype == np.uint8
    assert cuda.shape == cpu.shape == (12, 24, 3)
    max_diff = int(np.abs(cpu.astype(np.int16) - cuda.astype(np.int16)).max())
    assert max_diff <= _UINT8_MAX_DIFF, (
        f"CPU/CUDA uint8 output differs: max abs diff {max_diff} "
        f"(tolerance {_UINT8_MAX_DIFF}); grid shape {grid.shape}, params {params}. "
        f"Float-level heat parity is checked in test_bloom_cuda_heat_matches_cpu."
    )


@pytest.mark.cuda
def test_bloom_cuda_empty_grid_is_flat_and_identical() -> None:
    grid = np.zeros((13, 7), dtype=np.float32)

    cpu = bloom_raster(grid, sigma=1.5)
    cuda = bloom_raster(grid, sigma=1.5, device="cuda")

    # No activity: blur and tone-map of zeros are exactly zero on both
    # devices, so the frame must be the flat background color on each and
    # must not invent spikes. Inferno maps zero heat to black, so a shared
    # CPU/CUDA regression that lifts the whole frame must not pass either.
    np.testing.assert_array_equal(cpu, cuda)
    assert (cpu == cpu[0, 0]).all()
    np.testing.assert_array_equal(cpu, np.zeros_like(cpu))


@pytest.mark.cuda
def test_bloom_cuda_sparse_fixture_matches_cpu() -> None:
    case = load_axon_export("fixtures/axon-encoder/rate/tiny_synthetic")
    grid = sparse_to_dense(
        case.events, int(case.meta["n_steps"]), int(case.meta["n_neurons"])
    )
    assert 0 < np.count_nonzero(grid) < grid.size  # real sparse activity, not dense

    cpu_heat, cuda_heat = _bloom_heat_on_devices(grid, sigma=1.0)

    _assert_heat_parity(cpu_heat, cuda_heat)


def test_bloom_fixture_produces_nontrivial_png(tmp_path: Path) -> None:
    case = load_axon_export("fixtures/axon-encoder/rate/tiny_synthetic")
    grid = sparse_to_dense(
        case.events, int(case.meta["n_steps"]), int(case.meta["n_neurons"])
    )
    out = tmp_path / "bloom.png"

    rgb = bloom_raster(grid, sigma=1.0, out_path=out)

    assert out.is_file()
    with Image.open(out) as img:
        assert img.size == (rgb.shape[1], rgb.shape[0])
    assert rgb.max() > rgb.min()
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 2
