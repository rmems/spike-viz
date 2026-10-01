"""Neon bloom + tone-map tests (issue #11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from spike_viz import bloom_raster, load_axon_export
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
    with pytest.raises(ValueError, match="intensity"):
        bloom_raster(grid, intensity=0.0)
    with pytest.raises(ValueError, match="gamma"):
        bloom_raster(grid, gamma=0.0)
    with pytest.raises(ValueError, match="colormap"):
        bloom_raster(grid, colormap="not-a-map")
    with pytest.raises(ValueError, match=r"\[T, N\]"):
        bloom_raster(np.zeros(4, dtype=np.float32))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bloom_cuda_matches_cpu() -> None:
    grid = np.zeros((16, 8), dtype=np.float32)
    grid[3, 2] = 1.0
    grid[10, 5] = 3.0

    cpu = bloom_raster(grid, sigma=1.5, device="cpu").astype(np.int16)
    gpu = bloom_raster(grid, sigma=1.5, device="cuda").astype(np.int16)

    assert cpu.shape == gpu.shape
    assert np.abs(cpu - gpu).max() <= 2


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
