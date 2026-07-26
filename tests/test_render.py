"""CPU raster renderer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spike_viz.events import SpikeEvents
from spike_viz.render import render_raster


def test_render_raster_from_sparse_marks_expected_pixels(tmp_path: Path) -> None:
    events = SpikeEvents(
        t=np.array([0, 2, 4], dtype=np.int64),
        neuron_id=np.array([1, 0, 2], dtype=np.int64),
    )
    out_path = tmp_path / "raster.png"

    image = render_raster(events, n_steps=5, n_neurons=3, out_path=out_path)

    assert out_path.is_file()
    assert image.size == (5, 3)  # (width=n_steps, height=n_neurons)

    pixels = np.array(image.convert("L"))
    assert pixels.shape == (3, 5)  # (height, width)
    assert pixels[1, 0] == 255
    assert pixels[0, 2] == 255
    assert pixels[2, 4] == 255
    assert np.count_nonzero(pixels) == 3


def test_render_raster_from_dense_array() -> None:
    dense = np.zeros((4, 2), dtype=np.float32)
    dense[1, 1] = 1.0

    image = render_raster(dense)

    pixels = np.array(image.convert("L"))
    assert pixels.shape == (2, 4)
    assert pixels[1, 1] == 255
    assert np.count_nonzero(pixels) == 1


def test_render_raster_empty_events_is_all_black() -> None:
    events = SpikeEvents(t=np.array([], dtype=np.int64), neuron_id=np.array([], dtype=np.int64))

    image = render_raster(events, n_steps=3, n_neurons=2)

    pixels = np.array(image.convert("L"))
    assert np.count_nonzero(pixels) == 0


def test_render_raster_sparse_requires_grid_size() -> None:
    events = SpikeEvents(t=np.array([0], dtype=np.int64), neuron_id=np.array([0], dtype=np.int64))

    with pytest.raises(ValueError, match="n_steps and n_neurons"):
        render_raster(events)


def test_render_raster_rejects_non_2d_dense_array() -> None:
    with pytest.raises(ValueError, match=r"\[T, N\]"):
        render_raster(np.zeros(3, dtype=np.float32))


def test_render_raster_scale_upscales_pixels() -> None:
    dense = np.zeros((2, 2), dtype=np.float32)
    dense[0, 0] = 1.0

    image = render_raster(dense, scale=3)

    assert image.size == (6, 6)
