"""Sparse/dense I/O and schema helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spike_viz.events import SpikeEvents
from spike_viz.io import SpikeIOError, load_dense, load_sparse, sparse_to_dense


def test_load_sparse_roundtrip(tmp_path: Path) -> None:
    t = np.array([0, 2, 2], dtype=np.int64)
    neuron_id = np.array([1, 0, 3], dtype=np.int64)
    amp = np.array([1.0, 0.5, 1.0], dtype=np.float32)
    path = tmp_path / "spikes.npz"
    np.savez(path, t=t, neuron_id=neuron_id, amp=amp)

    events = load_sparse(path)
    assert len(events) == 3
    np.testing.assert_array_equal(events.t, t)
    np.testing.assert_array_equal(events.neuron_id, neuron_id)
    np.testing.assert_array_equal(events.amp, amp)


def test_load_sparse_without_amp(tmp_path: Path) -> None:
    path = tmp_path / "spikes.npz"
    np.savez(
        path,
        t=np.array([0], dtype=np.int64),
        neuron_id=np.array([0], dtype=np.int64),
    )
    events = load_sparse(path)
    assert events.amp is None


def test_load_sparse_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SpikeIOError, match="not found"):
        load_sparse(tmp_path / "nope.npz")


def test_load_sparse_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, t=np.array([0], dtype=np.int64))
    with pytest.raises(SpikeIOError, match="required arrays"):
        load_sparse(path)


def test_sparse_to_dense() -> None:
    events = SpikeEvents(
        t=np.array([0, 1, 1], dtype=np.int64),
        neuron_id=np.array([0, 0, 1], dtype=np.int64),
        amp=np.array([1.0, 1.0, 2.0], dtype=np.float32),
    )
    grid = sparse_to_dense(events, n_steps=2, n_neurons=2)
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 1.0
    assert grid[1, 0] == 1.0
    assert grid[1, 1] == 2.0
    assert grid[0, 1] == 0.0


def test_sparse_to_dense_out_of_range() -> None:
    events = SpikeEvents(
        t=np.array([5], dtype=np.int64),
        neuron_id=np.array([0], dtype=np.int64),
    )
    with pytest.raises(SpikeIOError, match="time out of range"):
        sparse_to_dense(events, n_steps=2, n_neurons=2)


def test_load_dense_npy(tmp_path: Path) -> None:
    arr = np.zeros((4, 3), dtype=np.float32)
    arr[1, 2] = 1.0
    path = tmp_path / "grid.npy"
    np.save(path, arr)
    loaded = load_dense(path)
    np.testing.assert_array_equal(loaded, arr)


def test_load_dense_wrong_rank(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(3))
    with pytest.raises(SpikeIOError, match=r"\[T, N\]"):
        load_dense(path)


def test_load_dense_rejects_unicode(tmp_path: Path) -> None:
    path = tmp_path / "u.npy"
    np.save(path, np.array([["a", "b"], ["c", "d"]], dtype="U1"))
    with pytest.raises(SpikeIOError, match="dtype"):
        load_dense(path)


def test_load_dense_rejects_complex(tmp_path: Path) -> None:
    path = tmp_path / "c.npy"
    np.save(path, np.zeros((2, 2), dtype=np.complex64))
    with pytest.raises(SpikeIOError, match=r"dtype|real"):
        load_dense(path)


def test_fractional_t_rejected(tmp_path: Path) -> None:
    path = tmp_path / "spikes.npz"
    np.savez(
        path,
        t=np.array([1.9], dtype=np.float64),
        neuron_id=np.array([0], dtype=np.int64),
    )
    with pytest.raises(SpikeIOError, match=r"integral|invalid"):
        load_sparse(path)


def test_integral_float_t_accepted(tmp_path: Path) -> None:
    path = tmp_path / "spikes.npz"
    np.savez(
        path,
        t=np.array([1.0], dtype=np.float64),
        neuron_id=np.array([0], dtype=np.int64),
    )
    events = load_sparse(path)
    assert events.t[0] == 1


def test_last_write_wins_is_deterministic() -> None:
    events = SpikeEvents(
        t=np.array([0, 0], dtype=np.int64),
        neuron_id=np.array([0, 0], dtype=np.int64),
        amp=np.array([1.0, 9.0], dtype=np.float32),
    )
    grid = sparse_to_dense(events, n_steps=1, n_neurons=1, accumulate=False)
    assert grid[0, 0] == 9.0
