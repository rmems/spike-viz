"""Load sparse/dense spike arrays from disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from spike_viz.events import SpikeEvents

PathLike = str | Path

_DENSE_ALLOWED_KINDS = frozenset({"f", "b", "i", "u"})  # float, bool, integer


class SpikeIOError(Exception):
    """Missing or invalid spike data — fail loud, never invent spikes."""


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def load_sparse(path: PathLike) -> SpikeEvents:
    """Load sparse COO spikes from ``.npz``.

    Required arrays: ``t``, ``neuron_id``.
    Optional: ``amp``.

    Raises
    ------
    SpikeIOError
        If the file is missing or required arrays are absent / inconsistent.
    """
    p = _as_path(path)
    if not p.is_file():
        raise SpikeIOError(f"sparse spike file not found: {p}")
    if p.suffix.lower() != ".npz":
        raise SpikeIOError(f"load_sparse expects a .npz file, got: {p}")

    try:
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            if "t" not in keys or "neuron_id" not in keys:
                raise SpikeIOError(
                    f"{p}: required arrays 't' and 'neuron_id' missing; "
                    f"found {sorted(keys)}"
                )
            t = z["t"]
            neuron_id = z["neuron_id"]
            amp = z["amp"] if "amp" in keys else None
    except SpikeIOError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface corrupt archives clearly
        raise SpikeIOError(f"failed to read sparse spikes from {p}: {exc}") from exc

    try:
        return SpikeEvents(t=t, neuron_id=neuron_id, amp=amp)
    except ValueError as exc:
        raise SpikeIOError(f"{p}: invalid sparse arrays: {exc}") from exc


def load_dense(path: PathLike) -> npt.NDArray[np.floating | np.bool_ | np.integer]:
    """Load a dense ``[T, N]`` spike grid from ``.npy`` or ``.npz``.

    For ``.npz``, the array must be named ``dense`` (or be the sole array).
    Allowed dtypes: floating, boolean, or integer (integers kept as-is; floats/bool OK).
    Complex, unicode, and object arrays are rejected.
    """
    p = _as_path(path)
    if not p.is_file():
        raise SpikeIOError(f"dense spike file not found: {p}")

    try:
        if p.suffix.lower() == ".npy":
            arr = np.load(p, allow_pickle=False)
        elif p.suffix.lower() == ".npz":
            with np.load(p, allow_pickle=False) as z:
                if "dense" in z.files:
                    arr = z["dense"]
                elif len(z.files) == 1:
                    arr = z[z.files[0]]
                else:
                    raise SpikeIOError(
                        f"{p}: expected array 'dense' (or a single array); "
                        f"found {sorted(z.files)}"
                    )
        else:
            raise SpikeIOError(f"load_dense expects .npy or .npz, got: {p}")
    except SpikeIOError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SpikeIOError(f"failed to read dense spikes from {p}: {exc}") from exc

    if arr.ndim != 2:
        raise SpikeIOError(
            f"{p}: dense grid must have shape [T, N], got {arr.shape}"
        )
    if arr.dtype.kind == "c" or np.issubdtype(arr.dtype, np.complexfloating):
        raise SpikeIOError(f"{p}: dense grid must be real-valued, got {arr.dtype}")
    if arr.dtype.kind not in _DENSE_ALLOWED_KINDS:
        raise SpikeIOError(
            f"{p}: dense grid dtype must be float, bool, or integer; "
            f"got {arr.dtype}"
        )
    return arr


def sparse_to_dense(
    events: SpikeEvents,
    n_steps: int,
    n_neurons: int,
    *,
    accumulate: bool = True,
) -> npt.NDArray[np.float32]:
    """Raster sparse events into a dense ``[T, N]`` float32 grid.

    Parameters
    ----------
    accumulate:
        If True, add amplitudes into bins (default). If False, last write wins
        for duplicate (t, neuron) pairs (applied in event order, deterministic).
    """
    if n_steps < 1 or n_neurons < 1:
        raise ValueError("n_steps and n_neurons must be >= 1")
    if len(events) == 0:
        return np.zeros((n_steps, n_neurons), dtype=np.float32)

    t = events.t
    n = events.neuron_id
    if np.any(t < 0) or np.any(t >= n_steps):
        raise SpikeIOError(
            f"event time out of range [0, {n_steps}): "
            f"min={int(t.min())} max={int(t.max())}"
        )
    if np.any(n < 0) or np.any(n >= n_neurons):
        raise SpikeIOError(
            f"neuron_id out of range [0, {n_neurons}): "
            f"min={int(n.min())} max={int(n.max())}"
        )

    amp = (
        events.amp
        if events.amp is not None
        else np.ones(len(events), dtype=np.float32)
    )
    amp_f = amp.astype(np.float32, copy=False)
    grid = np.zeros((n_steps, n_neurons), dtype=np.float32)
    if accumulate:
        np.add.at(grid, (t, n), amp_f)
    else:
        # Explicit loop: NumPy advanced indexing does not guarantee last-write
        # order when indices repeat.
        for ti, ni, ai in zip(t, n, amp_f, strict=True):
            grid[int(ti), int(ni)] = float(ai)
    return grid
