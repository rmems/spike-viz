"""In-memory sparse spike event schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _as_int64_indices(name: str, values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Validate 1-D integral indices and return int64 without silent truncation."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")

    if arr.dtype.kind == "c" or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real-valued indices")

    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        # Reject fractional floats (e.g. 1.9 would become 1 under bare astype).
        if not np.all(arr == np.trunc(arr)):
            raise ValueError(f"{name} must be integral indices (got non-integer floats)")
        # Safe range check before cast.
        imin = np.iinfo(np.int64).min
        if arr.size and (float(arr.min()) < imin or np.any(arr >= 2**63)):
            raise ValueError(f"{name} values exceed int64 range")
        return arr.astype(np.int64)

    if arr.dtype.kind == "u":
        imax = np.iinfo(np.int64).max
        if arr.size and int(arr.max()) > imax:
            raise ValueError(f"{name} values exceed int64 range")
        return arr.astype(np.int64)

    if arr.dtype.kind in "i":
        return arr.astype(np.int64, copy=False)

    if arr.dtype.kind == "b":
        return arr.astype(np.int64)

    raise ValueError(f"{name} must be integer-valued, got dtype {arr.dtype}")


def _as_float32_amp(values: npt.ArrayLike, n: int) -> npt.NDArray[np.float32]:
    amp = np.asarray(values)
    if amp.ndim != 1 or amp.shape[0] != n:
        raise ValueError("amp must be 1-D and match t length")
    if amp.dtype.kind == "c" or np.issubdtype(amp.dtype, np.complexfloating):
        raise ValueError("amp must be real-valued, not complex")
    if amp.dtype.kind not in "fib":
        raise ValueError(f"amp must be real numeric, got dtype {amp.dtype}")
    out = amp.astype(np.float32, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError("amp must be finite")
    return out


@dataclass(frozen=True, slots=True)
class SpikeEvents:
    """COO-like sparse spikes: parallel arrays of equal length.

    Parameters
    ----------
    t:
        Step indices (int64). Default time unit is **encoder step**, not seconds.
    neuron_id:
        Neuron / channel indices (int64). Maps from axon-encoder ``channel``.
    amp:
        Optional amplitudes (float32). If ``None``, loaders treat presence as 1.0.
        May encode axon-encoder ``polarity`` as 1.0 / 0.0 when present.
    """

    t: npt.NDArray[np.int64]
    neuron_id: npt.NDArray[np.int64]
    amp: npt.NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        t = _as_int64_indices("t", self.t)
        neuron_id = _as_int64_indices("neuron_id", self.neuron_id)
        if t.shape[0] != neuron_id.shape[0]:
            raise ValueError(
                f"t length {t.shape[0]} != neuron_id length {neuron_id.shape[0]}"
            )
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "neuron_id", neuron_id)
        if self.amp is not None:
            object.__setattr__(self, "amp", _as_float32_amp(self.amp, t.shape[0]))

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_events(self) -> int:
        return len(self)
