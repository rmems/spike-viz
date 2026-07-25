"""In-memory sparse spike event schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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
        t = np.asarray(self.t)
        neuron_id = np.asarray(self.neuron_id)
        if t.ndim != 1 or neuron_id.ndim != 1:
            raise ValueError("t and neuron_id must be 1-D arrays")
        if t.shape[0] != neuron_id.shape[0]:
            raise ValueError(
                f"t length {t.shape[0]} != neuron_id length {neuron_id.shape[0]}"
            )
        object.__setattr__(self, "t", t.astype(np.int64, copy=False))
        object.__setattr__(self, "neuron_id", neuron_id.astype(np.int64, copy=False))
        if self.amp is not None:
            amp = np.asarray(self.amp)
            if amp.ndim != 1 or amp.shape[0] != t.shape[0]:
                raise ValueError("amp must be 1-D and match t length")
            object.__setattr__(self, "amp", amp.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_events(self) -> int:
        return len(self)
