"""axon-encoder export layout loader (export-first bridge)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spike_viz.events import SpikeEvents
from spike_viz.io import SpikeIOError, load_sparse

PathLike = str | Path

# Required meta.json keys for a contract-complete case directory.
REQUIRED_META_KEYS = frozenset(
    {
        "encoder",
        "dt_seconds",
        "seed",
        "n_neurons",
        "n_steps",
        "schema_version",
    }
)


@dataclass(frozen=True, slots=True)
class AxonExportCase:
    """One fixture / export case under the axon-encoder layout."""

    path: Path
    events: SpikeEvents
    meta: dict[str, Any]
    spikes_path: Path
    meta_path: Path
    stimulus_path: Path | None


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def load_meta(path: PathLike) -> dict[str, Any]:
    """Load and validate ``meta.json`` for an export case."""
    p = _as_path(path)
    if not p.is_file():
        raise SpikeIOError(f"meta.json not found: {p}")
    try:
        with p.open(encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise SpikeIOError(f"invalid JSON in {p}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SpikeIOError(f"{p}: root must be a JSON object")

    missing = sorted(REQUIRED_META_KEYS - meta.keys())
    if missing:
        raise SpikeIOError(f"{p}: missing required keys: {missing}")

    for key in ("n_neurons", "n_steps", "seed"):
        if not isinstance(meta[key], int) or isinstance(meta[key], bool):
            raise SpikeIOError(f"{p}: '{key}' must be an integer")
    if not isinstance(meta["dt_seconds"], (int, float)) or isinstance(
        meta["dt_seconds"], bool
    ):
        raise SpikeIOError(f"{p}: 'dt_seconds' must be a number")
    if not isinstance(meta["encoder"], str) or not meta["encoder"]:
        raise SpikeIOError(f"{p}: 'encoder' must be a non-empty string")
    if not isinstance(meta["schema_version"], str) or not meta["schema_version"]:
        raise SpikeIOError(f"{p}: 'schema_version' must be a non-empty string")

    return meta


def load_axon_export(case_dir: PathLike) -> AxonExportCase:
    """Load a case directory: ``spikes.npz`` + ``meta.json`` [+ optional stimulus].

    Layout (see ``docs/axon-encoder-export.md``)::

        <case>/
          spikes.npz
          meta.json
          stimulus.npy   # optional

    Does **not** invent spikes if files are missing — raises ``SpikeIOError``.
    """
    root = _as_path(case_dir)
    if not root.is_dir():
        raise SpikeIOError(f"export case directory not found: {root}")

    spikes_path = root / "spikes.npz"
    meta_path = root / "meta.json"
    stimulus_path = root / "stimulus.npy"
    if not stimulus_path.is_file():
        stimulus_path = None

    meta = load_meta(meta_path)
    events = load_sparse(spikes_path)

    # Bounds check against declared geometry (fail loud).
    n_steps = int(meta["n_steps"])
    n_neurons = int(meta["n_neurons"])
    if len(events) > 0:
        if int(events.t.max()) >= n_steps or int(events.t.min()) < 0:
            raise SpikeIOError(
                f"{spikes_path}: t out of range for n_steps={n_steps} "
                f"(min={int(events.t.min())}, max={int(events.t.max())})"
            )
        if int(events.neuron_id.max()) >= n_neurons or int(events.neuron_id.min()) < 0:
            raise SpikeIOError(
                f"{spikes_path}: neuron_id out of range for n_neurons={n_neurons} "
                f"(min={int(events.neuron_id.min())}, max={int(events.neuron_id.max())})"
            )

    return AxonExportCase(
        path=root,
        events=events,
        meta=meta,
        spikes_path=spikes_path,
        meta_path=meta_path,
        stimulus_path=stimulus_path,
    )
