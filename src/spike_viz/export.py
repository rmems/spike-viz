"""axon-encoder export layout loader (export-first bridge)."""

from __future__ import annotations

import json
import math
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

# Supported export contract versions (bump when field semantics change).
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})


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


def _require_int(meta: dict[str, Any], key: str, path: Path) -> int:
    if key not in meta:
        raise SpikeIOError(f"{path}: missing required key '{key}'")
    value = meta[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpikeIOError(f"{path}: '{key}' must be an integer")
    return value


def _require_positive_int(meta: dict[str, Any], key: str, path: Path) -> int:
    value = _require_int(meta, key, path)
    if value < 1:
        raise SpikeIOError(f"{path}: '{key}' must be >= 1, got {value}")
    return value


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
    except OSError as exc:
        raise SpikeIOError(f"failed to read meta.json at {p}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SpikeIOError(f"{p}: root must be a JSON object")

    # Presence first — always raise SpikeIOError, never KeyError.
    missing = sorted(REQUIRED_META_KEYS - meta.keys())
    if missing:
        raise SpikeIOError(f"{p}: missing required keys: {missing}")

    for key in ("n_neurons", "n_steps"):
        _require_positive_int(meta, key, p)
    seed = _require_int(meta, "seed", p)
    if seed < 0:
        raise SpikeIOError(f"{p}: 'seed' must be >= 0, got {seed}")

    if "dt_seconds" not in meta:
        raise SpikeIOError(f"{p}: missing required key 'dt_seconds'")
    dt = meta["dt_seconds"]
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise SpikeIOError(f"{p}: 'dt_seconds' must be a number")
    dt_f = float(dt)
    if not math.isfinite(dt_f) or dt_f <= 0.0:
        raise SpikeIOError(f"{p}: 'dt_seconds' must be finite and > 0, got {dt!r}")

    if "encoder" not in meta:
        raise SpikeIOError(f"{p}: missing required key 'encoder'")
    if not isinstance(meta["encoder"], str) or not meta["encoder"]:
        raise SpikeIOError(f"{p}: 'encoder' must be a non-empty string")

    if "schema_version" not in meta:
        raise SpikeIOError(f"{p}: missing required key 'schema_version'")
    if not isinstance(meta["schema_version"], str) or not meta["schema_version"]:
        raise SpikeIOError(f"{p}: 'schema_version' must be a non-empty string")
    if meta["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise SpikeIOError(
            f"{p}: unsupported schema_version {meta['schema_version']!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

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

    # Geometry already validated as >= 1 in load_meta; bounds-check events.
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
