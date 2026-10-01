#!/usr/bin/env python3
"""Pack an axon-encoder export staging directory into a contract case.

Reads the `.npy` files written by axon-encoder's `export_spike_viz` example
(`t.npy`, `neuron_id.npy`, optional `amp.npy`, optional `stimulus.npy`,
`meta.json`), assembles `spikes.npz`, copies `meta.json` and `stimulus.npy`,
and validates the result by loading it through `load_axon_export`.

This script only assembles and validates the container. No spike generation
or encoder math lives here — encoding truth stays in axon-encoder.

Usage:
    python scripts/pack_axon_export.py <staging_dir> <case_dir>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spike_viz.export import load_axon_export  # noqa: E402
from spike_viz.io import SpikeIOError  # noqa: E402


def _load_npy(path: Path, *, required: bool) -> np.ndarray | None:
    if not path.is_file():
        if required:
            raise SpikeIOError(f"{path}: required staging file missing")
        return None
    return np.load(path, allow_pickle=False)


def _int_index(arr: np.ndarray, name: str, staging_dir: Path) -> np.ndarray:
    if not np.issubdtype(arr.dtype, np.integer) or not np.all(arr >= 0):
        raise SpikeIOError(f"{staging_dir}: {name}.npy must be non-negative integers")
    return np.asarray(arr, dtype=np.int64)


def _validated_amp(
    amp: np.ndarray, t_shape: tuple[int, ...], staging_dir: Path
) -> np.ndarray:
    if not np.issubdtype(amp.dtype, np.floating) and not np.issubdtype(
        amp.dtype, np.integer
    ):
        raise SpikeIOError(f"{staging_dir}: amp.npy must be numeric")
    amp = np.asarray(amp, dtype=np.float32)
    if not np.all(np.isfinite(amp)):
        raise SpikeIOError(f"{staging_dir}: amp.npy contains non-finite values")
    if amp.shape != t_shape:
        raise SpikeIOError(
            f"{staging_dir}: amp {amp.shape} length mismatch vs t {t_shape}"
        )
    return amp


def pack(staging_dir: Path, case_dir: Path) -> None:
    t = _load_npy(staging_dir / "t.npy", required=True)
    neuron_id = _load_npy(staging_dir / "neuron_id.npy", required=True)
    amp = _load_npy(staging_dir / "amp.npy", required=False)
    stimulus = _load_npy(staging_dir / "stimulus.npy", required=False)
    meta_path = staging_dir / "meta.json"
    if not meta_path.is_file():
        raise SpikeIOError(f"{meta_path}: required staging file missing")

    t = _int_index(t, "t", staging_dir)
    neuron_id = _int_index(neuron_id, "neuron_id", staging_dir)
    if t.shape != neuron_id.shape:
        raise SpikeIOError(
            f"{staging_dir}: t {t.shape} and neuron_id {neuron_id.shape} length mismatch"
        )
    arrays: dict[str, np.ndarray] = {"t": t, "neuron_id": neuron_id}
    if amp is not None:
        arrays["amp"] = _validated_amp(amp, t.shape, staging_dir)

    # Rebuild the case directory so stale files from a previous run can't
    # masquerade as outputs of this one (mixed provenance).
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    np.savez(case_dir / "spikes.npz", **arrays)
    shutil.copyfile(meta_path, case_dir / "meta.json")
    if stimulus is not None:
        np.save(case_dir / "stimulus.npy", np.asarray(stimulus, dtype=np.float32))

    # Round-trip validation through the real loader.
    load_axon_export(case_dir)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    pack(Path(argv[1]), Path(argv[2]))
    print(f"packed {argv[1]} -> {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
