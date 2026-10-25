#!/usr/bin/env python3
"""Write fixtures/axon-encoder/rate/tiny_fractional_amp golden case (synthetic).

Covers the parity path the all-representable-amp fixtures cannot: fractional
float32 amplitudes (0.1, 0.7) whose exact f64 widening must survive the
Rust/Python JSON parity comparison, plus a zero-amplitude (polarity-false)
event that the renderer substitutes with unit amplitude.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "axon-encoder" / "rate" / "tiny_fractional_amp"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Deterministic tiny sparse pattern for parity tests (not axon-encoder
    # output). Every event lands on a distinct (t, neuron) bin.
    t = np.array([0, 1, 2, 3], dtype=np.int64)
    neuron_id = np.array([0, 0, 1, 1], dtype=np.int64)
    amp = np.array([0.1, 0.7, 0.0, 1.0], dtype=np.float32)

    np.savez(OUT / "spikes.npz", t=t, neuron_id=neuron_id, amp=amp)

    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 4,
        "axon_encoder_git_sha": None,
        "synthetic": True,
        "stimulus_notes": "No continuous stimulus; schema layout smoke only.",
        "notes": (
            "Synthetic fixture with fractional float32 amplitudes and a "
            "zero-amplitude (polarity-false) event; locks the exact f64 "
            "widening used by the Rust/Python parity report."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
