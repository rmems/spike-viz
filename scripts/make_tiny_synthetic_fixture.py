#!/usr/bin/env python3
"""Write fixtures/axon-encoder/rate/tiny_synthetic golden case (synthetic)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "axon-encoder" / "rate" / "tiny_synthetic"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Deterministic tiny sparse pattern for layout tests (not axon-encoder output).
    t = np.array([0, 1, 1, 3, 5, 5, 7], dtype=np.int64)
    neuron_id = np.array([0, 0, 2, 1, 0, 3, 2], dtype=np.int64)
    amp = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0], dtype=np.float32)

    np.savez(OUT / "spikes.npz", t=t, neuron_id=neuron_id, amp=amp)

    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 4,
        "n_steps": 8,
        "axon_encoder_git_sha": None,
        "synthetic": True,
        "stimulus_notes": "No continuous stimulus; schema layout smoke only.",
        "notes": (
            "Synthetic golden fixture for spike-viz export contract tests. "
            "Replace with axon-encoder export when cargo example lands."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
