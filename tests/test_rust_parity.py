"""Rust/Python semantic parity gate for the Rust-first core.

Runs the Rust ``spike-viz`` CLI (``rust/spike-viz-core``) on the checked-in
fixtures and compares its output against this Python reference
implementation. Semantic parity only: identical event lists, identical
grid geometry (N, T, width, height), and final intensities within +/- 1.
Pixel-perfect PNG identity is *not* required (and not asserted).

Skips cleanly when no Rust toolchain is available, the same way CUDA tests
skip without a GPU. The gate is documented in docs/rust-migration.md.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spike_viz import load_axon_export, render_raster, sparse_to_dense
from spike_viz.events import SpikeEvents

REPO = Path(__file__).resolve().parents[1]
SCALE = 8
INTENSITY_TOLERANCE = 1  # semantic parity: +/- 1 intensity level

# tiny_synthetic (schema smoke), shared_sine_v1 (real rate export, the
# checked-in rate fixture), shared_sine_gain_0_v1 (real silent export).
CASES = [
    "rate/tiny_synthetic",
    "rate/shared_sine_v1",
    "rate/shared_sine_gain_0_v1",
]

pytestmark = [
    pytest.mark.rust,
    pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo toolchain not available for the Rust parity gate",
    ),
]


def _render_with_rust(
    case_dir: Path, out_png: Path, report_json: Path, scale: int
) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "spike-viz",
            "--",
            "render",
            str(case_dir),
            "--out",
            str(out_png),
            "--scale",
            str(scale),
            "--report",
            str(report_json),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )


def _python_peak(case) -> float:
    """Peak of the Python render pipeline (zero-amp substitution included)."""
    events = case.events
    if events.amp is not None:
        amp = np.where(events.amp == 0, np.float32(1.0), events.amp)
        events = SpikeEvents(t=events.t, neuron_id=events.neuron_id, amp=amp)
    grid = sparse_to_dense(events, case.meta["n_steps"], case.meta["n_neurons"])
    return float(grid.max()) if grid.size else 0.0


@pytest.mark.parametrize("case_name", CASES)
def test_rust_render_matches_python_semantics(case_name: str, tmp_path: Path) -> None:
    case_dir = REPO / "fixtures" / "axon-encoder" / case_name
    case = load_axon_export(case_dir)

    out_png = tmp_path / "rust.png"
    report_json = tmp_path / "rust.json"
    _render_with_rust(case_dir, out_png, report_json, SCALE)
    report = json.loads(report_json.read_text(encoding="utf-8"))

    # 1. Events: exact list, same load order — no invented or dropped spikes.
    amp = case.events.amp
    expected_events = [
        (
            int(t),
            int(n),
            None if amp is None else float(a),
        )
        for t, n, a in zip(
            case.events.t,
            case.events.neuron_id,
            amp if amp is not None else [None] * len(case.events.t),
        )
    ]
    rust_events = [
        (e["t"], e["neuron_id"], e["amp"]) for e in report["events"]
    ]
    assert rust_events == expected_events

    # 2. Grid geometry: same N, T, and rendered dimensions at the same scale.
    n_steps = int(case.meta["n_steps"])
    n_neurons = int(case.meta["n_neurons"])
    assert report["n_steps"] == n_steps
    assert report["n_neurons"] == n_neurons
    assert (report["width"], report["height"]) == (n_steps * SCALE, n_neurons * SCALE)

    # 3. Intensity rule: same peak semantics (substituted amplitudes,
    #    accumulated duplicates, peak > 0 normalization).
    assert report["peak"] == pytest.approx(_python_peak(case), rel=1e-6, abs=1e-6)

    # 4. Rendered bins: same non-black support set, values within tolerance.
    #    Report bins are unscaled (t, neuron); map the scaled image back.
    image = render_raster(
        case.events, n_steps=n_steps, n_neurons=n_neurons, scale=SCALE
    )
    assert image.size == (n_steps * SCALE, n_neurons * SCALE)
    pixels = np.asarray(image.convert("L"))
    py_bins: dict[tuple[int, int], int] = {}
    for row, col in zip(*np.nonzero(pixels)):
        key = (int(row) // SCALE, int(col) // SCALE)
        py_bins[key] = int(pixels[row, col])
    rust_bins = {
        (int(b["neuron_id"]), int(b["t"])): int(b["value"]) for b in report["bins"]
    }
    assert set(rust_bins) == set(py_bins)
    for key, rust_value in rust_bins.items():
        assert abs(rust_value - py_bins[key]) <= INTENSITY_TOLERANCE

    # 5. The written Rust PNG decodes to the same semantics as its report.
    with Image.open(out_png) as png:
        rust_pixels = np.asarray(png.convert("L"))
    assert rust_pixels.shape == pixels.shape
    diff = np.abs(rust_pixels.astype(np.int16) - pixels.astype(np.int16))
    assert diff.max() <= INTENSITY_TOLERANCE
    assert set(zip(*np.nonzero(rust_pixels))) == set(zip(*np.nonzero(pixels)))

    # 6. Provenance honesty: the report carries the export's metadata.
    assert report["tool"] == "spike-viz-core"
    assert report["encoder"] == case.meta["encoder"]
    assert report["schema_version"] == case.meta["schema_version"]
    assert report["synthetic"] == case.meta.get("synthetic")
    assert report["axon_encoder_git_sha"] == case.meta.get("axon_encoder_git_sha")
    assert report["seed"] == case.meta["seed"]
    assert report["dt_seconds"] == pytest.approx(float(case.meta["dt_seconds"]))


def test_rust_render_silent_export_is_honestly_empty(tmp_path: Path) -> None:
    """The real silent rate-gain export must not gain spikes in Rust."""
    case_dir = REPO / "fixtures" / "axon-encoder" / "rate" / "shared_sine_gain_0_v1"
    out_png = tmp_path / "rust.png"
    report_json = tmp_path / "rust.json"
    _render_with_rust(case_dir, out_png, report_json, SCALE)
    report = json.loads(report_json.read_text(encoding="utf-8"))

    assert report["n_events"] == 0
    assert report["bins"] == []
    assert report["peak"] == 0.0
    with Image.open(out_png) as png:
        pixels = np.asarray(png.convert("L"))
    assert pixels.shape == (8 * SCALE, 64 * SCALE)
    assert np.count_nonzero(pixels) == 0
