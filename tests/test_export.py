"""axon-encoder export layout loader + golden fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spike_viz.export import load_axon_export, load_meta
from spike_viz.io import SpikeIOError

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "fixtures" / "axon-encoder" / "rate" / "tiny_synthetic"
FIXTURE_ROOT = REPO / "fixtures" / "axon-encoder"
GENERATED_CASES = sorted(
    p for p in FIXTURE_ROOT.glob("*/*") if (p / "meta.json").is_file()
)


def test_golden_fixture_loads() -> None:
    assert GOLDEN.is_dir(), f"missing golden fixture at {GOLDEN}"
    case = load_axon_export(GOLDEN)
    assert case.meta["encoder"] == "rate"
    assert case.meta["synthetic"] is True
    assert case.meta["schema_version"] == "1.0"
    assert case.meta["n_neurons"] == 4
    assert case.meta["n_steps"] == 8
    assert len(case.events) == 7
    assert case.stimulus_path is None


def test_missing_case_dir(tmp_path: Path) -> None:
    with pytest.raises(SpikeIOError, match="not found"):
        load_axon_export(tmp_path / "missing")


def test_missing_spikes(tmp_path: Path) -> None:
    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 2,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="not found"):
        load_axon_export(tmp_path)


def test_meta_missing_keys(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"encoder": "rate"}), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="missing required keys"):
        load_meta(tmp_path / "meta.json")


def _write_case(tmp_path: Path, meta: dict, **npz_arrays: np.ndarray) -> None:
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(tmp_path / "spikes.npz", **npz_arrays)


def test_geometry_mismatch(tmp_path: Path) -> None:
    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 2,
    }
    _write_case(
        tmp_path,
        meta,
        t=np.array([10], dtype=np.int64),
        neuron_id=np.array([0], dtype=np.int64),
    )
    with pytest.raises(SpikeIOError, match="out of range"):
        load_axon_export(tmp_path)


def test_zero_geometry_rejected_even_if_empty_spikes(tmp_path: Path) -> None:
    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 0,
        "n_steps": 2,
    }
    _write_case(
        tmp_path,
        meta,
        t=np.array([], dtype=np.int64),
        neuron_id=np.array([], dtype=np.int64),
    )
    with pytest.raises(SpikeIOError, match="n_neurons"):
        load_axon_export(tmp_path)


def test_bad_dt_seconds(tmp_path: Path) -> None:
    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.0,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 2,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="dt_seconds"):
        load_meta(tmp_path / "meta.json")


def test_unsupported_schema_version(tmp_path: Path) -> None:
    meta = {
        "schema_version": "banana",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 2,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="unsupported schema_version"):
        load_meta(tmp_path / "meta.json")


def test_meta_wrong_types_raise_spike_io_error(tmp_path: Path) -> None:
    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": "nope",
        "n_neurons": 2,
        "n_steps": 2,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="seed"):
        load_meta(tmp_path / "meta.json")


def test_oversized_dt_seconds_is_spike_io_error(tmp_path: Path) -> None:
    # Integer too large for float conversion → OverflowError must become SpikeIOError.
    huge = int("9" * 400)
    meta = {
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": huge,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 2,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="dt_seconds"):
        load_meta(tmp_path / "meta.json")


def _case_id(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


@pytest.mark.parametrize(
    "case_dir", GENERATED_CASES, ids=[_case_id(p) for p in GENERATED_CASES]
)
def test_every_checked_in_fixture_loads(case_dir: Path) -> None:
    """Every fixture dir loads through load_axon_export, no re-encoding."""
    case = load_axon_export(case_dir)
    meta = case.meta
    assert meta["encoder"] == case_dir.parent.name
    assert case.events.t.dtype == np.int64
    assert case.events.neuron_id.dtype == np.int64
    assert len(case.events.t) <= meta["n_neurons"] * meta["n_steps"]


def test_generated_fixtures_have_provenance() -> None:
    """Real (non-synthetic) fixtures record the full provenance block."""
    generated = [
        p
        for p in GENERATED_CASES
        if json.loads((p / "meta.json").read_text()).get("synthetic") is False
    ]
    assert generated, "no generated (synthetic: false) fixtures found"
    for case_dir in generated:
        meta = json.loads((case_dir / "meta.json").read_text())
        for key in (
            "axon_encoder_git_sha",
            "encoder",
            "seed",
            "dt_seconds",
            "n_neurons",
            "n_steps",
        ):
            assert key in meta, f"{case_dir}: missing provenance key {key}"
        assert meta["axon_encoder_git_sha"], (
            f"{case_dir}: generated fixture has null/empty axon_encoder_git_sha"
        )


def test_generated_fixtures_use_absolute_ticks() -> None:
    """Regression: TickOffset is call-relative — fixtures must export absolute
    ticks. A collapsed export would put every streaming spike at t=0."""
    rate_case = FIXTURE_ROOT / "rate" / "shared_sine_v1"
    if not rate_case.is_dir():
        pytest.skip("generated rate fixture not present")
    case = load_axon_export(rate_case)
    # Rate is INSTANT: absolute tick == call index. Multiple encode_step calls
    # mean later events must land on t > 0.
    assert int(case.events.t.max()) > 0
    assert int(case.events.t.max()) < case.meta["n_steps"]

    latency_case = FIXTURE_ROOT / "latency" / "shared_sine_v1"
    if latency_case.is_dir():
        case = load_axon_export(latency_case)
        # Window model (span = max_latency + 1 = 11): later presentations must
        # export spikes past the first window's range.
        assert int(case.events.t.max()) >= 11
