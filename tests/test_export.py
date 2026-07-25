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
