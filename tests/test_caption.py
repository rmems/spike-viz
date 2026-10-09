"""Provenance captions use export metadata, never inferred provenance."""

import json
from pathlib import Path

import pytest

import spike_viz
from spike_viz import SpikeIOError


@pytest.fixture
def meta_path(tmp_path: Path) -> Path:
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "encoder": "poisson",
                "dt_seconds": 0.0025,
                "seed": 42,
                "n_neurons": 3,
                "n_steps": 17,
                "axon_encoder_git_sha": "1234567890abcdef1234567890abcdef12345678",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_caption_complete_metadata_and_sidecar(meta_path: Path) -> None:
    # Catches swapped N/T, rounded dt, truncated SHA, and order-dependent output.
    expected = (
        'encoder="poisson" | dt_seconds=0.0025 | seed=42 | N=3 | T=17 | '
        'axon_encoder_git_sha="1234567890abcdef1234567890abcdef12345678"'
    )
    out_path = meta_path.with_suffix(".txt")
    assert spike_viz.provenance_caption(meta_path, out_path=out_path) == expected
    assert out_path.read_bytes() == (expected + "\n").encode("utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(
        json.dumps(dict(reversed(list(meta.items())))), encoding="utf-8"
    )
    assert spike_viz.provenance_caption(meta_path) == expected


@pytest.mark.parametrize("sha", [None, "", "   ", "absent"])
def test_caption_unknown_commit(meta_path: Path, sha: str | None) -> None:
    # No replacement hash may be inferred from this checkout or the environment.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if sha == "absent":
        del meta["axon_encoder_git_sha"]
    else:
        meta["axon_encoder_git_sha"] = sha
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert spike_viz.provenance_caption(meta_path) == (
        'encoder="poisson" | dt_seconds=0.0025 | seed=42 | N=3 | T=17 | '
        "axon_encoder_git_sha=unknown"
    )


def test_caption_marks_synthetic_data(meta_path: Path) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["synthetic"] = True
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert spike_viz.provenance_caption(meta_path).endswith(" | synthetic=true")


@pytest.mark.parametrize("sha", [123, False, []])
def test_caption_rejects_invalid_commit_type(meta_path: Path, sha: object) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["axon_encoder_git_sha"] = sha
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SpikeIOError, match="axon_encoder_git_sha"):
        spike_viz.provenance_caption(meta_path)


def test_caption_invalid_metadata_does_not_overwrite_sidecar(meta_path: Path) -> None:
    # Missing required provenance must fail, not produce a plausible caption.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["seed"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out_path = meta_path.with_suffix(".txt")
    out_path.write_text("existing caption\n", encoding="utf-8")
    with pytest.raises(SpikeIOError, match="seed"):
        spike_viz.provenance_caption(meta_path, out_path=out_path)
    assert out_path.read_text(encoding="utf-8") == "existing caption\n"


def test_caption_missing_metadata_fails(tmp_path: Path) -> None:
    with pytest.raises(SpikeIOError, match="not found"):
        spike_viz.provenance_caption(tmp_path / "missing.json")
