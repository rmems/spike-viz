"""Provenance captions use export metadata, never inferred provenance."""

import json
from pathlib import Path

import pytest

import spike_viz
from spike_viz import SpikeIOError

REPO = Path(__file__).resolve().parents[1]
RATE_FIXTURE_META = (
    REPO / "fixtures" / "axon-encoder" / "rate" / "shared_sine_v1" / "meta.json"
)


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


def test_caption_collapses_parent_segments(meta_path: Path, tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    caption = spike_viz.provenance_caption(
        meta_path, out_path=nested / ".." / "caption.txt"
    )
    written = tmp_path / "caption.txt"
    assert written.read_text(encoding="utf-8") == caption + "\n"
    assert not (nested / "caption.txt").exists()


def test_caption_missing_parent_raises_before_metadata(tmp_path: Path) -> None:
    # Path checks happen before metadata validation, so a missing parent
    # fails even when meta.json is also invalid.
    meta = tmp_path / "meta.json"
    meta.write_text("{}", encoding="utf-8")
    out = tmp_path / "missing" / "caption.txt"
    with pytest.raises(OSError, match="Parent directory does not exist"):
        spike_viz.provenance_caption(meta, out_path=out)
    assert not out.exists()


def test_caption_pinned_rate_fixture() -> None:
    assert spike_viz.provenance_caption(RATE_FIXTURE_META) == (
        'encoder="rate" | dt_seconds=0.001 | seed=1592590337 | N=8 | T=64 | '
        'axon_encoder_git_sha="908a34797c8aaff0a57e0ea4167aa00f3780c940"'
    )


def test_caption_refuses_sidecar_that_resolves_to_metadata(meta_path: Path) -> None:
    original = meta_path.read_bytes()
    alias = meta_path.parent / ".." / meta_path.parent.name / meta_path.name
    with pytest.raises(SpikeIOError, match="metadata file"):
        spike_viz.provenance_caption(meta_path, out_path=alias)
    assert meta_path.read_bytes() == original


def test_caption_refuses_symlink_to_metadata(meta_path: Path, tmp_path: Path) -> None:
    original = meta_path.read_bytes()
    link = tmp_path / "meta-link.json"
    link.symlink_to(meta_path)
    with pytest.raises(SpikeIOError, match="metadata file"):
        spike_viz.provenance_caption(meta_path, out_path=link)
    assert meta_path.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "value", "escaped"),
    [
        ("encoder", "rate\u2028poisson", "\\u2028"),
        ("encoder", "rate\u2029poisson", "\\u2029"),
        ("encoder", "rate\ud800", "\\ud800"),
        ("axon_encoder_git_sha", "abc\u2028def", "\\u2028"),
        ("axon_encoder_git_sha", "abc\u2029def", "\\u2029"),
        ("axon_encoder_git_sha", "abc\ud800", "\\ud800"),
    ],
)
def test_caption_escapes_line_separators_and_surrogates(
    meta_path: Path, field: str, value: str, escaped: str
) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[field] = value
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = meta_path.with_suffix(".txt")
    caption = spike_viz.provenance_caption(meta_path, out_path=out)
    assert "\u2028" not in caption
    assert "\u2029" not in caption
    assert "\ud800" not in caption
    assert "\n" not in caption
    assert escaped in caption
    raw = out.read_bytes()
    assert raw.count(b"\n") == 1
    raw.decode("utf-8")


@pytest.mark.parametrize("value", ["true", 1, None])
def test_caption_rejects_non_boolean_synthetic(meta_path: Path, value: object) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["synthetic"] = value
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = meta_path.with_suffix(".txt")
    out.write_text("keep\n", encoding="utf-8")
    with pytest.raises(SpikeIOError, match="synthetic"):
        spike_viz.provenance_caption(meta_path, out_path=out)
    assert out.read_text(encoding="utf-8") == "keep\n"
