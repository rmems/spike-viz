"""Seven-encoder gallery: shared stimulus, real exports only, explicit pending."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import spike_viz
from spike_viz.gallery import GALLERY_ENCODERS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "axon-encoder"
EXPORTED = GALLERY_ENCODERS[:-1]
WHITE = np.array([255, 255, 255])
MAGENTA = np.array([255, 64, 208])


def _dirs(**overrides):
    dirs = {e: FIXTURES / e / "shared_sine_v1" for e in EXPORTED}
    dirs["phase"] = None
    dirs.update(overrides)
    return dirs


def _rasters(image: Image.Image) -> list[np.ndarray]:
    """Raster interiors top to bottom; only rasters use black/white/magenta."""
    pixels = np.asarray(image)
    mask = (
        np.all(pixels == 0, axis=2)
        | np.all(pixels == WHITE, axis=2)
        | np.all(pixels == MAGENTA, axis=2)
    )
    rows = np.flatnonzero(mask.any(axis=1))
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    out = []
    for run in runs:
        block = mask[run[0] : run[-1] + 1]
        cols = np.flatnonzero(block.any(axis=0))
        out.append(pixels[run[0] : run[-1] + 1, cols[0] : cols[-1] + 1])
    return out


def _copy(tmp_path: Path, source: str, **meta_changes) -> Path:
    case = tmp_path / source
    shutil.copytree(FIXTURES / source / "shared_sine_v1", case)
    meta_path = case / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(meta_changes)
    meta_path.write_text(json.dumps(meta))
    return case


def test_gallery_paints_every_exported_bin_and_provenance(tmp_path: Path) -> None:
    out = tmp_path / "gallery.png"
    image = spike_viz.render_encoder_gallery(
        _dirs(), rate_mode="streaming", out_path=out
    )
    with Image.open(out) as saved:
        assert saved.format == "PNG"
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(image))
        caption = saved.info["Description"]

    rasters = _rasters(image)
    # The pending phase slot must not look like an (empty) raster.
    assert len(rasters) == len(EXPORTED)
    # One shared sample-axis width and panel height across encoders.
    assert len({r.shape for r in rasters}) == 1
    for encoder, raster in zip(EXPORTED, rasters):
        case = spike_viz.load_axon_export(FIXTURES / encoder / "shared_sine_v1")
        n, t = case.meta["n_neurons"], case.meta["n_steps"]
        bins = np.asarray(
            Image.fromarray(raster).resize((t, n), Image.Resampling.NEAREST)
        )
        expected = np.zeros((n, t), dtype=bool)
        expected[case.events.neuron_id, case.events.t] = True
        np.testing.assert_array_equal(np.any(bins > 0, axis=2), expected)
        negative = np.zeros((n, t), dtype=bool)
        neg = case.events.amp < 0
        negative[case.events.neuron_id[neg], case.events.t[neg]] = True
        np.testing.assert_array_equal(np.all(bins == MAGENTA, axis=2), negative)
        assert negative.any() == (encoder == "predictive")

    sha = json.loads((FIXTURES / "rate/shared_sine_v1/meta.json").read_text())[
        "axon_encoder_git_sha"
    ]
    assert caption.count(sha) == len(EXPORTED)
    for text in (
        "RateEncoder (deterministic streaming, no RNG)",
        "PoissonEncoder (stochastic, seeded RNG)",
        "LatencyEncoder (time-to-first-spike, 11-tick window per sample)",
        "PopulationEncoder (tuned population, 8 channels x 4 neurons)",
        "N=32  T=64 (64 ms)",
        "N=8  T=704 (704 ms)",
        "PhaseEncoder: no export supplied (pending); nothing painted",
    ):
        assert text in caption


def test_gallery_paints_a_supplied_phase_export(tmp_path: Path) -> None:
    # Stand-in geometry only: a real phase export replaces the pending slot.
    phase = _copy(tmp_path, "rate", encoder="phase")
    image = spike_viz.render_encoder_gallery(_dirs(phase=phase), rate_mode="streaming")
    assert len(_rasters(image)) == len(GALLERY_ENCODERS)
    assert "pending" not in image.info["Description"]


@pytest.mark.parametrize(
    "encoder, change, message",
    [
        ("poisson", {"synthetic": True}, "synthetic"),
        ("latency", {"axon_encoder_git_sha": None}, "axon_encoder_git_sha"),
        ("temporal", {"encoder": "predictive"}, "temporal"),
    ],
)
def test_gallery_rejects_unpinned_or_mislabeled_exports(
    tmp_path: Path, encoder: str, change: dict, message: str
) -> None:
    out = tmp_path / "gallery.png"
    case = _copy(tmp_path, encoder, **change)
    with pytest.raises(spike_viz.SpikeIOError, match=message):
        spike_viz.render_encoder_gallery(
            _dirs(**{encoder: case}), rate_mode="streaming", out_path=out
        )
    assert not out.exists()


@pytest.mark.parametrize("encoder", ["population", "latency"])
def test_gallery_requires_the_same_stimulus(tmp_path: Path, encoder: str) -> None:
    case = _copy(tmp_path, encoder)
    stimulus_path = case / "stimulus.npy"
    stimulus = np.load(stimulus_path)
    stimulus[5, 3] += 0.01
    np.save(stimulus_path, stimulus)
    with pytest.raises(spike_viz.SpikeIOError, match="same stimulus"):
        spike_viz.render_encoder_gallery(
            _dirs(**{encoder: case}), rate_mode="streaming"
        )
    stimulus_path.unlink()
    with pytest.raises(spike_viz.SpikeIOError, match="stimulus.npy"):
        spike_viz.render_encoder_gallery(
            _dirs(**{encoder: case}), rate_mode="streaming"
        )
    (case / "spikes.npz").unlink()
    with pytest.raises(spike_viz.SpikeIOError, match="spike file not found"):
        spike_viz.render_encoder_gallery(
            _dirs(**{encoder: case}), rate_mode="streaming"
        )


def test_gallery_requires_every_encoder_named_explicitly() -> None:
    dirs = _dirs()
    del dirs["phase"]
    with pytest.raises(ValueError, match="phase"):
        spike_viz.render_encoder_gallery(dirs, rate_mode="streaming")
    with pytest.raises(ValueError, match="bogus"):
        spike_viz.render_encoder_gallery(
            _dirs(bogus=FIXTURES / "rate/shared_sine_v1"), rate_mode="streaming"
        )
    with pytest.raises(spike_viz.SpikeIOError, match="at least one export"):
        spike_viz.render_encoder_gallery(
            dict.fromkeys(GALLERY_ENCODERS), rate_mode="streaming"
        )


def test_gallery_rate_mode_must_be_explicit_and_valid() -> None:
    with pytest.raises(TypeError, match="rate_mode"):
        spike_viz.render_encoder_gallery(_dirs())
    with pytest.raises(ValueError, match="rate_mode"):
        spike_viz.render_encoder_gallery(_dirs(), rate_mode="unknown")
    image = spike_viz.render_encoder_gallery(_dirs(), rate_mode="batch")
    assert "RateEncoder (stochastic batch)" in image.info["Description"]
