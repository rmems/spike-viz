"""Rate vs Poisson hero: fair axes, honest provenance, no invented data."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import spike_viz

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "axon-encoder"


def test_hero_saves_shared_scale_rasters_and_provenance(tmp_path: Path) -> None:
    # Independent panel scaling, transposition, or swapped panels would change
    # the event positions from the checked-in Rust exports.
    out = tmp_path / "hero.png"
    image = spike_viz.render_rate_poisson_hero(
        FIXTURES / "rate/shared_sine_v1",
        FIXTURES / "poisson/shared_sine_v1",
        rate_mode="streaming",
        out_path=out,
    )
    with Image.open(out) as saved:
        assert saved.format == "PNG"
        assert saved.size == image.size
        caption = saved.info["Description"]
    assert "RateEncoder (deterministic streaming)" in caption
    assert "PoissonEncoder (stochastic)" in caption
    assert "seed=1592590337" in caption
    assert "dt_seconds=0.001" in caption
    assert "N=8" in caption and "T=64" in caption
    assert caption.count("becb40d0c8722710677dabbf66d420a9c77f2eda") == 2
    # Compare all observed spike bins, not decorative text or incidental layout.
    # Locate the black raster interiors in each half without fixing coordinates.
    sizes = []
    for encoder, half in zip(("rate", "poisson"), np.split(np.array(image), 2, axis=1)):
        y, x = np.where(np.all(half == 0, axis=2))
        raster = Image.fromarray(half[y.min() : y.max() + 1, x.min() : x.max() + 1])
        sizes.append(raster.size)
        pixels = np.array(raster.resize((64, 8), Image.Resampling.NEAREST))
        with np.load(FIXTURES / encoder / "shared_sine_v1/spikes.npz") as spikes:
            expected = np.zeros((8, 64), dtype=bool)
            expected[spikes["neuron_id"], spikes["t"]] = True
        np.testing.assert_array_equal(np.any(pixels > 0, axis=2), expected)
    assert sizes[0] == sizes[1]


def test_hero_does_not_drop_spikes_in_larger_exports(tmp_path: Path) -> None:
    # A fixed-width nearest-neighbor downsample loses the spike at step 3.
    cases = []
    for encoder, step, neuron in (("rate", 3, 2), ("poisson", 509, 7)):
        case = tmp_path / encoder
        shutil.copytree(FIXTURES / encoder / "shared_sine_v1", case)
        meta_path = case / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["n_steps"] = 512
        meta_path.write_text(json.dumps(meta))
        stimulus_path = case / "stimulus.npy"
        np.save(stimulus_path, np.tile(np.load(stimulus_path), (8, 1)))
        np.savez(case / "spikes.npz", t=[step], neuron_id=[neuron])
        cases.append(case)
    image = spike_viz.render_rate_poisson_hero(*cases, rate_mode="streaming")
    for half, step, neuron in zip(
        np.split(np.array(image), 2, axis=1), (3, 509), (2, 7)
    ):
        y, x = np.where(np.all(half == 0, axis=2))
        raster = Image.fromarray(half[y.min() : y.max() + 1, x.min() : x.max() + 1])
        pixels = np.array(raster.resize((512, 8), Image.Resampling.NEAREST))
        assert np.any(pixels[neuron, step] > 0)
        assert np.count_nonzero(np.any(pixels > 0, axis=2)) == 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"dt_seconds": 0.002}, "dt_seconds"),
        ({"n_steps": 65}, "n_steps"),
        ({"n_neurons": 9}, "n_neurons"),
        ({"synthetic": True}, "synthetic"),
        ({"axon_encoder_git_sha": None}, "axon_encoder_git_sha"),
        ({"encoder": "latency"}, "poisson"),
    ],
)
def test_hero_rejects_misleading_comparisons(
    tmp_path: Path, change: dict, message: str
) -> None:
    case = tmp_path / "poisson"
    shutil.copytree(FIXTURES / "poisson/shared_sine_v1", case)
    meta_path = case / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(change)
    meta_path.write_text(json.dumps(meta))
    out = tmp_path / "hero.png"
    with pytest.raises(spike_viz.SpikeIOError, match=message):
        spike_viz.render_rate_poisson_hero(
            FIXTURES / "rate/shared_sine_v1",
            case,
            rate_mode="streaming",
            out_path=out,
        )
    assert not out.exists()


def test_hero_requires_same_stimulus_and_no_fallback(tmp_path: Path) -> None:
    case = tmp_path / "poisson"
    shutil.copytree(FIXTURES / "poisson/shared_sine_v1", case)
    stimulus_path = case / "stimulus.npy"
    stimulus = np.load(stimulus_path)
    stimulus[12, 3] += 0.01
    np.save(stimulus_path, stimulus)
    with pytest.raises(spike_viz.SpikeIOError, match="same stimulus"):
        spike_viz.render_rate_poisson_hero(
            FIXTURES / "rate/shared_sine_v1",
            case,
            rate_mode="streaming",
        )
    stimulus_path.unlink()
    with pytest.raises(spike_viz.SpikeIOError, match="stimulus.npy"):
        spike_viz.render_rate_poisson_hero(
            FIXTURES / "rate/shared_sine_v1",
            case,
            rate_mode="streaming",
        )
    (case / "spikes.npz").unlink()
    with pytest.raises(spike_viz.SpikeIOError, match="spike file not found"):
        spike_viz.render_rate_poisson_hero(
            FIXTURES / "rate/shared_sine_v1",
            case,
            rate_mode="streaming",
        )


def test_hero_rate_mode_must_be_explicit_and_valid() -> None:
    args = (FIXTURES / "rate/shared_sine_v1", FIXTURES / "poisson/shared_sine_v1")
    with pytest.raises(TypeError, match="rate_mode"):
        spike_viz.render_rate_poisson_hero(*args)
    with pytest.raises(ValueError, match="rate_mode"):
        spike_viz.render_rate_poisson_hero(*args, rate_mode="unknown")
    image = spike_viz.render_rate_poisson_hero(*args, rate_mode="batch")
    assert "RateEncoder (stochastic batch)" in image.info["Description"]
