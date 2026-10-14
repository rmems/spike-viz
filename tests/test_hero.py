"""Hero stills: shared-scale Rate vs Poisson, and real rate-gain comparisons."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

import spike_viz
from spike_viz import hero

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
    # Echo the checked-in pin rather than a second hardcoded SHA, so a
    # fixture re-pin does not leave the caption assertion behind.
    rate_sha = json.loads((FIXTURES / "rate/shared_sine_v1/meta.json").read_text())[
        "axon_encoder_git_sha"
    ]
    poisson_sha = json.loads(
        (FIXTURES / "poisson/shared_sine_v1/meta.json").read_text()
    )["axon_encoder_git_sha"]
    assert rate_sha == poisson_sha
    assert caption.count(rate_sha) == 2
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


def test_hero_round_trips_precise_sampling_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default ``g`` keeps six significant digits: 0.00123456789 paints as
    # 0.00123457, and the shared time endpoint 79.01234496 ms paints as 79.0123.
    dt = 0.00123456789
    cases = []
    for encoder in ("rate", "poisson"):
        case = tmp_path / encoder
        shutil.copytree(FIXTURES / encoder / "shared_sine_v1", case)
        meta_path = case / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["dt_seconds"] = dt
        meta_path.write_text(json.dumps(meta))
        cases.append(case)

    drawn: list[str] = []
    real_draw = hero.ImageDraw.Draw

    def factory(image: Image.Image):
        delegate = real_draw(image)

        class _Proxy:
            def text(self, xy, text, *args, **kwargs):
                drawn.append(str(text))
                return delegate.text(xy, text, *args, **kwargs)

            def multiline_text(self, xy, text, *args, **kwargs):
                drawn.append(str(text))
                return delegate.multiline_text(xy, text, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(delegate, name)

        return _Proxy()

    monkeypatch.setattr(hero.ImageDraw, "Draw", factory)
    image = spike_viz.render_rate_poisson_hero(*cases, rate_mode="streaming")
    painted = "\n".join(drawn)
    description = image.info["Description"]
    assert description.count("dt_seconds=0.00123456789") == 2
    assert "dt_seconds=0.00123457" not in description
    assert painted.count("79.01234496 ms") == 2
    assert "79.0123 ms" not in painted

    # Binary n_steps * dt * 1000 is 1.9200000000000002 for dt_seconds=0.00003.
    drawn.clear()
    for case in cases:
        meta_path = case / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["dt_seconds"] = 0.00003
        meta_path.write_text(json.dumps(meta))
    spike_viz.render_rate_poisson_hero(*cases, rate_mode="streaming")
    noisy = "\n".join(drawn)
    assert noisy.count("1.92 ms") == 2
    assert "1.9200000000000002" not in noisy


def test_hero_rate_mode_must_be_explicit_and_valid() -> None:
    args = (FIXTURES / "rate/shared_sine_v1", FIXTURES / "poisson/shared_sine_v1")
    with pytest.raises(TypeError, match="rate_mode"):
        spike_viz.render_rate_poisson_hero(*args)
    with pytest.raises(ValueError, match="rate_mode"):
        spike_viz.render_rate_poisson_hero(*args, rate_mode="unknown")
    image = spike_viz.render_rate_poisson_hero(*args, rate_mode="batch")
    assert "RateEncoder (stochastic batch)" in image.info["Description"]


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "axon-encoder" / "rate"


@pytest.fixture
def gain_cases():
    return tuple(
        spike_viz.load_axon_export(ROOT / f"shared_sine_gain_{gain}_v1")
        for gain in (0, 1, 2)
    )


def test_gain_triptych_preserves_exported_occupancy_and_png_provenance(
    gain_cases, tmp_path
):
    out = tmp_path / "triptych.png"
    image = spike_viz.render_gain_triptych(*gain_cases, out_path=out, scale=1)
    with Image.open(out) as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(image))
        caption = saved.info["Description"]
    for text in (
        "Silence",
        "Normal",
        "Elevated",
        "firing_rate_scale=0",
        "firing_rate_scale=1",
        "firing_rate_scale=2",
        "modulators: none",
        "encoder=rate",
        "dt=0.001 s",
        "seed=1592590337",
        "N=8",
        "T=64",
        gain_cases[0].meta["axon_encoder_git_sha"],
    ):
        assert text in caption

    # Cyan is reserved for event bins, never used for labels or decoration.
    # Compare every exported position, not just aggregate counts: lost events,
    # jitter, transposition, and fabricated residual silence must all fail.
    pixels = np.asarray(image)
    cyan = np.all(pixels == [0, 230, 255], axis=2)
    for case, panel in zip(gain_cases, np.array_split(cyan, 3, axis=1)):
        expected = np.zeros((8, 64), dtype=bool)
        expected[case.events.neuron_id, case.events.t] = True
        assert panel.sum() == expected.sum() * 9  # Each bin is 3 x 3.
        if expected.any():
            # Identify the raster origin from its first occupied bin. This
            # leaves text spacing free to change without changing the test.
            first_row, first_col = np.argwhere(expected)[0]
            y, x = np.argwhere(panel)[0] - [first_row * 3, first_col * 3]
            actual = panel[y : y + 24, x : x + 192]
            np.testing.assert_array_equal(
                actual, np.repeat(np.repeat(expected, 3, axis=0), 3, axis=1)
            )


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", 7),
        ("encoder_config", {"base_rate_hz": 75.0}),
        ("axon_encoder_git_sha", "a" * 40),
        ("synthetic", True),
        ("exporter_patch_sha256", "0" * 64),
        ("encoding_gains", {"firing_rate_scale": 1.0}),
    ],
)
def test_gain_triptych_rejects_invalid_comparisons(gain_cases, key, value):
    cases = list(gain_cases)
    meta = deepcopy(cases[2].meta)
    meta[key] = value
    cases[2] = replace(cases[2], meta=meta)
    with pytest.raises(ValueError):
        spike_viz.render_gain_triptych(*cases)


def test_gain_triptych_rejects_different_stimulus(gain_cases, tmp_path):
    stimulus = np.load(gain_cases[2].stimulus_path, allow_pickle=False)
    stimulus[0, 1] += 0.01
    path = tmp_path / "different.npy"
    np.save(path, stimulus)
    elevated = replace(gain_cases[2], stimulus_path=path)
    with pytest.raises(ValueError, match="stimulus"):
        spike_viz.render_gain_triptych(*gain_cases[:2], elevated)


def test_gain_triptych_rejects_nonempty_silence_and_wrong_order(gain_cases):
    silence = replace(gain_cases[0], events=gain_cases[1].events)
    with pytest.raises(ValueError, match="silence"):
        spike_viz.render_gain_triptych(silence, *gain_cases[1:])
    with pytest.raises(ValueError, match="gain"):
        spike_viz.render_gain_triptych(*reversed(gain_cases))


def test_gain_triptych_captions_exported_modulator_summary(gain_cases):
    cases = []
    for case in gain_cases:
        meta = deepcopy(case.meta)
        meta["modulators"] = {"dopamine": 0.25, "serotonin": 0.75}
        cases.append(replace(case, meta=meta))
    image = spike_viz.render_gain_triptych(*cases)
    assert (
        'modulators: {"dopamine": 0.25, "serotonin": 0.75}' in image.info["Description"]
    )


@pytest.mark.parametrize(
    "key",
    ["threshold_scale", "sensitivity_scale", "latency_scale"],
)
def test_gain_triptych_rejects_boolean_identity_gains(gain_cases, key):
    meta = deepcopy(gain_cases[2].meta)
    meta["encoding_gains"] = dict(meta["encoding_gains"])
    meta["encoding_gains"][key] = True
    elevated = replace(gain_cases[2], meta=meta)
    with pytest.raises(ValueError, match="identity non-rate"):
        spike_viz.render_gain_triptych(*gain_cases[:2], elevated)


def test_gain_triptych_rejects_overflowing_firing_rate_gain(gain_cases):
    meta = deepcopy(gain_cases[1].meta)
    meta["encoding_gains"] = dict(meta["encoding_gains"])
    meta["encoding_gains"]["firing_rate_scale"] = 10**1000
    normal = replace(gain_cases[1], meta=meta)
    with pytest.raises(ValueError, match="finite"):
        spike_viz.render_gain_triptych(gain_cases[0], normal, gain_cases[2])


def test_gain_triptych_rejects_differing_modulators(gain_cases):
    cases = []
    for index, case in enumerate(gain_cases):
        meta = deepcopy(case.meta)
        meta["modulators"] = {"dopamine": 0.25 if index < 2 else 0.75}
        cases.append(replace(case, meta=meta))
    with pytest.raises(ValueError, match="identical modulators"):
        spike_viz.render_gain_triptych(*cases)


@pytest.mark.parametrize("digest", [None, "not-a-digest", "A" * 64, "0" * 63])
def test_gain_triptych_rejects_unpinned_exporter_patch(gain_cases, digest):
    cases = []
    for case in gain_cases:
        meta = deepcopy(case.meta)
        if digest is None:
            del meta["exporter_patch_sha256"]
        else:
            meta["exporter_patch_sha256"] = digest
        cases.append(replace(case, meta=meta))
    with pytest.raises(ValueError, match="SHA-256"):
        spike_viz.render_gain_triptych(*cases)


def test_gain_triptych_wraps_wide_modulator_captions(gain_cases):
    wide = "W" * 80
    cases = []
    for case in gain_cases:
        meta = deepcopy(case.meta)
        meta["modulators"] = {wide: 1}
        cases.append(replace(case, meta=meta))
    image = spike_viz.render_gain_triptych(*cases, scale=1)
    plain = spike_viz.render_gain_triptych(*gain_cases, scale=1)
    assert image.size[0] == plain.size[0]
    assert ImageFont.load_default().getlength(wide) > image.size[0] / 3
    pixels = np.asarray(image)
    margin = pixels[0, -1]
    assert np.all(pixels[:, -16:] == margin)
    assert wide in image.info["Description"]


def test_gain_triptych_accepts_numpy_integer_scale(gain_cases):
    base = spike_viz.render_gain_triptych(*gain_cases, scale=1)
    scaled = spike_viz.render_gain_triptych(*gain_cases, scale=np.int64(2))
    assert scaled.size == (base.size[0] * 2, base.size[1] * 2)
