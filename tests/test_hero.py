"""Scientific gain comparisons: real exports in, unchanged occupancy out."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import spike_viz

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
