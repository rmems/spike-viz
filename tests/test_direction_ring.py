"""Public contract for explicit preferred-direction geometry and spike truth."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import spike_viz
from spike_viz import SpikeEvents, load_axon_export

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/axon-encoder/population/shared_sine_v1"
)


def direction_case():
    # Test-only geometry, not a claim about the real scalar population fixture.
    case = load_axon_export(FIXTURE)
    return replace(
        case,
        events=SpikeEvents(t=[0, 1, 1, 2], neuron_id=[0, 1, 1, 2], amp=[1, -1, 0, 1]),
        meta={
            **case.meta,
            "n_neurons": 4,
            "n_steps": 4,
            "synthetic": True,
            "preferred_angles_radians": [np.pi, 0, np.pi / 2, 3 * np.pi / 2],
        },
    )


def test_direction_ring_uses_exported_order_and_half_open_window(tmp_path):
    case = direction_case()
    before = case.events.t.copy(), case.events.neuron_id.copy(), case.events.amp.copy()
    path = tmp_path / "ring.png"
    image = spike_viz.render_direction_ring(
        case, window=(1, 2), radius=100, glow=False, out_path=path
    )
    # Shuffled angles: neuron 1 is east, not north. Both negative and zero
    # polarity events count; step 0 and the exclusive end at step 2 do not.
    pixels = np.asarray(image)
    cx, cy = image.width // 2, (image.width // 2) + 40
    assert pixels[cy, cx + 100].max() > 200
    assert pixels[cy, cx - 100].max() < 100
    assert pixels[cy - 100, cx].max() < 100
    assert pixels[cy + 100, cx].max() < 100
    for original, current in zip(
        before, (case.events.t, case.events.neuron_id, case.events.amp)
    ):
        np.testing.assert_array_equal(original, current)
    with Image.open(path) as saved:
        np.testing.assert_array_equal(np.asarray(saved), pixels)
        caption = saved.info["provenance"]
        for value in (
            "population",
            "dt=0.001",
            "seed=1592590337",
            "N=4",
            "T=4",
            case.meta["axon_encoder_git_sha"],
            "synthetic=True",
            "[1, 2)",
            "source_events=2",
        ):
            assert value in caption


def test_direction_ring_counts_events_and_places_positive_angle_north():
    image = spike_viz.render_direction_ring(
        direction_case(), window=(0, 3), radius=100, glow=False
    )
    pixels = np.asarray(image)
    cx, cy = image.width // 2, image.width // 2 + 40
    # Counts are west=1, east=2, north=1, south=0. Summing amplitudes would
    # cancel the east spikes; clockwise angles would put north activity south.
    assert pixels[cy, cx + 100].max() > pixels[cy - 100, cx].max() > 100
    assert pixels[cy, cx - 100].max() > pixels[cy + 100, cx].max()
    assert pixels[cy + 100, cx].max() < 100


def test_direction_ring_default_is_last_export_step_not_last_spike():
    case = direction_case()
    default = spike_viz.render_direction_ring(case, radius=100)
    silent = spike_viz.render_direction_ring(case, radius=100, window=(3, 4))
    np.testing.assert_array_equal(np.asarray(default), np.asarray(silent))
    active = spike_viz.render_direction_ring(case, radius=100, window=(0, 3))
    assert np.asarray(active).sum() > np.asarray(silent).sum()


def test_scalar_population_fixture_is_not_silently_given_angles():
    with pytest.raises(ValueError, match="preferred_angles_radians"):
        spike_viz.render_direction_ring(load_axon_export(FIXTURE))


@pytest.mark.parametrize(
    "angles", [[0], [0, 1, 2, float("nan")], [[0, 1], [2, 3]], [0, 1, 2, True]]
)
def test_direction_ring_rejects_invalid_geometry(angles):
    case = direction_case()
    case.meta["preferred_angles_radians"] = angles
    with pytest.raises(ValueError, match="preferred_angles_radians"):
        spike_viz.render_direction_ring(case)


@pytest.mark.parametrize(
    "window",
    [(-1, 2), (1, 1), (0, 5), (0.5, 2), (True, 2), (np.float64(1), 2), (np.bool_(True), 2)],
)
def test_direction_ring_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        spike_viz.render_direction_ring(direction_case(), window=window)


@pytest.mark.parametrize(
    "radius", [0, 31, 1.5, True, np.float64(100), np.bool_(True), np.int64(31)]
)
def test_direction_ring_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        spike_viz.render_direction_ring(direction_case(), radius=radius)


def test_direction_ring_accepts_numpy_integer_window_and_radius():
    case = direction_case()
    image = spike_viz.render_direction_ring(
        case, window=(np.int64(1), np.int64(2)), radius=np.int64(100), glow=False
    )
    expected = spike_viz.render_direction_ring(case, window=(1, 2), radius=100, glow=False)
    np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))


@pytest.mark.parametrize("neuron_id", [[4], [-1], [0, 4]])
def test_direction_ring_rejects_neuron_ids_outside_population(neuron_id):
    case = replace(
        direction_case(),
        events=SpikeEvents(
            t=[0] * len(neuron_id), neuron_id=neuron_id, amp=[1] * len(neuron_id)
        ),
    )
    # Default window is the last step, so these ids are outside the painted window
    # and must still be rejected before counting.
    with pytest.raises(ValueError, match=r"neuron_id must be within \[0, n_neurons\)"):
        spike_viz.render_direction_ring(case, radius=64, glow=False)


def test_coincident_angles_keep_each_neuron_and_the_stronger_count():
    case = replace(
        direction_case(),
        events=SpikeEvents(t=[0] * 11, neuron_id=[0] * 10 + [1], amp=[1] * 11),
    )
    case.meta["n_neurons"] = 2
    case.meta["preferred_angles_radians"] = [0.0, 0.0]
    image = spike_viz.render_direction_ring(case, window=(0, 1), radius=100, glow=False)
    pixels = np.asarray(image)
    cx, cy = image.width // 2, image.width // 2 + 40
    stronger = pixels[cy, cx + 100]
    weaker = pixels[cy - 24, cx + 100]
    assert stronger.max() > 200
    assert 40 < weaker.max() < stronger.max()
    assert "source_events=11" in image.info["provenance"]


def test_thirty_two_coincident_neurons_stay_in_the_plot_at_minimum_radius():
    n = 32
    radius = 32
    counts = [n - neuron for neuron in range(n)]
    neuron_id = [neuron for neuron, count in enumerate(counts) for _ in range(count)]
    case = replace(
        direction_case(),
        events=SpikeEvents(
            t=[0] * len(neuron_id), neuron_id=neuron_id, amp=[1] * len(neuron_id)
        ),
    )
    case.meta["n_neurons"] = n
    case.meta["preferred_angles_radians"] = [0.0] * n
    image = spike_viz.render_direction_ring(case, window=(0, 1), radius=radius, glow=False)
    pixels = np.asarray(image)
    height, width = pixels.shape[:2]
    cx, cy = width // 2, width // 2 + 40
    caption_top = width + 20 - 28
    assert pixels[cy, cx + radius].max() > 200
    # Weaker neurons alternate along the tangent at 24px, strongest held on the ring.
    for index in range(n - 1):
        step = (index // 2 + 1) * 24
        sign = -1 if index % 2 else 1
        y = cy - sign * step
        x = cx + radius
        assert 80 <= y < caption_top - 10
        assert 0 <= x < width
        assert y < height
        assert pixels[y, x].max() > 40
    assert f"source_events={sum(counts)}" in image.info["provenance"]


def test_direction_ring_rejects_plot_larger_than_4096_pixels():
    n = 1024
    case = replace(
        direction_case(),
        events=SpikeEvents(t=[0], neuron_id=[0], amp=[1]),
    )
    case.meta["n_neurons"] = n
    case.meta["preferred_angles_radians"] = [0.0] * n
    with pytest.raises(ValueError, match="4096"):
        spike_viz.render_direction_ring(case, radius=32, glow=False)


def test_provenance_prints_source_dt_and_escapes_surrogate_strings(tmp_path):
    case = direction_case()
    case.meta["dt_seconds"] = 0.001000001
    case.meta["encoder"] = "rate\ud800"
    case.meta["axon_encoder_git_sha"] = "abc\ud800"
    path = tmp_path / "ring.png"
    image = spike_viz.render_direction_ring(case, radius=64, glow=False, out_path=path)
    caption = image.info["provenance"]
    assert "dt=0.001000001 s/step" in caption
    assert json.dumps("rate\ud800") in caption
    assert json.dumps("abc\ud800") in caption
    with Image.open(path) as saved:
        assert saved.info["provenance"] == caption
