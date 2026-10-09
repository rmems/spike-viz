"""Public contract for explicit preferred-direction geometry and spike truth."""

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


@pytest.mark.parametrize("window", [(-1, 2), (1, 1), (0, 5), (0.5, 2), (True, 2)])
def test_direction_ring_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        spike_viz.render_direction_ring(direction_case(), window=window)


@pytest.mark.parametrize("radius", [0, 31, 1.5, True])
def test_direction_ring_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        spike_viz.render_direction_ring(direction_case(), radius=radius)
