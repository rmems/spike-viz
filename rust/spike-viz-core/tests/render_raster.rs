//! CPU raster renderer tests: sparse event placement, dimensions,
//! intensity rules, invalid data, and empty/silent activity.

mod common;

use common::*;
use spike_viz_core::{
    render_dense, render_raster, sparse_to_dense, write_png, DenseGrid, SpikeEvents, SpikeIOError,
};

fn make_events(t: &[i64], n: &[i64], amp: Option<&[f32]>) -> SpikeEvents {
    SpikeEvents::new(t.to_vec(), n.to_vec(), amp.map(|a| a.to_vec())).unwrap()
}

#[test]
fn sparse_placement_marks_expected_pixels() {
    let events = make_events(&[0, 2, 4], &[1, 0, 2], None);
    let raster = render_raster(&events, 5, 3, 1).unwrap();
    assert_eq!((raster.width(), raster.height()), (5, 3));

    let out = temp_dir("placement").join("raster.png");
    write_png(&raster, &out).unwrap();
    let (width, height, rgb, color) = decode_png(&out);
    assert_eq!((width, height), (5, 3));
    assert_eq!(color, png::ColorType::Rgb);

    // time is horizontal (columns), neuron is vertical (rows)
    assert_eq!(gray_at(&rgb, 5, 1, 0), 255);
    assert_eq!(gray_at(&rgb, 5, 0, 2), 255);
    assert_eq!(gray_at(&rgb, 5, 2, 4), 255);
    let lit = (0..3)
        .flat_map(|row| (0..5).map(move |col| (row, col)))
        .filter(|(row, col)| gray_at(&rgb, 5, *row, *col) != 0)
        .count();
    assert_eq!(lit, 3);
}

#[test]
fn dimensions_scale_with_nearest_neighbor() {
    let events = make_events(&[0], &[0], None);
    let raster = render_raster(&events, 2, 2, 3).unwrap();
    assert_eq!((raster.width(), raster.height()), (6, 6));
    // Unscaled pixels stay [N, T].
    assert_eq!(raster.pixels().len(), 4);
    let scaled = raster.scaled_pixels().unwrap();
    assert_eq!(scaled.len(), 36);
    // The single spike becomes a 3x3 bright block at (row 0, col 0..2).
    assert_eq!(scaled[0], 255);
    assert_eq!(scaled[2], 255);
    assert_eq!(scaled[6], 255);
    assert_eq!(scaled[8], 255);
    assert_eq!(scaled[17], 0);
}

#[test]
fn empty_events_render_all_black() {
    let events = make_events(&[], &[], None);
    let raster = render_raster(&events, 3, 2, 1).unwrap();
    assert_eq!((raster.width(), raster.height()), (3, 2));
    assert!(raster.pixels().iter().all(|value| *value == 0));

    let out = temp_dir("empty-raster").join("raster.png");
    write_png(&raster, &out).unwrap();
    let (width, height, rgb, _) = decode_png(&out);
    assert_eq!((width, height), (3, 2));
    assert!(rgb.iter().all(|value| *value == 0));
}

#[test]
fn silent_real_export_renders_all_black() {
    let case = spike_viz_core::load_axon_export(fixture("rate", "shared_sine_gain_0_v1")).unwrap();
    let raster = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        1,
    )
    .unwrap();
    assert_eq!((raster.width(), raster.height()), (64, 8));
    assert!(raster.pixels().iter().all(|value| *value == 0));
    assert_eq!(raster.peak(), 0.0);
}

#[test]
fn zero_amp_events_are_preserved() {
    // axon-encoder writes polarity=false as amp=0.0; those events must
    // still appear rather than blending into the black background.
    let events = make_events(&[0, 1], &[0, 1], Some(&[1.0, 0.0]));
    let raster = render_raster(&events, 2, 2, 1).unwrap();
    assert_eq!(raster.pixels()[0], 255); // (n=0, t=0)
    assert_eq!(raster.pixels()[3], 255); // (n=1, t=1)
}

#[test]
fn duplicate_events_accumulate_and_normalize() {
    // Two events in one bin (peak 2) plus one event in another (amp 1):
    // 1/2 * 255 = 127.5 rounds half-to-even to 128.
    let events = make_events(&[0, 0, 1], &[0, 0, 0], Some(&[1.0, 1.0, 1.0]));
    let raster = render_raster(&events, 2, 1, 1).unwrap();
    assert_eq!(raster.pixels(), &[255, 128]);
    assert_eq!(raster.peak(), 2.0);
}

#[test]
fn negative_amplitudes_clip_to_black() {
    // Mixed polarity: negative bins clip to 0, positive bins normalize.
    let events = make_events(&[0, 1], &[0, 0], Some(&[1.0, -1.0]));
    let raster = render_raster(&events, 2, 1, 1).unwrap();
    assert_eq!(raster.pixels(), &[255, 0]);
}

#[test]
fn all_negative_amplitudes_render_all_black() {
    let events = make_events(&[0], &[0], Some(&[-1.0]));
    let raster = render_raster(&events, 1, 1, 1).unwrap();
    assert_eq!(raster.pixels(), &[0]);
    assert_eq!(raster.peak(), -1.0);
}

#[test]
fn peak_bin_is_full_white() {
    let events = make_events(&[0, 1], &[0, 0], Some(&[0.25, 1.0]));
    let raster = render_raster(&events, 2, 1, 1).unwrap();
    assert_eq!(raster.pixels()[0], 64); // 0.25 * 255 = 63.75 -> 64
    assert_eq!(raster.pixels()[1], 255);
}

#[test]
fn scale_below_one_rejected() {
    let events = make_events(&[0], &[0], None);
    let err = render_raster(&events, 1, 1, 0).unwrap_err();
    assert!(
        err.to_string().contains("scale must be an integer >= 1"),
        "got: {err}"
    );
}

#[test]
fn zero_geometry_rejected() {
    let events = make_events(&[0], &[0], None);
    for (n_steps, n_neurons) in [(0, 2), (2, 0)] {
        let err = render_raster(&events, n_steps, n_neurons, 1).unwrap_err();
        assert!(err.to_string().contains("must be >= 1"), "got: {err}");
    }
}

#[test]
fn oversized_dense_geometry_returns_invalid() {
    let events = make_events(&[], &[], None);
    for (n_steps, n_neurons) in [
        (usize::MAX / 2 + 1, 2), // cell count overflows usize
        (2, usize::MAX / 2 + 1),
        (isize::MAX as usize / 4 + 1, 1), // float32 allocation exceeds isize::MAX
    ] {
        assert!(matches!(
            DenseGrid::new(n_steps, n_neurons, vec![]),
            Err(SpikeIOError::Invalid(_))
        ));
        assert!(matches!(
            sparse_to_dense(&events, n_steps, n_neurons),
            Err(SpikeIOError::Invalid(_))
        ));
    }
}

#[test]
fn oversized_scaled_rasters_return_invalid_before_allocation() {
    let events = make_events(&[], &[], None);
    for (n_steps, n_neurons, scale) in [
        (2, 1, usize::MAX / 2 + 1),    // width overflows
        (1, 2, usize::MAX / 2 + 1),    // height overflows
        (1, 1, i32::MAX as usize + 1), // exceeds PNG dimension limit
        (1, 1, i32::MAX as usize),     // RGB buffer exceeds addressable size
    ] {
        let grid = DenseGrid::new(n_steps, n_neurons, vec![0.0; n_steps * n_neurons]).unwrap();
        assert!(matches!(
            render_dense(&grid, scale),
            Err(SpikeIOError::Invalid(_))
        ));
        assert!(matches!(
            render_raster(&events, n_steps, n_neurons, scale),
            Err(SpikeIOError::Invalid(_))
        ));
    }
    // An oversized unscaled frame must fail before allocating a dense grid.
    assert!(matches!(
        render_raster(&events, i32::MAX as usize + 1, 1, 1),
        Err(SpikeIOError::Invalid(_))
    ));
}

#[test]
fn out_of_geometry_events_rejected_at_raster() {
    let events = make_events(&[5], &[0], None);
    let err = render_raster(&events, 2, 2, 1).unwrap_err();
    assert!(
        err.to_string().contains("event time out of range"),
        "got: {err}"
    );

    let events = make_events(&[0], &[7], None);
    let err = render_raster(&events, 2, 2, 1).unwrap_err();
    assert!(
        err.to_string().contains("neuron_id out of range"),
        "got: {err}"
    );
}

#[test]
fn sparse_to_dense_accumulates_without_substituting_zero_amp() {
    // sparse_to_dense mirrors Python: stored zero amps stay zero (the
    // zero->1 substitution is a render-time rule only).
    let events = make_events(&[0, 1], &[0, 0], Some(&[2.0, 0.0]));
    let grid = sparse_to_dense(&events, 2, 1).unwrap();
    assert_eq!(grid.data(), &[2.0, 0.0]);
    let raster = render_raster(&events, 2, 1, 1).unwrap();
    // Render substitutes the zero amp: grid becomes [2.0, 1.0], so the
    // second bin normalizes to 1/2 * 255 = 127.5 -> 128 (half to even).
    assert_eq!(raster.pixels(), &[255, 128]);
}

#[test]
fn dense_grid_rejects_non_finite_values() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let grid = DenseGrid::new(1, 1, vec![bad]).unwrap();
        let err = render_dense(&grid, 1).unwrap_err();
        assert!(
            err.to_string().contains("non-finite or out-of-range"),
            "got: {err}"
        );
    }
}

#[test]
fn rate_fixture_renders_without_inventing_or_dropping_events() {
    let case = spike_viz_core::load_axon_export(fixture("rate", "shared_sine_v1")).unwrap();
    let raster = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        8,
    )
    .unwrap();
    assert_eq!((raster.width(), raster.height()), (512, 64));

    // Every event bin must be lit (no dropped spikes) and every lit bin
    // must correspond to at least one real event (no invented spikes).
    let mut event_bins = std::collections::HashSet::new();
    for (t, n, _amp) in case.events.iter() {
        event_bins.insert((t, n));
    }
    let mut lit_bins = std::collections::HashSet::new();
    for (index, value) in raster.pixels().iter().enumerate() {
        if *value != 0 {
            let n = (index / raster.grid_shape().0) as i64;
            let t = (index % raster.grid_shape().0) as i64;
            lit_bins.insert((t, n));
        }
    }
    assert_eq!(lit_bins, event_bins);
    // Every lit bin is full white (uniform 1.0 amplitudes against peak 1.0);
    // the unlit background stays black.
    assert_eq!(
        raster.pixels().iter().filter(|value| **value != 0).count(),
        event_bins.len()
    );
    assert!(raster
        .pixels()
        .iter()
        .filter(|value| **value != 0)
        .all(|value| *value == 255));
    assert_eq!(raster.peak(), 1.0);
}

#[test]
fn tiny_synthetic_fixture_renders_expected_intensities() {
    let case = spike_viz_core::load_axon_export(fixture("rate", "tiny_synthetic")).unwrap();
    let raster = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        1,
    )
    .unwrap();
    let (n_steps, n_neurons) = raster.grid_shape();
    let value_at = |t: usize, n: usize| raster.pixels()[n * n_steps + t];

    // Seven events, one with amp 0.5: six full-white bins and one 128 bin.
    let lit: Vec<u8> = raster
        .pixels()
        .iter()
        .copied()
        .filter(|value| *value != 0)
        .collect();
    assert_eq!(lit.len(), 7);
    assert_eq!(lit.iter().filter(|value| **value == 255).count(), 6);
    // amp 0.5 sits at (t=5, neuron=0).
    assert_eq!(value_at(5, 0), 128);
    assert_eq!(value_at(0, 0), 255);
    assert_eq!(value_at(5, 3), 255);
    assert_eq!(value_at(7, 2), 255);
    assert_eq!(n_neurons, 4);
    assert_eq!(n_steps, 8);
}

#[test]
fn png_is_rgb8_like_the_python_renderer() {
    let events = make_events(&[0], &[0], None);
    let raster = render_raster(&events, 1, 1, 1).unwrap();
    let out = temp_dir("rgb-png").join("raster.png");
    write_png(&raster, &out).unwrap();
    let (width, height, rgb, color) = decode_png(&out);
    assert_eq!((width, height), (1, 1));
    assert_eq!(color, png::ColorType::Rgb);
    assert_eq!(rgb, &[255, 255, 255]);
}

#[test]
fn write_png_requires_existing_parent() {
    let events = make_events(&[0], &[0], None);
    let raster = render_raster(&events, 1, 1, 1).unwrap();
    let out = temp_dir("no-parent").join("missing").join("raster.png");
    let err = write_png(&raster, &out).unwrap_err();
    assert!(
        err.to_string().contains("failed to write PNG"),
        "got: {err}"
    );
}
