//! Semantic report tests: the Rust/Python parity bridge payload.

mod common;

use common::*;
use spike_viz_core::{load_axon_export, render_raster, semantic_report, SemanticReport};

#[test]
fn golden_fixture_report() {
    let case = load_axon_export(fixture("rate", "tiny_synthetic")).unwrap();
    let report = semantic_report(&case, 16).unwrap();

    assert_eq!(report.tool, "spike-viz-core");
    assert_eq!(report.encoder, "rate");
    assert_eq!(report.schema_version, "1.0");
    assert_eq!(report.n_steps, 8);
    assert_eq!(report.n_neurons, 4);
    assert_eq!(report.n_events, 7);
    assert_eq!(report.scale, 16);
    assert_eq!(report.width, 128);
    assert_eq!(report.height, 64);
    assert_eq!(report.peak, 1.0);
    assert_eq!(report.synthetic, Some(true));
    assert_eq!(report.axon_encoder_git_sha, None);

    // Events preserved in load order, no invention or dropping.
    let expected: Vec<(i64, i64, Option<f64>)> = vec![
        (0, 0, Some(1.0)),
        (1, 0, Some(1.0)),
        (1, 2, Some(1.0)),
        (3, 1, Some(1.0)),
        (5, 0, Some(0.5)),
        (5, 3, Some(1.0)),
        (7, 2, Some(1.0)),
    ];
    let actual: Vec<(i64, i64, Option<f64>)> = report
        .events
        .iter()
        .map(|event| (event.t, event.neuron_id, event.amp))
        .collect();
    assert_eq!(actual, expected);

    // Bins: six full-white, one 128 (the 0.5 amplitude), time-major order.
    assert_eq!(report.bins.len(), 7);
    assert_eq!(report.bins.iter().filter(|b| b.value == 255).count(), 6);
    assert_eq!(report.bins.iter().filter(|b| b.value == 128).count(), 1);
    let dim_bin = report
        .bins
        .iter()
        .find(|b| b.value == 128)
        .expect("one dim bin");
    assert_eq!((dim_bin.t, dim_bin.neuron_id), (5, 0));
}

#[test]
fn real_rate_report_keeps_provenance() {
    let case = load_axon_export(fixture("rate", "shared_sine_v1")).unwrap();
    let report = semantic_report(&case, 8).unwrap();
    assert_eq!(report.synthetic, Some(false));
    assert_eq!(
        report.axon_encoder_git_sha.as_deref(),
        Some("908a34797c8aaff0a57e0ea4167aa00f3780c940")
    );
    assert_eq!(report.n_events, 127);
    assert_eq!(report.bins.len(), 127);
    assert_eq!(report.peak, 1.0);
    assert_eq!(report.dt_seconds, 0.001);
    assert_eq!(report.seed, 1592590337);
    // Every reported bin traces back to a reported event.
    let event_bins: std::collections::HashSet<(i64, i64)> = report
        .events
        .iter()
        .map(|event| (event.t, event.neuron_id))
        .collect();
    for bin in &report.bins {
        assert!(event_bins.contains(&(bin.t, bin.neuron_id)));
    }
}

#[test]
fn silent_report_is_honestly_empty() {
    let case = load_axon_export(fixture("rate", "shared_sine_gain_0_v1")).unwrap();
    let report = semantic_report(&case, 1).unwrap();
    assert_eq!(report.n_events, 0);
    assert!(report.bins.is_empty());
    assert_eq!(report.peak, 0.0);
    assert_eq!((report.width, report.height), (64, 8));
}

#[test]
fn report_serializes_to_json() {
    let case = load_axon_export(fixture("rate", "tiny_synthetic")).unwrap();
    let report = semantic_report(&case, 1).unwrap();
    let json = serde_json::to_string(&report).expect("serialize report");
    assert!(json.contains("\"tool\":\"spike-viz-core\""));
    assert!(json.contains("\"n_events\":7"));

    let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse report");
    assert_eq!(parsed["n_steps"], 8);
    assert_eq!(parsed["bins"].as_array().expect("bins array").len(), 7);
    // `amp` nulls when the export omitted the array; here it is present.
    assert_eq!(parsed["events"][4]["amp"], 0.5);
}

#[test]
fn from_render_matches_render_report() {
    let case = load_axon_export(fixture("rate", "tiny_synthetic")).unwrap();
    let raster = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        4,
    )
    .unwrap();
    let report = SemanticReport::from_render(&case, &raster).unwrap();
    assert_eq!(report.scale, 4);
    assert_eq!((report.width, report.height), (32, 16));
    assert_eq!(report.n_events, 7);
}

#[test]
fn from_render_rejects_geometry_mismatch() {
    // `from_render` takes independent `case` and `raster` arguments; a
    // raster rendered from different geometry must fail loud rather than
    // silently misreport bins against the wrong case.
    let case = load_axon_export(fixture("rate", "tiny_synthetic")).unwrap();
    // Geometry large enough to hold every event but different from the
    // case meta (8x4): from_render must reject the mismatch.
    let raster = render_raster(&case.events, 16, 8, 1).unwrap();
    let err = SemanticReport::from_render(&case, &raster).unwrap_err();
    assert!(
        err.to_string().contains("does not match case geometry"),
        "got: {err}"
    );
}

#[test]
fn fractional_amp_serializes_exactly_for_parity() {
    // The Python parity gate widens the in-memory f32 amplitudes to f64
    // and compares with `==`. serde_json writes the shortest round-trip
    // *f32* decimal for an f32 field (`0.1`), which is NOT equal to the
    // f64 widening (`0.10000000149011612`) — so the report must widen
    // before serializing. Fractional amps only exist in this fixture;
    // every other checked-in case has representable amps (0.5/1.0).
    let case = load_axon_export(fixture("rate", "tiny_fractional_amp")).unwrap();
    let report = semantic_report(&case, 1).unwrap();
    let json = serde_json::to_string(&report).expect("serialize report");
    assert!(
        json.contains("\"amp\":0.10000000149011612"),
        "exact f64 widening of 0.1f32 missing: {json}"
    );
    assert!(
        json.contains("\"amp\":0.699999988079071"),
        "exact f64 widening of 0.7f32 missing: {json}"
    );
    // 0.0 (polarity-false export) is serialized as a plain zero, and the
    // zero-amp event renders at full intensity (substituted unit amplitude)
    // while the fractional amps land on 26 (0.1) and 178 (0.7).
    assert!(
        json.contains("\"amp\":0.0}"),
        "raw zero amp missing: {json}"
    );
    let values: Vec<u8> = report.bins.iter().map(|bin| bin.value).collect();
    assert_eq!(values, vec![26, 178, 255, 255]);
    assert_eq!(report.peak, 1.0);
}
