//! `load_axon_export` / `load_meta` contract tests: `meta.json` validation,
//! geometry bounds, and the checked-in golden fixtures.

mod common;

use common::*;
use spike_viz_core::{load_axon_export, load_meta, SpikeIOError};

fn golden() -> std::path::PathBuf {
    fixture("rate", "tiny_synthetic")
}

#[test]
fn golden_fixture_loads() {
    let case = load_axon_export(golden()).unwrap();
    assert_eq!(case.meta.encoder, "rate");
    assert_eq!(case.meta.schema_version, "1.0");
    assert_eq!(case.meta.n_neurons, 4);
    assert_eq!(case.meta.n_steps, 8);
    assert_eq!(case.meta.dt_seconds, 0.001);
    assert_eq!(case.meta.seed, 0);
    assert_eq!(case.meta.synthetic, Some(true));
    assert_eq!(case.meta.axon_encoder_git_sha, None);
    assert_eq!(case.events.len(), 7);
    assert!(case.stimulus_path.is_none());
    assert_eq!(case.spikes_path, golden().join("spikes.npz"));
    assert_eq!(case.meta_path, golden().join("meta.json"));
}

#[test]
fn every_checked_in_fixture_loads() {
    let cases = all_fixture_cases();
    assert!(
        cases.len() >= 8,
        "expected the checked-in fixtures, got {cases:?}"
    );
    for case_dir in &cases {
        let case = load_axon_export(case_dir)
            .unwrap_or_else(|e| panic!("{} failed to load: {e}", case_dir.display()));
        // Geometry must be at least 1x1 even when the export is silent.
        assert!(case.meta.n_steps >= 1);
        assert!(case.meta.n_neurons >= 1);
        // Events were bounds-checked during load; a silent export is valid.
        assert_eq!(case.events.len(), case.events.t().len());
    }
}

#[test]
fn silent_real_export_loads_with_zero_events() {
    // rate gain 0 is an honestly empty real axon-encoder export.
    let case = load_axon_export(fixture("rate", "shared_sine_gain_0_v1")).unwrap();
    assert_eq!(case.meta.synthetic, Some(false));
    assert!(case.events.is_empty());
    assert_eq!(case.meta.n_neurons, 8);
    assert_eq!(case.meta.n_steps, 64);
    assert!(case.stimulus_path.is_some());
}

#[test]
fn real_rate_export_keeps_provenance() {
    let case = load_axon_export(fixture("rate", "shared_sine_v1")).unwrap();
    assert_eq!(case.meta.synthetic, Some(false));
    assert_eq!(
        case.meta.axon_encoder_git_sha.as_deref(),
        Some("908a34797c8aaff0a57e0ea4167aa00f3780c940")
    );
    assert!(!case.events.is_empty());
}

#[test]
fn missing_case_dir_fails_loud() {
    let dir = temp_dir("no-case");
    let err = load_axon_export(dir.join("missing")).unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

#[test]
fn missing_spikes_fails_loud() {
    let dir = temp_dir("no-spikes");
    write_meta(&dir.join("meta.json"), valid_meta());
    let err = load_axon_export(&dir).unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

#[test]
fn missing_meta_fails_loud() {
    let dir = temp_dir("no-meta");
    write_npz(&dir.join("spikes.npz"), &[0], &[0], None);
    let err = load_axon_export(&dir).unwrap_err();
    assert!(
        err.to_string().contains("meta.json not found"),
        "got: {err}"
    );
}

#[test]
fn meta_missing_required_keys_fails_loud() {
    let dir = temp_dir("missing-keys");
    let meta = serde_json::json!({"encoder": "rate"});
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(
        err.to_string().contains("missing required keys"),
        "got: {err}"
    );
}

#[test]
fn geometry_mismatch_t_out_of_range() {
    let dir = temp_dir("t-oob");
    write_case(&dir, &valid_meta(), &[10], &[0], None);
    let err = load_axon_export(&dir).unwrap_err();
    assert!(err.to_string().contains("t out of range"), "got: {err}");
}

#[test]
fn geometry_mismatch_neuron_out_of_range() {
    let dir = temp_dir("n-oob");
    write_case(&dir, &valid_meta(), &[0], &[5], None);
    let err = load_axon_export(&dir).unwrap_err();
    assert!(
        err.to_string().contains("neuron_id out of range"),
        "got: {err}"
    );
}

#[test]
fn negative_time_rejected() {
    let dir = temp_dir("t-neg");
    write_case(&dir, &valid_meta(), &[-1], &[0], None);
    let err = load_axon_export(&dir).unwrap_err();
    assert!(err.to_string().contains("t out of range"), "got: {err}");
}

#[test]
fn zero_geometry_rejected_even_if_spikes_empty() {
    let dir = temp_dir("zero-geom");
    let mut meta = valid_meta();
    meta["n_neurons"] = serde_json::json!(0);
    write_case(&dir, &meta, &[], &[], None);
    let err = load_axon_export(&dir).unwrap_err();
    assert!(err.to_string().contains("n_neurons"), "got: {err}");
}

#[test]
fn zero_steps_rejected() {
    let dir = temp_dir("zero-steps");
    let mut meta = valid_meta();
    meta["n_steps"] = serde_json::json!(0);
    write_case(&dir, &meta, &[], &[], None);
    let err = load_axon_export(&dir).unwrap_err();
    assert!(err.to_string().contains("n_steps"), "got: {err}");
}

#[test]
fn zero_dt_seconds_rejected() {
    let dir = temp_dir("zero-dt");
    let mut meta = valid_meta();
    meta["dt_seconds"] = serde_json::json!(0.0);
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("dt_seconds"), "got: {err}");
}

#[test]
fn negative_dt_seconds_rejected() {
    let dir = temp_dir("neg-dt");
    let mut meta = valid_meta();
    meta["dt_seconds"] = serde_json::json!(-0.001);
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("dt_seconds"), "got: {err}");
}

#[test]
fn string_dt_seconds_rejected() {
    let dir = temp_dir("str-dt");
    let mut meta = valid_meta();
    meta["dt_seconds"] = serde_json::json!("0.001");
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("must be a number"), "got: {err}");
}

#[test]
fn oversized_dt_seconds_rejected() {
    // An integer too large for any float must fail loud, not wrap to 0/inf.
    let dir = temp_dir("huge-dt");
    let huge: String = "9".repeat(400);
    let meta_text = format!(
        "{{\"schema_version\": \"1.0\", \"encoder\": \"rate\", \"dt_seconds\": {huge}, \"seed\": 0, \"n_neurons\": 2, \"n_steps\": 2}}"
    );
    std::fs::write(dir.join("meta.json"), meta_text).expect("write meta");
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("dt_seconds") || message.contains("invalid JSON"),
        "got: {err}"
    );
}

#[test]
fn unsupported_schema_version_rejected() {
    let dir = temp_dir("bad-schema");
    let mut meta = valid_meta();
    meta["schema_version"] = serde_json::json!("banana");
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(
        err.to_string().contains("unsupported schema_version"),
        "got: {err}"
    );
}

#[test]
fn string_seed_rejected() {
    let dir = temp_dir("str-seed");
    let mut meta = valid_meta();
    meta["seed"] = serde_json::json!("nope");
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("seed"), "got: {err}");
}

#[test]
fn negative_seed_rejected() {
    let dir = temp_dir("neg-seed");
    let mut meta = valid_meta();
    meta["seed"] = serde_json::json!(-3);
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("seed"), "got: {err}");
}

#[test]
fn float_seed_rejected() {
    // Python rejects JSON 1.0 for an int field; so does the Rust loader.
    let dir = temp_dir("float-seed");
    let mut meta = valid_meta();
    meta["seed"] = serde_json::json!(1.0);
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("seed"), "got: {err}");
}

#[test]
fn boolean_geometry_rejected() {
    let dir = temp_dir("bool-geom");
    let mut meta = valid_meta();
    meta["n_neurons"] = serde_json::json!(true);
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(
        err.to_string().contains("'n_neurons' must be an integer"),
        "got: {err}"
    );
}

#[test]
fn empty_encoder_rejected() {
    let dir = temp_dir("empty-encoder");
    let mut meta = valid_meta();
    meta["encoder"] = serde_json::json!("");
    write_meta(&dir.join("meta.json"), meta);
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(
        err.to_string()
            .contains("'encoder' must be a non-empty string"),
        "got: {err}"
    );
}

#[test]
fn non_object_root_rejected() {
    let dir = temp_dir("root-array");
    std::fs::write(dir.join("meta.json"), "[1, 2, 3]").expect("write meta");
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(
        err.to_string().contains("root must be a JSON object"),
        "got: {err}"
    );
}

#[test]
fn invalid_json_rejected() {
    let dir = temp_dir("bad-json");
    std::fs::write(dir.join("meta.json"), "{not json").expect("write meta");
    let err = load_meta(dir.join("meta.json")).unwrap_err();
    assert!(err.to_string().contains("invalid JSON"), "got: {err}");
}

#[test]
fn silent_case_with_empty_arrays_loads() {
    let dir = temp_dir("silent-case");
    write_case(&dir, &valid_meta(), &[], &[], Some(&[]));
    let case = load_axon_export(&dir).unwrap();
    assert!(case.events.is_empty());
}

#[test]
fn optional_stimulus_detected_when_present() {
    let dir = temp_dir("with-stimulus");
    write_case(&dir, &valid_meta(), &[0], &[0], None);
    let stim = dir.join("stimulus.npy");
    let data: Vec<u8> = [0.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let npy = npy_bytes("<f4", false, &[2], &data);
    std::fs::write(&stim, npy).expect("write stimulus");
    let case = load_axon_export(&dir).unwrap();
    assert_eq!(case.stimulus_path, Some(stim));
}

#[test]
fn raw_meta_preserved_for_forward_compatibility() {
    let case = load_axon_export(golden()).unwrap();
    let notes = case
        .meta
        .raw
        .get("notes")
        .and_then(|value| value.as_str())
        .expect("notes preserved");
    assert!(notes.contains("Synthetic golden fixture"));
}

#[test]
fn error_type_is_spike_io_error() {
    let dir = temp_dir("err-kind");
    let err: SpikeIOError = load_axon_export(dir.join("missing")).unwrap_err();
    let _: &dyn std::error::Error = &err;
}
