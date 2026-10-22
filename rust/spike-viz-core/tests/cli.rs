//! CLI end-to-end tests: the parity-bridge binary.

mod common;

use common::*;
use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_spike-viz"))
}

#[test]
fn render_golden_writes_png_and_report() {
    let dir = temp_dir("cli-render");
    let out = dir.join("raster.png");
    let report = dir.join("raster.json");

    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--scale",
            "16",
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("run spike-viz");

    assert!(
        status.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let stdout = String::from_utf8_lossy(&status.stdout);
    assert!(stdout.contains("encoder=rate"), "stdout: {stdout}");
    assert!(stdout.contains("events=7"), "stdout: {stdout}");

    let (width, height, _, color) = decode_png(&out);
    assert_eq!((width, height), (128, 64));
    assert_eq!(color, png::ColorType::Rgb);

    let parsed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(parsed["n_events"], 7);
    assert_eq!(parsed["n_steps"], 8);
    assert_eq!(parsed["n_neurons"], 4);
    assert_eq!(parsed["width"], 128);
    assert_eq!(parsed["height"], 64);
    assert_eq!(parsed["synthetic"], true);
}

#[test]
fn render_silent_export_succeeds() {
    let dir = temp_dir("cli-silent");
    let out = dir.join("silent.png");
    let report = dir.join("silent.json");

    let status = cli()
        .args([
            "render",
            fixture("rate", "shared_sine_gain_0_v1").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
        ])
        .output()
        .expect("run spike-viz");

    assert!(
        status.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let (_, _, rgb, _) = decode_png(&out);
    assert!(rgb.iter().all(|value| *value == 0));

    let parsed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(parsed["n_events"], 0);
    assert_eq!(parsed["bins"].as_array().unwrap().len(), 0);
    assert_eq!(parsed["peak"], 0.0);
}

#[test]
fn missing_output_target_is_a_usage_error() {
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(stderr.contains("nothing to do"), "stderr: {stderr}");
}

#[test]
fn same_out_and_report_path_is_a_usage_error() {
    // `File::create` truncates, so a report written to the PNG path would
    // silently replace the just-written image with JSON while exiting 0.
    let dir = temp_dir("cli-same-path");
    let target = dir.join("raster.png");
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--out",
            target.to_str().unwrap(),
            "--report",
            target.to_str().unwrap(),
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("must be different paths"),
        "stderr: {stderr}"
    );
    // Nothing was written to the target.
    assert!(!target.exists(), "target must not be created");
}

#[test]
fn aliased_out_and_report_path_is_a_usage_error() {
    // Differently-spelled paths to the same target (`./` alias, relative
    // spelling) must be caught too.
    let dir = temp_dir("cli-aliased-path");
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--out",
            "raster.png",
            "--report",
            "./raster.png",
        ])
        .current_dir(&dir)
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("must be different paths"),
        "stderr: {stderr}"
    );
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    let status = cli().args(["paint", "x"]).output().expect("run spike-viz");
    assert_eq!(status.status.code(), Some(2));
}

#[test]
fn unknown_option_is_a_usage_error() {
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--bogus",
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(stderr.contains("unknown option"), "stderr: {stderr}");
}

#[test]
fn missing_case_dir_fails_loud() {
    let dir = temp_dir("cli-missing");
    let out = dir.join("raster.png");
    let status = cli()
        .args([
            "render",
            dir.join("missing-case").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(stderr.contains("not found"), "stderr: {stderr}");
    assert!(!out.exists(), "no PNG on failure");
}

#[test]
fn non_integer_scale_is_a_usage_error() {
    let dir = temp_dir("cli-bad-scale");
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--out",
            dir.join("raster.png").to_str().unwrap(),
            "--scale",
            "abc",
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(2));
}

#[test]
fn zero_scale_fails_loud() {
    let dir = temp_dir("cli-zero-scale");
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--out",
            dir.join("raster.png").to_str().unwrap(),
            "--scale",
            "0",
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("scale must be an integer >= 1"),
        "stderr: {stderr}"
    );
}

#[test]
fn inline_scale_form_works() {
    let dir = temp_dir("cli-inline-scale");
    let out = dir.join("raster.png");
    let status = cli()
        .args([
            "render",
            fixture("rate", "tiny_synthetic").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--scale=4",
        ])
        .output()
        .expect("run spike-viz");
    assert!(
        status.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let (width, height, _, _) = decode_png(&out);
    assert_eq!((width, height), (32, 16));
}

#[test]
fn invalid_meta_case_fails_loud() {
    let dir = temp_dir("cli-bad-meta");
    write_meta(
        &dir.join("meta.json"),
        serde_json::json!({"encoder": "rate"}),
    );
    let status = cli()
        .args([
            "render",
            dir.to_str().unwrap(),
            "--out",
            dir.join("raster.png").to_str().unwrap(),
        ])
        .output()
        .expect("run spike-viz");
    assert_eq!(status.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(stderr.contains("missing required keys"), "stderr: {stderr}");
}
