//! `load_sparse` contract tests: `.npz` reading, dtype rules, and
//! loud failure on invalid data (never invented spikes).

mod common;

use common::*;
use spike_viz_core::{load_sparse, SpikeIOError};

fn i64_data(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_data(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_data(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Valid index pair members for a raw archive.
fn index_pair(t_descr: &str, t_data: &[u8], n_descr: &str, n_data: &[u8]) -> Vec<RawMember> {
    vec![
        RawMember {
            name: "t".to_string(),
            descr: t_descr.to_string(),
            shape: vec![(t_data.len() / 8) as u64],
            data: t_data.to_vec(),
        },
        RawMember {
            name: "neuron_id".to_string(),
            descr: n_descr.to_string(),
            shape: vec![(n_data.len() / 8) as u64],
            data: n_data.to_vec(),
        },
    ]
}

#[test]
fn loads_golden_fixture_arrays() {
    let events = load_sparse(fixture("rate", "tiny_synthetic").join("spikes.npz")).unwrap();
    assert_eq!(events.t(), &[0, 1, 1, 3, 5, 5, 7]);
    assert_eq!(events.neuron_id(), &[0, 0, 2, 1, 0, 3, 2]);
    assert_eq!(events.amp(), Some(&[1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0][..]));
    assert_eq!(events.len(), 7);
}

#[test]
fn amp_absent_loads_as_none() {
    let dir = temp_dir("amp-absent");
    let path = dir.join("spikes.npz");
    write_npz(&path, &[0, 1], &[1, 0], None);
    let events = load_sparse(&path).unwrap();
    assert!(events.amp().is_none());
    // Iterator still yields unit amplitudes.
    assert_eq!(
        events.iter().collect::<Vec<_>>(),
        vec![(0, 1, 1.0), (1, 0, 1.0)]
    );
}

#[test]
fn missing_file_fails_loud() {
    let dir = temp_dir("missing-file");
    let err = load_sparse(dir.join("spikes.npz")).unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

#[test]
fn wrong_extension_fails_loud() {
    let dir = temp_dir("wrong-ext");
    let path = dir.join("spikes.npy");
    std::fs::write(&path, b"not an npz").expect("write fake npy");
    let err = load_sparse(&path).unwrap_err();
    assert!(
        err.to_string().contains("expects a .npz file"),
        "got: {err}"
    );
}

#[test]
fn missing_required_array_fails_loud() {
    let dir = temp_dir("missing-key");
    // Archive with `t` only: `neuron_id` is absent and must fail loud.
    let partial = dir.join("partial.npz");
    write_npz_raw_member(&partial, "t", "<i8", &[2], &i64_data(&[0, 1]));
    let err = load_sparse(&partial).unwrap_err();
    assert!(
        err.to_string()
            .contains("required arrays 't' and 'neuron_id' missing"),
        "got: {err}"
    );
}

#[test]
fn integral_float_indices_cast() {
    let dir = temp_dir("float-integral");
    let path = dir.join("spikes.npz");
    let members = index_pair("<f8", &f64_data(&[0.0, 1.0]), "<i8", &i64_data(&[0, 0]));
    write_npz_raw(&path, &members);
    let events = load_sparse(&path).unwrap();
    assert_eq!(events.t(), &[0, 1]);
}

#[test]
fn fractional_float_indices_rejected() {
    let dir = temp_dir("float-frac");
    let path = dir.join("spikes.npz");
    let members = index_pair("<f8", &f64_data(&[0.5, 1.5]), "<i8", &i64_data(&[0, 0]));
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("non-integer floats"), "got: {err}");
}

#[test]
fn non_finite_float_indices_rejected() {
    let dir = temp_dir("float-inf");
    let path = dir.join("spikes.npz");
    let members = index_pair("<f8", &f64_data(&[f64::INFINITY]), "<i8", &i64_data(&[0]));
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("must be finite"), "got: {err}");
}

#[test]
fn oversized_float_indices_rejected() {
    let dir = temp_dir("float-huge");
    let path = dir.join("spikes.npz");
    let members = index_pair("<f8", &f64_data(&[2f64.powi(63)]), "<i8", &i64_data(&[0]));
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("exceed int64 range"), "got: {err}");
}

#[test]
fn uint8_indices_widen() {
    let dir = temp_dir("uint8");
    let path = dir.join("spikes.npz");
    let members = vec![
        RawMember {
            name: "t".into(),
            descr: "|u1".into(),
            shape: vec![2],
            data: vec![7, 9],
        },
        RawMember {
            name: "neuron_id".into(),
            descr: "|u1".into(),
            shape: vec![2],
            data: vec![0, 0],
        },
    ];
    write_npz_raw(&path, &members);
    let events = load_sparse(&path).unwrap();
    assert_eq!(events.t(), &[7, 9]);
}

#[test]
fn uint64_indices_overflow_rejected() {
    let dir = temp_dir("uint64-overflow");
    let path = dir.join("spikes.npz");
    let members = index_pair("<u8", &u64::MAX.to_le_bytes(), "<i8", &i64_data(&[0]));
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("exceed int64 range"), "got: {err}");
}

#[test]
fn bool_indices_cast() {
    let dir = temp_dir("bool-idx");
    let path = dir.join("spikes.npz");
    let members = vec![
        RawMember {
            name: "t".into(),
            descr: "|b1".into(),
            shape: vec![2],
            data: vec![1, 0],
        },
        RawMember {
            name: "neuron_id".into(),
            descr: "|b1".into(),
            shape: vec![2],
            data: vec![1, 0],
        },
    ];
    write_npz_raw(&path, &members);
    let events = load_sparse(&path).unwrap();
    assert_eq!(events.t(), &[1, 0]);
    assert_eq!(events.neuron_id(), &[1, 0]);
}

#[test]
fn object_dtype_rejected() {
    // numpy writes pickled object arrays with descr '|O'; the contract has
    // no pickle channel, so loading must fail loud in any language.
    let dir = temp_dir("object-dtype");
    let path = dir.join("spikes.npz");
    write_npz_raw_member(&path, "t", "|O", &[1], &[]);
    let err = load_sparse(&path).unwrap_err();
    assert!(!err.to_string().is_empty(), "object dtype must not load");
}

#[test]
fn unicode_dtype_rejected_as_indices() {
    let dir = temp_dir("unicode-dtype");
    let path = dir.join("spikes.npz");
    write_npz_raw_member(&path, "t", "<U4", &[1], &[0; 16]);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("integer-valued"), "got: {err}");
}

#[test]
fn complex_amp_rejected() {
    let dir = temp_dir("complex-amp");
    let path = dir.join("spikes.npz");
    let mut members = index_pair("<i8", &i64_data(&[1]), "<i8", &i64_data(&[0]));
    members.push(RawMember {
        name: "amp".into(),
        descr: "<c8".into(),
        shape: vec![1],
        data: vec![0; 8],
    });
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("real numeric"), "got: {err}");
}

#[test]
fn int_amp_casts_to_f32() {
    let dir = temp_dir("int-amp");
    let path = dir.join("spikes_int_amp.npz");
    let mut members = index_pair("<i8", &i64_data(&[0]), "<i8", &i64_data(&[1]));
    members.push(RawMember {
        name: "amp".into(),
        descr: "<i4".into(),
        shape: vec![1],
        data: 2i32.to_le_bytes().to_vec(),
    });
    write_npz_raw(&path, &members);
    let events = load_sparse(&path).unwrap();
    assert_eq!(events.amp(), Some(&[2.0f32][..]));
}

#[test]
fn float64_amp_overflow_rejected() {
    let dir = temp_dir("f64-amp-overflow");
    let path = dir.join("spikes.npz");
    let mut members = index_pair("<i8", &i64_data(&[0]), "<i8", &i64_data(&[0]));
    members.push(RawMember {
        name: "amp".into(),
        descr: "<f8".into(),
        shape: vec![1],
        data: f64_data(&[1e300]),
    });
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("amp must be finite"), "got: {err}");
}

#[test]
fn nan_amp_rejected() {
    let dir = temp_dir("nan-amp");
    let path = dir.join("spikes.npz");
    let mut members = index_pair("<i8", &i64_data(&[0]), "<i8", &i64_data(&[0]));
    members.push(RawMember {
        name: "amp".into(),
        descr: "<f4".into(),
        shape: vec![1],
        data: f32_data(&[f32::NAN]),
    });
    write_npz_raw(&path, &members);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("amp must be finite"), "got: {err}");
}

#[test]
fn two_dimensional_t_rejected() {
    let dir = temp_dir("two-d");
    let path = dir.join("spikes.npz");
    write_npz_raw_member(&path, "t", "<i8", &[1, 2], &[0; 16]);
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("1-D"), "got: {err}");
}

#[test]
fn zero_dimensional_t_rejected() {
    let dir = temp_dir("zero-d");
    let path = dir.join("spikes.npz");
    write_npz_raw_member(&path, "t", "<i8", &[], &5i64.to_le_bytes());
    let err = load_sparse(&path).unwrap_err();
    assert!(err.to_string().contains("1-D"), "got: {err}");
}

#[test]
fn length_mismatch_rejected() {
    let dir = temp_dir("len-mismatch");
    let path = dir.join("spikes.npz");
    write_npz(&path, &[0, 1], &[0], None);
    let err = load_sparse(&path).unwrap_err();
    assert!(
        err.to_string().contains("t length 2 != neuron_id length 1"),
        "got: {err}"
    );
}

#[test]
fn amp_length_mismatch_rejected() {
    let dir = temp_dir("amp-len");
    let path = dir.join("spikes.npz");
    write_npz(&path, &[0, 1], &[0, 0], Some(&[1.0]));
    let err = load_sparse(&path).unwrap_err();
    assert!(
        err.to_string()
            .contains("amp must be 1-D and match t length"),
        "got: {err}"
    );
}

#[test]
fn empty_events_load() {
    let dir = temp_dir("empty");
    let path = dir.join("spikes.npz");
    write_npz(&path, &[], &[], Some(&[]));
    let events = load_sparse(&path).unwrap();
    assert_eq!(events.len(), 0);
    assert!(events.is_empty());
}

#[test]
fn error_type_is_spike_io_error() {
    let dir = temp_dir("err-type");
    let err: SpikeIOError = load_sparse(dir.join("spikes.npz")).unwrap_err();
    let _: &dyn std::error::Error = &err;
}
