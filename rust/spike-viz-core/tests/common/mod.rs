//! Shared helpers for the integration tests.
//!
//! Fixtures live in the repository (`fixtures/axon-encoder/...`) so the
//! Rust core is exercised against the exact on-disk contract Python uses;
//! hand-built archives cover the invalid cases.

#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use npyz::npz::NpzWriter;
use npyz::WriterBuilder;

/// Repository root (holds `fixtures/`, `src/`, `rust/`).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root resolves")
}

/// Path of a checked-in export case.
pub fn fixture(encoder: &str, case: &str) -> PathBuf {
    repo_root()
        .join("fixtures")
        .join("axon-encoder")
        .join(encoder)
        .join(case)
}

/// Every checked-in export case (directories holding `meta.json`).
pub fn all_fixture_cases() -> Vec<PathBuf> {
    let root = repo_root().join("fixtures").join("axon-encoder");
    let mut cases = Vec::new();
    for encoder in fs::read_dir(&root).expect("fixtures/axon-encoder exists") {
        let encoder = encoder.expect("readable encoder dir");
        if !encoder.path().is_dir() {
            continue;
        }
        for case in fs::read_dir(encoder.path()).expect("readable encoder case dir") {
            let case = case.expect("readable case dir");
            if case.path().join("meta.json").is_file() {
                cases.push(case.path());
            }
        }
    }
    cases.sort();
    cases
}

/// Unique scratch directory for a test (under cargo's test tmpdir).
pub fn temp_dir(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{tag}-{id}"));
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Write a schema-valid `spikes.npz` with `t`, `neuron_id`, optional `amp`.
pub fn write_npz(path: &Path, t: &[i64], neuron_id: &[i64], amp: Option<&[f32]>) {
    let mut writer = NpzWriter::create(path).expect("create npz");
    write_member(&mut writer, "t", t);
    write_member(&mut writer, "neuron_id", neuron_id);
    if let Some(amp) = amp {
        write_member(&mut writer, "amp", amp);
    }
}

fn write_member<T>(
    writer: &mut NpzWriter<std::io::BufWriter<std::fs::File>>,
    name: &str,
    values: &[T],
) where
    T: npyz::AutoSerialize + Clone,
    npyz::WriteOptions<T>: npyz::WriterBuilder<T>,
{
    let options = npyz::zip::write::FileOptions::default();
    writer
        .array::<T>(name, options)
        .expect("begin npz member")
        .default_dtype()
        .shape(&[values.len() as u64])
        .begin_nd()
        .expect("begin npy member")
        .extend(values.iter().cloned())
        .expect("write npy member");
}

/// One raw `.npz` member for [`write_npz_raw`] (dtype edge cases that numpy
/// can produce but the contract must reject).
pub struct RawMember {
    pub name: String,
    pub descr: String,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Write a `.npz` archive with hand-built `.npy` members (dtype edge cases:
/// object pickles, unicode, complex, 2-D shapes) that the loader must reject.
pub fn write_npz_raw(path: &Path, members: &[RawMember]) {
    let mut writer = NpzWriter::create(path).expect("create raw npz");
    for member in members {
        let npy = npy_bytes(&member.descr, false, &member.shape, &member.data);
        let options = npyz::zip::write::FileOptions::default();
        let zip = writer.zip_writer();
        zip.start_file(format!("{}.npy", member.name), options)
            .expect("start raw member");
        zip.write_all(&npy).expect("write raw member");
    }
}

/// Write a `.npz` with a single hand-built `.npy` member.
pub fn write_npz_raw_member(path: &Path, member: &str, descr: &str, shape: &[u64], data: &[u8]) {
    write_npz_raw(
        path,
        &[RawMember {
            name: member.to_string(),
            descr: descr.to_string(),
            shape: shape.to_vec(),
            data: data.to_vec(),
        }],
    );
}

/// Build minimal `.npy` file bytes (format version 1.0).
pub fn npy_bytes(descr: &str, fortran_order: bool, shape: &[u64], data: &[u8]) -> Vec<u8> {
    // The header is a Python literal; numpy writes capitalized booleans.
    let order = if fortran_order { "True" } else { "False" };
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else {
        format!(
            "({},)",
            shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': {order}, 'shape': {shape_str}, }}");
    // numpy pads the header so the data section starts 64-byte aligned.
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - (unpadded % 64)) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');

    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend(b"\x93NUMPY");
    out.extend([0x01, 0x00]);
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(data);
    out
}

/// Write a `meta.json` (any JSON value; tests pass invalid ones on purpose).
pub fn write_meta(path: &Path, value: serde_json::Value) {
    fs::write(path, serde_json::to_string(&value).expect("serialize meta"))
        .expect("write meta.json");
}

/// Valid contract-complete `meta.json` for synthetic test cases.
pub fn valid_meta() -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.0",
        "encoder": "rate",
        "dt_seconds": 0.001,
        "seed": 0,
        "n_neurons": 2,
        "n_steps": 2,
        "synthetic": true,
    })
}

/// Write a complete export case directory.
pub fn write_case(
    dir: &Path,
    meta: &serde_json::Value,
    t: &[i64],
    neuron_id: &[i64],
    amp: Option<&[f32]>,
) {
    fs::create_dir_all(dir).expect("case dir");
    write_meta(&dir.join("meta.json"), meta.clone());
    write_npz(&dir.join("spikes.npz"), t, neuron_id, amp);
}

/// Decode a PNG back into `(width, height, rgb bytes)` for round-trip checks.
pub fn decode_png(path: &Path) -> (u32, u32, Vec<u8>, png::ColorType) {
    let file = fs::File::open(path).expect("png readable");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("png header");
    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).expect("png pixels");
    (
        info.width,
        info.height,
        buffer[..info.buffer_size()].to_vec(),
        info.color_type,
    )
}

/// Grayscale value of an RGB-decoded pixel (channels must agree).
pub fn gray_at(rgb: &[u8], width: usize, row: usize, col: usize) -> u8 {
    let index = (row * width + col) * 3;
    let (r, g, b) = (rgb[index], rgb[index + 1], rgb[index + 2]);
    assert_eq!(r, g);
    assert_eq!(g, b);
    r
}
