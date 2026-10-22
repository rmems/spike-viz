use std::path::Path;

use serde_json::Value;

use crate::error::SpikeIOError;

/// Required `meta.json` keys for a contract-complete case directory
/// (see `docs/axon-encoder-export.md`).
pub const REQUIRED_META_KEYS: &[&str] = &[
    "dt_seconds",
    "encoder",
    "n_neurons",
    "n_steps",
    "schema_version",
    "seed",
];

/// Supported export contract versions; unknown versions fail load.
pub const SUPPORTED_SCHEMA_VERSIONS: &[&str] = &["1.0"];

/// Validated `meta.json` for one export case.
///
/// Required fields are typed; the raw JSON object is preserved in
/// [`ExportMeta::raw`] for forward-compatible, exporter-specific fields.
#[derive(Debug, Clone)]
pub struct ExportMeta {
    /// Contract version; only `"1.0"` is supported.
    pub schema_version: String,
    /// Encoder id, e.g. `"rate"`, `"poisson"`.
    pub encoder: String,
    /// Seconds per displayed step (the export sampling convention).
    pub dt_seconds: f64,
    /// RNG seed used for the export.
    pub seed: u64,
    /// Geometry `N`; must be `>= 1` even if spikes are empty.
    pub n_neurons: u64,
    /// Geometry `T`; must be `>= 1` even if spikes are empty.
    pub n_steps: u64,
    /// Commit of axon-encoder when known.
    pub axon_encoder_git_sha: Option<String>,
    /// `true` when the export is not from real axon-encoder.
    pub synthetic: Option<bool>,
    /// The full raw `meta.json` object.
    pub raw: Value,
}

/// Load and validate `meta.json` for an export case.
///
/// Port of the Python `load_meta` (`src/spike_viz/export.py`): presence
/// first, then strict types (`bool` is never an integer, floats are never
/// integers), geometry `>= 1`, finite positive `dt_seconds`, non-empty
/// `encoder`, and `schema_version` restricted to
/// [`SUPPORTED_SCHEMA_VERSIONS`].
///
/// `dt_seconds` keeps its seconds-per-displayed-step convention; it is not
/// necessarily an encoder-owned physical timebase. Rust stores `seed` as
/// `u64`, so seeds above `u64::MAX` (accepted by Python's arbitrary-precision
/// ints) fail loud instead of being reinterpreted.
pub fn load_meta<P: AsRef<Path>>(path: P) -> Result<ExportMeta, SpikeIOError> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(SpikeIOError::io(format!(
            "meta.json not found: {}",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path).map_err(|e| {
        SpikeIOError::io(format!(
            "failed to read meta.json at {}: {e}",
            path.display()
        ))
    })?;
    let raw: Value = serde_json::from_str(&text)
        .map_err(|e| SpikeIOError::Invalid(format!("invalid JSON in {}: {e}", path.display())))?;
    let obj = match &raw {
        Value::Object(map) => map,
        _ => {
            return Err(SpikeIOError::Invalid(format!(
                "{}: root must be a JSON object",
                path.display()
            )))
        }
    };

    let missing: Vec<&str> = REQUIRED_META_KEYS
        .iter()
        .filter(|key| !obj.contains_key(**key))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(SpikeIOError::Invalid(format!(
            "{}: missing required keys: {missing:?}",
            path.display()
        )));
    }

    let n_neurons = require_positive_int(obj, "n_neurons", path)?;
    let n_steps = require_positive_int(obj, "n_steps", path)?;
    let seed = require_seed(obj, path)?;
    let dt_seconds = require_dt_seconds(obj, path)?;
    let encoder = require_non_empty_string(obj, "encoder", path)?;
    let schema_version = require_schema_version(obj, path)?;

    // Optional provenance fields tolerate absence and explicit null, but a
    // present value of the wrong type must fail loud — malformed
    // provenance must not silently read as absent.
    let axon_encoder_git_sha = optional_string(obj, "axon_encoder_git_sha", path)?;
    let synthetic = optional_bool(obj, "synthetic", path)?;

    Ok(ExportMeta {
        schema_version,
        encoder,
        dt_seconds,
        seed,
        n_neurons,
        n_steps,
        axon_encoder_git_sha,
        synthetic,
        raw,
    })
}

fn require_positive_int(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    path: &Path,
) -> Result<u64, SpikeIOError> {
    match &obj[key] {
        Value::Number(number) if number.is_u64() => {
            let value = number.as_u64().expect("checked is_u64");
            if value < 1 {
                Err(SpikeIOError::Invalid(format!(
                    "{}: '{key}' must be >= 1, got {value}",
                    path.display()
                )))
            } else {
                Ok(value)
            }
        }
        Value::Number(number) if number.is_i64() => {
            let value = number.as_i64().expect("checked is_i64");
            if value < 1 {
                Err(SpikeIOError::Invalid(format!(
                    "{}: '{key}' must be >= 1, got {value}",
                    path.display()
                )))
            } else {
                Ok(value as u64)
            }
        }
        _ => Err(SpikeIOError::Invalid(format!(
            "{}: '{key}' must be an integer",
            path.display()
        ))),
    }
}

fn require_seed(obj: &serde_json::Map<String, Value>, path: &Path) -> Result<u64, SpikeIOError> {
    match &obj["seed"] {
        Value::Number(number) if number.is_u64() => Ok(number.as_u64().expect("checked is_u64")),
        Value::Number(number) if number.is_i64() => {
            let value = number.as_i64().expect("checked is_i64");
            if value < 0 {
                Err(SpikeIOError::Invalid(format!(
                    "{}: 'seed' must be >= 0, got {value}",
                    path.display()
                )))
            } else {
                Ok(value as u64)
            }
        }
        _ => Err(SpikeIOError::Invalid(format!(
            "{}: 'seed' must be an integer",
            path.display()
        ))),
    }
}

/// Optional string field: absent or `null` → `None`, string → `Some`.
///
/// A present value of any other type is rejected: malformed provenance
/// must fail loud rather than silently read as absent.
fn optional_string(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    path: &Path,
) -> Result<Option<String>, SpikeIOError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(SpikeIOError::Invalid(format!(
            "{}: '{key}' must be a string or null, got {other}",
            path.display()
        ))),
    }
}

/// Optional boolean field: absent or `null` → `None`, boolean → `Some`.
/// See [`optional_string`] for the fail-loud rule.
fn optional_bool(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    path: &Path,
) -> Result<Option<bool>, SpikeIOError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(other) => Err(SpikeIOError::Invalid(format!(
            "{}: '{key}' must be a boolean or null, got {other}",
            path.display()
        ))),
    }
}

fn require_dt_seconds(
    obj: &serde_json::Map<String, Value>,
    path: &Path,
) -> Result<f64, SpikeIOError> {
    let value = &obj["dt_seconds"];
    let dt = match value {
        Value::Number(number) => match number.as_u64() {
            Some(value) => value as f64,
            None => match number.as_i64() {
                Some(value) => value as f64,
                None => number.as_f64().expect("serde_json numbers are finite"),
            },
        },
        _ => {
            return Err(SpikeIOError::Invalid(format!(
                "{}: 'dt_seconds' must be a number",
                path.display()
            )))
        }
    };
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SpikeIOError::Invalid(format!(
            "{}: 'dt_seconds' must be finite and > 0, got {value}",
            path.display()
        )));
    }
    Ok(dt)
}

fn require_non_empty_string(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    path: &Path,
) -> Result<String, SpikeIOError> {
    match &obj[key] {
        Value::String(value) if !value.is_empty() => Ok(value.clone()),
        _ => Err(SpikeIOError::Invalid(format!(
            "{}: '{key}' must be a non-empty string",
            path.display()
        ))),
    }
}

fn require_schema_version(
    obj: &serde_json::Map<String, Value>,
    path: &Path,
) -> Result<String, SpikeIOError> {
    let value = &obj["schema_version"];
    let version = match value {
        Value::String(version) if !version.is_empty() => version.clone(),
        _ => {
            return Err(SpikeIOError::Invalid(format!(
                "{}: 'schema_version' must be a non-empty string",
                path.display()
            )))
        }
    };
    if !SUPPORTED_SCHEMA_VERSIONS.contains(&version.as_str()) {
        return Err(SpikeIOError::Invalid(format!(
            "{}: unsupported schema_version {value:?}; supported: {SUPPORTED_SCHEMA_VERSIONS:?}",
            path.display()
        )));
    }
    Ok(version)
}
