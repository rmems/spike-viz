use std::path::Path;

use npyz::npz::NpzArchive;
use npyz::{DType, NpyFile, TypeChar};

use crate::error::SpikeIOError;
use crate::events::SpikeEvents;

/// Exact float64 `2^63`; comparisons against it detect int64 overflow the
/// same way the Python loader does (it compares against the exact power of
/// two so a float cannot round `2^63 - 1` up to `2^63` and pass).
const INT64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// Load sparse COO spikes from a `.npz` export.
///
/// Port of the Python `load_sparse` (`src/spike_viz/io.py`). Required
/// arrays: `t`, `neuron_id`. Optional: `amp`.
///
/// Accepted dtypes mirror Python exactly:
///
/// - `t` / `neuron_id`: any integer width, unsigned integers, booleans, or
///   float arrays whose values are all finite **and integral** (cast, never
///   truncated). Complex, string, object, and structured dtypes are rejected.
/// - `amp`: any float width, **signed** integer, or boolean array, cast to
///   `f32`; values that overflow to non-finite `f32` are rejected. Unsigned
///   integers are rejected, mirroring the Python `_as_float32_amp` rules
///   (`dtype.kind` must be `f`/`i`/`b`).
///
/// Raises [`SpikeIOError`] when the file is missing, not `.npz`, the
/// required arrays are absent, or the arrays are invalid. A missing file is
/// an error, never empty spikes.
pub fn load_sparse<P: AsRef<Path>>(path: P) -> Result<SpikeEvents, SpikeIOError> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(SpikeIOError::io(format!(
            "sparse spike file not found: {}",
            path.display()
        )));
    }
    if !path
        .extension()
        .map(|ext| ext.to_ascii_lowercase().to_string_lossy() == "npz")
        .unwrap_or(false)
    {
        return Err(SpikeIOError::Invalid(format!(
            "load_sparse expects a .npz file, got: {}",
            path.display()
        )));
    }

    let mut archive = NpzArchive::open(path).map_err(|e| {
        SpikeIOError::io(format!(
            "failed to read sparse spikes from {}: {e}",
            path.display()
        ))
    })?;
    let mut names: Vec<String> = archive.array_names().map(str::to_string).collect();
    names.sort();

    let t = read_index_member(&mut archive, "t", &names, path)?;
    let neuron_id = read_index_member(&mut archive, "neuron_id", &names, path)?;
    let amp = match archive.by_name("amp") {
        Ok(Some(npy)) => Some(read_amp_array(npy, "amp")?),
        Ok(None) => None,
        Err(e) => {
            return Err(SpikeIOError::io(format!(
                "failed to read sparse spikes from {}: {e}",
                path.display()
            )))
        }
    };

    SpikeEvents::new(t, neuron_id, amp).map_err(|e| {
        SpikeIOError::Invalid(format!("{}: invalid sparse arrays: {e}", path.display()))
    })
}

/// Read a required index array member (`t` or `neuron_id`) of the archive.
///
/// A required member that is absent fails loud with the members that were
/// present; it never reads as empty activity.
fn read_index_member<R>(
    archive: &mut NpzArchive<R>,
    name: &str,
    names: &[String],
    path: &Path,
) -> Result<Vec<i64>, SpikeIOError>
where
    R: std::io::Read + std::io::Seek,
{
    match archive.by_name(name) {
        Ok(Some(npy)) => read_index_array(npy, name),
        Ok(None) => Err(SpikeIOError::Invalid(format!(
            "{}: required arrays 't' and 'neuron_id' missing; found {names:?}",
            path.display()
        ))),
        Err(e) => Err(SpikeIOError::io(format!(
            "failed to read sparse spikes from {}: {e}",
            path.display()
        ))),
    }
}

/// Read an index array (`t` or `neuron_id`) into `i64` without silent
/// truncation, mirroring the Python `_as_int64_indices` rules.
fn read_index_array<R: std::io::Read>(
    npy: NpyFile<R>,
    name: &str,
) -> Result<Vec<i64>, SpikeIOError> {
    require_1d(npy.shape(), name)?;
    let dtype = npy.dtype();
    match &dtype {
        DType::Plain(type_str) => match (type_str.type_char(), type_str.size_field()) {
            (TypeChar::Int, 8) => npy.into_vec::<i64>().map_err(|e| read_failed(name, e)),
            (TypeChar::Int, 4) => widen(npy.into_vec::<i32>().map_err(|e| read_failed(name, e))?),
            (TypeChar::Int, 2) => widen(npy.into_vec::<i16>().map_err(|e| read_failed(name, e))?),
            (TypeChar::Int, 1) => widen(npy.into_vec::<i8>().map_err(|e| read_failed(name, e))?),
            (TypeChar::Uint, 8) => npy
                .into_vec::<u64>()
                .map_err(|e| read_failed(name, e))
                .and_then(|values| {
                    values
                        .into_iter()
                        .map(|value| {
                            i64::try_from(value).map_err(|_| {
                                SpikeIOError::Invalid(format!("{name} values exceed int64 range"))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                }),
            (TypeChar::Uint, 4) => widen(npy.into_vec::<u32>().map_err(|e| read_failed(name, e))?),
            (TypeChar::Uint, 2) => widen(npy.into_vec::<u16>().map_err(|e| read_failed(name, e))?),
            (TypeChar::Uint, 1) => widen(npy.into_vec::<u8>().map_err(|e| read_failed(name, e))?),
            (TypeChar::Bool, 1) => npy
                .into_vec::<bool>()
                .map_err(|e| read_failed(name, e))
                .map(|values| {
                    values
                        .into_iter()
                        .map(|b| if b { 1 } else { 0 })
                        .collect::<Vec<_>>()
                }),
            (TypeChar::Float, 8) => npy
                .into_vec::<f64>()
                .map_err(|e| read_failed(name, e))
                .and_then(|values| floats_to_indices(values, name)),
            (TypeChar::Float, 4) => npy
                .into_vec::<f32>()
                .map_err(|e| read_failed(name, e))
                .and_then(|values| floats_to_indices(values, name)),
            _ => Err(SpikeIOError::Invalid(format!(
                "{name} must be integer-valued, got dtype {type_str}"
            ))),
        },
        DType::Record(_) | DType::Array(..) => Err(SpikeIOError::Invalid(format!(
            "{name} must be integer-valued, got structured dtype {dtype:?}"
        ))),
    }
}

/// Read the optional `amp` array into `f32`, mirroring the Python
/// `_as_float32_amp` rules (real numeric only, finite after the cast).
fn read_amp_array<R: std::io::Read>(npy: NpyFile<R>, name: &str) -> Result<Vec<f32>, SpikeIOError> {
    require_1d(npy.shape(), name)?;
    let dtype = npy.dtype();
    let values: Vec<f32> = match &dtype {
        DType::Plain(type_str) => match (type_str.type_char(), type_str.size_field()) {
            (TypeChar::Float, 4) => npy.into_vec::<f32>().map_err(|e| read_failed(name, e))?,
            (TypeChar::Float, 8) => npy
                .into_vec::<f64>()
                .map_err(|e| read_failed(name, e))?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            (TypeChar::Int, 8) => npy
                .into_vec::<i64>()
                .map_err(|e| read_failed(name, e))?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            (TypeChar::Int, 4) => npy
                .into_vec::<i32>()
                .map_err(|e| read_failed(name, e))?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            (TypeChar::Int, 2) => npy
                .into_vec::<i16>()
                .map_err(|e| read_failed(name, e))?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            (TypeChar::Int, 1) => npy
                .into_vec::<i8>()
                .map_err(|e| read_failed(name, e))?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            (TypeChar::Bool, 1) => npy
                .into_vec::<bool>()
                .map_err(|e| read_failed(name, e))?
                .into_iter()
                .map(|b| if b { 1.0 } else { 0.0 })
                .collect(),
            _ => {
                return Err(SpikeIOError::Invalid(format!(
                    "{name} must be real numeric, got dtype {type_str}"
                )))
            }
        },
        DType::Record(_) | DType::Array(..) => {
            return Err(SpikeIOError::Invalid(format!(
                "{name} must be real numeric, got structured dtype {dtype:?}"
            )))
        }
    };
    // Overflowing f64 -> f32 casts saturate to inf; those must fail loud
    // rather than paint an infinite-intensity pixel.
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(SpikeIOError::Invalid(format!(
            "{name} must be finite (non-finite value at index {index})"
        )));
    }
    Ok(values)
}

fn require_1d(shape: &[u64], name: &str) -> Result<(), SpikeIOError> {
    if shape.len() != 1 {
        return Err(SpikeIOError::Invalid(format!(
            "{name} must be a 1-D array, got shape {shape:?}"
        )));
    }
    Ok(())
}

fn read_failed(name: &str, error: std::io::Error) -> SpikeIOError {
    SpikeIOError::io(format!("{name}: failed to read array: {error}"))
}

fn widen<T>(values: Vec<T>) -> Result<Vec<i64>, SpikeIOError>
where
    T: Into<i64> + Copy,
{
    Ok(values.into_iter().map(|value| value.into()).collect())
}

/// Cast integral-valued floats to `i64`, mirroring the Python rules:
/// reject non-finite, fractional, and out-of-range values.
fn floats_to_indices<I>(values: I, name: &str) -> Result<Vec<i64>, SpikeIOError>
where
    I: IntoIterator,
    I::Item: Into<f64> + Copy,
{
    let mut out = Vec::new();
    for value in values {
        let value: f64 = value.into();
        if !value.is_finite() {
            return Err(SpikeIOError::Invalid(format!("{name} must be finite")));
        }
        if value != value.trunc() {
            return Err(SpikeIOError::Invalid(format!(
                "{name} must be integral indices (got non-integer floats)"
            )));
        }
        if value >= INT64_LIMIT || value < -INT64_LIMIT {
            return Err(SpikeIOError::Invalid(format!(
                "{name} values exceed int64 range"
            )));
        }
        out.push(value as i64);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_floats_cast_to_indices() {
        let values: Vec<f64> = vec![0.0, 1.0, -2.0, 9.0];
        let out = floats_to_indices(values, "t").unwrap();
        assert_eq!(out, vec![0, 1, -2, 9]);
    }

    #[test]
    fn non_finite_floats_rejected() {
        let err = floats_to_indices(vec![0.0f64, f64::NAN], "t").unwrap_err();
        assert!(err.to_string().contains("must be finite"));
    }

    #[test]
    fn fractional_floats_rejected() {
        let err = floats_to_indices(vec![1.5f32], "t").unwrap_err();
        assert!(err.to_string().contains("non-integer floats"));
    }

    #[test]
    fn oversized_floats_rejected() {
        let err = floats_to_indices(vec![u64::MAX as f64], "t").unwrap_err();
        assert!(err.to_string().contains("exceed int64 range"));
    }
}
