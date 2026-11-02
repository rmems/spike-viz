//! Strict JSON field access for the bundle manifest.
//!
//! Known fields are typed strictly (a `bool` is never an integer, `8.0` is
//! never an integer, a string is never a number). Keys a [`Fields`] reader
//! did not consume are reported with their JSON pointer, so unknown fields
//! from newer producers are kept apart instead of silently ignored.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::BundleError;

/// Reader over one JSON object that tracks the keys it consumed.
pub(crate) struct Fields<'a> {
    obj: &'a Map<String, Value>,
    path: String,
    consumed: Vec<&'static str>,
}

impl<'a> Fields<'a> {
    /// `value` must be a JSON object; `path` is its JSON pointer.
    pub(crate) fn new(value: &'a Value, path: String) -> Result<Self, BundleError> {
        match value {
            Value::Object(obj) => Ok(Self {
                obj,
                path,
                consumed: Vec::new(),
            }),
            other => Err(invalid(
                &path,
                format!("must be a JSON object, got {}", type_name(other)),
            )),
        }
    }

    /// JSON pointer of `key` inside this object.
    pub(crate) fn child(&self, key: &str) -> String {
        child_pointer(&self.path, key)
    }

    /// A required key. `null` counts as present: nullable fields are
    /// required keys whose explicit `null` means "not known".
    pub(crate) fn required(&mut self, key: &'static str) -> Result<&'a Value, BundleError> {
        self.consumed.push(key);
        self.obj
            .get(key)
            .ok_or_else(|| invalid(&self.path, format!("missing required key {key:?}")))
    }

    /// An optional key: `None` when absent.
    pub(crate) fn optional(&mut self, key: &'static str) -> Option<&'a Value> {
        self.consumed.push(key);
        self.obj.get(key)
    }

    /// Record every key this reader did not consume as unrecognized.
    pub(crate) fn finish(self, unrecognized: &mut BTreeMap<String, Value>) {
        for (key, value) in self.obj {
            if !self.consumed.contains(&key.as_str()) {
                unrecognized.insert(child_pointer(&self.path, key), value.clone());
            }
        }
    }
}

/// JSON pointer (RFC 6901) of `key` under `parent`.
pub(crate) fn child_pointer(parent: &str, key: &str) -> String {
    format!("{parent}/{}", key.replace('~', "~0").replace('/', "~1"))
}

/// `Invalid` error located at a JSON pointer (`""` is the document root).
pub(crate) fn invalid(path: &str, message: impl std::fmt::Display) -> BundleError {
    let at = if path.is_empty() { "(root)" } else { path };
    BundleError::Invalid(format!("{at}: {message}"))
}

pub(crate) fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// A string with at least one non-whitespace character.
pub(crate) fn text(value: &Value, path: &str) -> Result<String, BundleError> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Ok(text.clone()),
        Value::String(_) => Err(invalid(path, "must be a non-blank string")),
        other => Err(invalid(
            path,
            format!("must be a non-blank string, got {}", type_name(other)),
        )),
    }
}

/// `null` → `None`, otherwise a non-blank string.
pub(crate) fn nullable_text(value: &Value, path: &str) -> Result<Option<String>, BundleError> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) if !text.trim().is_empty() => Ok(Some(text.clone())),
        Value::String(_) => Err(invalid(
            path,
            "must be null or a non-blank string (state an unknown value as null)",
        )),
        other => Err(invalid(
            path,
            format!(
                "must be null or a non-blank string, got {}",
                type_name(other)
            ),
        )),
    }
}

/// `null` → `None`, otherwise a boolean.
pub(crate) fn nullable_bool(value: &Value, path: &str) -> Result<Option<bool>, BundleError> {
    match value {
        Value::Null => Ok(None),
        Value::Bool(flag) => Ok(Some(*flag)),
        other => Err(invalid(
            path,
            format!("must be null or a boolean, got {}", type_name(other)),
        )),
    }
}

/// A non-negative JSON integer (never a float, never a boolean).
pub(crate) fn uint(value: &Value, path: &str) -> Result<u64, BundleError> {
    match value {
        Value::Number(number) if number.is_u64() => Ok(number.as_u64().expect("checked is_u64")),
        Value::Number(number) if number.is_i64() => Err(invalid(
            path,
            format!("must be a non-negative integer, got {number}"),
        )),
        other => Err(invalid(
            path,
            format!("must be a non-negative integer, got {}", describe(other)),
        )),
    }
}

/// `null` → `None`, otherwise a non-negative integer.
pub(crate) fn nullable_uint(value: &Value, path: &str) -> Result<Option<u64>, BundleError> {
    match value {
        Value::Null => Ok(None),
        other => uint(other, path).map(Some),
    }
}

/// An integer `>= 1`.
pub(crate) fn positive_uint(value: &Value, path: &str) -> Result<u64, BundleError> {
    let number = uint(value, path)?;
    if number == 0 {
        return Err(invalid(path, "must be >= 1"));
    }
    Ok(number)
}

/// A finite number `> 0` (integers are accepted as exact floats).
pub(crate) fn positive_number(value: &Value, path: &str) -> Result<f64, BundleError> {
    let number = match value {
        Value::Number(number) => number.as_f64().expect("serde_json numbers are finite"),
        other => {
            return Err(invalid(
                path,
                format!("must be a number, got {}", type_name(other)),
            ))
        }
    };
    if !number.is_finite() || number <= 0.0 {
        return Err(invalid(
            path,
            format!("must be finite and > 0, got {value}"),
        ));
    }
    Ok(number)
}

/// An array of non-blank strings (possibly empty).
pub(crate) fn text_list(value: &Value, path: &str) -> Result<Vec<String>, BundleError> {
    match value {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| text(item, &format!("{path}/{index}")))
            .collect(),
        other => Err(invalid(
            path,
            format!("must be an array of strings, got {}", type_name(other)),
        )),
    }
}

/// An array of non-negative integers (possibly empty).
pub(crate) fn uint_list(value: &Value, path: &str) -> Result<Vec<u64>, BundleError> {
    match value {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| uint(item, &format!("{path}/{index}")))
            .collect(),
        other => Err(invalid(
            path,
            format!("must be an array of integers, got {}", type_name(other)),
        )),
    }
}

/// Type name, plus the value for numbers (so `8.0` reads as a float).
fn describe(value: &Value) -> String {
    match value {
        Value::Number(number) => format!("non-integer number {number}"),
        other => type_name(other).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pointer_escapes_tilde_and_slash() {
        assert_eq!(child_pointer("/channels", "a/b~c"), "/channels/a~1b~0c");
    }

    #[test]
    fn uint_is_strict() {
        assert_eq!(uint(&json!(3), "/n").unwrap(), 3);
        for bad in [json!(3.0), json!(-1), json!(true), json!("3"), json!(null)] {
            assert!(uint(&bad, "/n").is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn nullable_text_rejects_blank() {
        assert_eq!(nullable_text(&json!(null), "/v").unwrap(), None);
        assert_eq!(
            nullable_text(&json!("1.2"), "/v").unwrap().as_deref(),
            Some("1.2")
        );
        assert!(nullable_text(&json!("  "), "/v").is_err());
        assert!(nullable_text(&json!(12), "/v").is_err());
    }

    #[test]
    fn fields_report_unconsumed_keys() {
        let value = json!({"known": 1, "future": {"x": 2}});
        let mut fields = Fields::new(&value, "/p".to_string()).unwrap();
        assert!(fields.required("known").is_ok());
        assert!(fields.required("absent").is_err());
        let mut unrecognized = BTreeMap::new();
        fields.finish(&mut unrecognized);
        assert_eq!(unrecognized.len(), 1);
        assert_eq!(unrecognized["/p/future"], json!({"x": 2}));
    }
}
