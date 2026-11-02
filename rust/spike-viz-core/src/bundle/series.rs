//! Dense time series sampled on the bundle's step axis.

use std::collections::BTreeSet;

use super::manifest::Dimensions;
use super::BundleError;

/// What each column of a [`Series`] means.
///
/// Column semantics are always explicit: a column is either a neuron on the
/// bundle's neuron axis or a named signal that is **not** a neuron index
/// (e.g. a modulator or a telemetry sensor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Columns {
    /// Column `j` is neuron `ids[j]` (`< n_neurons`, no duplicates).
    /// Lists may be a subset of the population, in any order.
    Neurons(Vec<u64>),
    /// Column `j` is the named signal `labels[j]` (non-blank, no duplicates).
    Labels(Vec<String>),
}

impl Columns {
    /// Every neuron of an `n_neurons` population, in id order.
    pub fn all_neurons(n_neurons: u64) -> Self {
        Columns::Neurons((0..n_neurons).collect())
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        match self {
            Columns::Neurons(ids) => ids.len(),
            Columns::Labels(labels) => labels.len(),
        }
    }

    /// True when there are no columns (never valid in a bundle).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Structural validity; with `n_neurons`, also neuron-id bounds.
    pub(crate) fn validate(&self, n_neurons: Option<u64>) -> Result<(), String> {
        match self {
            Columns::Neurons(ids) => {
                if ids.is_empty() {
                    return Err("neuron_ids must name at least one neuron".to_string());
                }
                let mut seen = BTreeSet::new();
                for id in ids {
                    if !seen.insert(*id) {
                        return Err(format!("neuron_ids contains duplicate id {id}"));
                    }
                    if let Some(n_neurons) = n_neurons {
                        if *id >= n_neurons {
                            return Err(format!(
                                "neuron_ids contains {id}, out of range for n_neurons={n_neurons}"
                            ));
                        }
                    }
                }
            }
            Columns::Labels(labels) => {
                if labels.is_empty() {
                    return Err("labels must name at least one signal".to_string());
                }
                let mut seen = BTreeSet::new();
                for label in labels {
                    if label.trim().is_empty() {
                        return Err("labels must be non-blank strings".to_string());
                    }
                    if !seen.insert(label.as_str()) {
                        return Err(format!("labels contains duplicate {label:?}"));
                    }
                }
            }
        }
        Ok(())
    }
}

/// A dense time series: `values[i, j]` is column `j` sampled at step `t[i]`.
///
/// Time semantics are explicit: `t` holds **step indices on the bundle's
/// shared step axis** (the same axis as spike `t`), strictly increasing.
/// Samples are instantaneous readings at those steps; nothing is defined
/// between samples, so a renderer must not interpolate silently. Values are
/// stored row-major (`[n_samples, n_columns]`) and must be finite — a
/// missing reading is an omitted sample, never a NaN.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    t: Vec<i64>,
    values: Vec<f64>,
    columns: Columns,
    unit: Option<String>,
}

impl Series {
    /// Build a validated series. `values` is row-major
    /// `[t.len(), columns.len()]`; `unit` (e.g. `"mV"`) is `None` when the
    /// producer did not state one.
    ///
    /// Bounds against a bundle's dimensions are checked when the series is
    /// added to a [`super::BundleBuilder`] or loaded from a bundle.
    pub fn new(
        t: Vec<i64>,
        values: Vec<f64>,
        columns: Columns,
        unit: Option<String>,
    ) -> Result<Self, BundleError> {
        columns.validate(None).map_err(BundleError::Invalid)?;
        let expected = t
            .len()
            .checked_mul(columns.len())
            .ok_or_else(|| BundleError::Invalid("series shape overflows usize".to_string()))?;
        if values.len() != expected {
            return Err(BundleError::Invalid(format!(
                "series values length {} != n_samples {} * n_columns {}",
                values.len(),
                t.len(),
                columns.len()
            )));
        }
        if let Some(first) = t.first() {
            if *first < 0 {
                return Err(BundleError::Invalid(format!(
                    "series t must be >= 0, got {first}"
                )));
            }
        }
        if let Some(index) = t.windows(2).position(|pair| pair[1] <= pair[0]) {
            return Err(BundleError::Invalid(format!(
                "series t must be strictly increasing (t[{}]={} then t[{}]={})",
                index,
                t[index],
                index + 1,
                t[index + 1]
            )));
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(BundleError::Invalid(format!(
                "series values must be finite (non-finite value at flat index {index})"
            )));
        }
        if let Some(unit) = &unit {
            if unit.trim().is_empty() {
                return Err(BundleError::Invalid(
                    "series unit must be non-blank (use None when unknown)".to_string(),
                ));
            }
        }
        Ok(Self {
            t,
            values,
            columns,
            unit,
        })
    }

    /// Sample step indices on the bundle's step axis (strictly increasing).
    pub fn t(&self) -> &[i64] {
        &self.t
    }

    /// Row-major values, `[n_samples, n_columns]`.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The values of sample `index` (one per column).
    pub fn sample(&self, index: usize) -> &[f64] {
        let width = self.columns.len();
        &self.values[index * width..(index + 1) * width]
    }

    /// Number of samples (rows).
    pub fn n_samples(&self) -> usize {
        self.t.len()
    }

    /// Number of columns.
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Column semantics.
    pub fn columns(&self) -> &Columns {
        &self.columns
    }

    /// Physical unit of the values, when the producer stated one.
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Check `t` and neuron columns against a bundle's dimensions.
    pub(crate) fn check_bounds(&self, dims: &Dimensions) -> Result<(), String> {
        if let Some(last) = self.t.last() {
            // t is strictly increasing and t[0] >= 0, so `last` is the max
            // and the u64 cast is sound.
            if *last as u64 >= dims.n_steps {
                return Err(format!(
                    "series t out of range for n_steps={} (max={last})",
                    dims.n_steps
                ));
            }
        }
        self.columns.validate(Some(dims.n_neurons))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(names: &[&str]) -> Columns {
        Columns::Labels(names.iter().map(|name| name.to_string()).collect())
    }

    #[test]
    fn rows_are_row_major() {
        let series = Series::new(
            vec![1, 4],
            vec![1.0, 2.0, 3.0, 4.0],
            labels(&["a", "b"]),
            Some("mV".into()),
        )
        .unwrap();
        assert_eq!(series.n_samples(), 2);
        assert_eq!(series.n_columns(), 2);
        assert_eq!(series.sample(1), &[3.0, 4.0]);
        assert_eq!(series.unit(), Some("mV"));
    }

    #[test]
    fn empty_series_is_valid() {
        let series = Series::new(vec![], vec![], labels(&["a"]), None).unwrap();
        assert_eq!(series.n_samples(), 0);
    }

    #[test]
    fn rejects_non_increasing_negative_and_non_finite() {
        assert!(Series::new(vec![2, 2], vec![0.0, 0.0], labels(&["a"]), None).is_err());
        assert!(Series::new(vec![3, 1], vec![0.0, 0.0], labels(&["a"]), None).is_err());
        assert!(Series::new(vec![-1], vec![0.0], labels(&["a"]), None).is_err());
        assert!(Series::new(vec![0], vec![f64::NAN], labels(&["a"]), None).is_err());
        assert!(Series::new(vec![0], vec![0.0, 1.0], labels(&["a"]), None).is_err());
        assert!(Series::new(vec![0], vec![0.0], labels(&["a"]), Some(" ".into())).is_err());
    }

    #[test]
    fn rejects_bad_columns() {
        assert!(Series::new(vec![], vec![], Columns::Neurons(vec![]), None).is_err());
        assert!(Series::new(vec![], vec![], Columns::Neurons(vec![1, 1]), None).is_err());
        assert!(Series::new(vec![], vec![], labels(&["a", "a"]), None).is_err());
        assert!(Series::new(vec![], vec![], labels(&[""]), None).is_err());
    }

    #[test]
    fn bounds_follow_dimensions() {
        let dims = Dimensions {
            n_steps: 4,
            n_neurons: 2,
        };
        let ok = Series::new(vec![3], vec![0.0], Columns::Neurons(vec![1]), None).unwrap();
        assert!(ok.check_bounds(&dims).is_ok());
        let late = Series::new(vec![4], vec![0.0], Columns::Neurons(vec![1]), None).unwrap();
        assert!(late.check_bounds(&dims).is_err());
        let wide = Series::new(vec![0], vec![0.0], Columns::Neurons(vec![2]), None).unwrap();
        assert!(wide.check_bounds(&dims).is_err());
    }
}
