use crate::error::SpikeIOError;

/// COO-like sparse spikes: parallel arrays of equal length.
///
/// Port of the Python `SpikeEvents` (`src/spike_viz/events.py`). `t` is the
/// encoder **step** index (absolute tick, not seconds), `neuron_id` maps from
/// axon-encoder `channel`, and `amp` defaults to 1.0 when absent. Amplitudes
/// may encode axon-encoder `polarity` as `1.0` / `-1.0` (or `0.0`), which the
/// renderer treats specially (see [`crate::render_raster`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeEvents {
    t: Vec<i64>,
    neuron_id: Vec<i64>,
    amp: Option<Vec<f32>>,
}

impl SpikeEvents {
    /// Build validated sparse events.
    ///
    /// Fails loud on mismatched lengths or non-finite amplitudes; never
    /// silently truncates. Index dtype validation happens at load time
    /// (see [`crate::load_sparse`]).
    pub fn new(
        t: Vec<i64>,
        neuron_id: Vec<i64>,
        amp: Option<Vec<f32>>,
    ) -> Result<Self, SpikeIOError> {
        if t.len() != neuron_id.len() {
            return Err(SpikeIOError::Invalid(format!(
                "t length {} != neuron_id length {}",
                t.len(),
                neuron_id.len()
            )));
        }
        if let Some(amp) = &amp {
            if amp.len() != t.len() {
                return Err(SpikeIOError::Invalid(
                    "amp must be 1-D and match t length".to_string(),
                ));
            }
            if let Some(index) = amp.iter().position(|value| !value.is_finite()) {
                return Err(SpikeIOError::Invalid(format!(
                    "amp must be finite (non-finite value at index {index})"
                )));
            }
        }
        Ok(Self { t, neuron_id, amp })
    }

    /// Step indices, in load order.
    pub fn t(&self) -> &[i64] {
        &self.t
    }

    /// Neuron / channel indices, in load order.
    pub fn neuron_id(&self) -> &[i64] {
        &self.neuron_id
    }

    /// Amplitudes, or `None` when the export omitted `amp`.
    pub fn amp(&self) -> Option<&[f32]> {
        self.amp.as_deref()
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// True when there are no events (silent activity).
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// Iterate events in load order as `(t, neuron_id, amp)`.
    ///
    /// `amp` is 1.0 when the export omitted the array; stored zero
    /// amplitudes are preserved as 0.0 (substitution is a render-time rule).
    pub fn iter(&self) -> impl Iterator<Item = (i64, i64, f32)> + '_ {
        let amp = self.amp.as_deref();
        self.t
            .iter()
            .zip(self.neuron_id.iter())
            .enumerate()
            .map(move |(index, (t, neuron))| {
                let amp = match amp {
                    Some(amp) => amp[index],
                    None => 1.0,
                };
                (*t, *neuron, amp)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_matching_lengths() {
        let events = SpikeEvents::new(vec![0, 2], vec![1, 0], None).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events.t(), &[0, 2]);
        assert_eq!(events.neuron_id(), &[1, 0]);
        assert!(events.amp().is_none());
        assert!(!events.is_empty());
    }

    #[test]
    fn new_accepts_optional_amp() {
        let events = SpikeEvents::new(vec![0], vec![1], Some(vec![0.5])).unwrap();
        assert_eq!(events.amp(), Some(&[0.5][..]));
        assert_eq!(events.iter().collect::<Vec<_>>(), vec![(0, 1, 0.5)]);
    }

    #[test]
    fn new_rejects_length_mismatch() {
        let err = SpikeEvents::new(vec![0, 1], vec![0], None).unwrap_err();
        assert!(err.to_string().contains("t length 2 != neuron_id length 1"));
    }

    #[test]
    fn new_rejects_amp_length_mismatch() {
        let err = SpikeEvents::new(vec![0], vec![0], Some(vec![])).unwrap_err();
        assert!(err
            .to_string()
            .contains("amp must be 1-D and match t length"));
    }

    #[test]
    fn new_rejects_non_finite_amp() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = SpikeEvents::new(vec![0], vec![0], Some(vec![bad])).unwrap_err();
            assert!(err.to_string().contains("amp must be finite"));
        }
    }

    #[test]
    fn iter_defaults_amp_to_one() {
        let events = SpikeEvents::new(vec![3], vec![2], None).unwrap();
        assert_eq!(events.iter().collect::<Vec<_>>(), vec![(3, 2, 1.0)]);
    }

    #[test]
    fn empty_events_are_valid() {
        let events = SpikeEvents::new(vec![], vec![], None).unwrap();
        assert_eq!(events.len(), 0);
        assert!(events.is_empty());
    }
}
