//! The adapter interface, and the adapter for the axon-encoder export
//! contract (`meta.json` + `spikes.npz`, schema 1.0).

use serde_json::Value;

use crate::export::AxonExportCase;

use super::manifest::{ChannelKind, Dimensions, Producer, TimeConvention, Timebase};
use super::store::{BundleBuilder, ChannelSpec};
use super::BundleError;

/// Anything that can contribute producers and channels to a bundle.
///
/// This is the adapter interface (see `docs/experiment-bundle.md`): an
/// adapter translates its source — an on-disk artifact, or a sibling crate's
/// in-memory types behind an optional cargo feature — into bundle channels
/// through [`BundleBuilder`]. It never adds its own types to the bundle
/// schema, and whatever it contributes can be written to disk and read back
/// without the adapter.
///
/// Implementations must check that their data matches the builder's shared
/// timebase and dimensions and fail loud otherwise. Call adapters through
/// [`BundleBuilder::contribute`], which rolls back a failed contribution.
pub trait BundleAdapter {
    fn contribute(&self, bundle: &mut BundleBuilder) -> Result<(), BundleError>;
}

/// Wraps one axon-encoder export case as a producer plus a `spike_events`
/// channel, without losing provenance.
///
/// Mapping (schema 1.0 `meta.json` → bundle):
///
/// | export | bundle |
/// |---|---|
/// | `dt_seconds` | `timebase.dt_seconds`, convention `sampling` |
/// | `n_steps`, `n_neurons` | `dimensions` |
/// | `axon_encoder_version` | `producer.version` (absent → `null`) |
/// | `axon_encoder_git_sha` | `producer.git_sha` (absent / null / blank → `null`) |
/// | `seed` | `producer.seed` |
/// | `synthetic` | `producer.synthetic` (absent → `null`, never `false`) |
/// | `encoder` | `producer.component` |
/// | whole `meta.json` | `producer.metadata`, verbatim |
/// | `spikes.npz` | `spike_events` channel (layout `events`) |
///
/// The export's `dt_seconds` is "seconds per displayed step" and may not be
/// encoder-owned physical time, so the wrap claims only `sampling`.
/// `stimulus.npy` has no time/index semantics in the export contract, so it
/// is not converted (its presence is not invented as a channel).
#[derive(Debug, Clone)]
pub struct AxonExportAdapter<'a> {
    case: &'a AxonExportCase,
    producer_id: String,
    channel_id: String,
}

impl<'a> AxonExportAdapter<'a> {
    pub const DEFAULT_PRODUCER_ID: &'static str = "axon-encoder";
    pub const DEFAULT_CHANNEL_ID: &'static str = "spikes";

    pub fn new(case: &'a AxonExportCase) -> Self {
        Self {
            case,
            producer_id: Self::DEFAULT_PRODUCER_ID.to_string(),
            channel_id: Self::DEFAULT_CHANNEL_ID.to_string(),
        }
    }

    /// Use other ids, e.g. to put two exports (rate and Poisson on one
    /// stimulus) in one bundle.
    pub fn with_ids(mut self, producer_id: &str, channel_id: &str) -> Self {
        self.producer_id = producer_id.to_string();
        self.channel_id = channel_id.to_string();
        self
    }

    /// The export's step axis, claiming only the sampling convention.
    pub fn timebase(&self) -> Timebase {
        Timebase {
            dt_seconds: self.case.meta.dt_seconds,
            convention: TimeConvention::Sampling,
        }
    }

    pub fn dimensions(&self) -> Dimensions {
        Dimensions {
            n_steps: self.case.meta.n_steps,
            n_neurons: self.case.meta.n_neurons,
        }
    }

    /// Producer provenance mapped from `meta.json` (see the table above).
    pub fn producer(&self) -> Result<Producer, BundleError> {
        let meta = &self.case.meta;
        let version = match meta.raw.get("axon_encoder_version") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) if !text.trim().is_empty() => Some(text.clone()),
            Some(other) => {
                return Err(BundleError::Invalid(format!(
                    "{}: 'axon_encoder_version' must be a non-blank string or null, got {other}",
                    self.case.meta_path.display()
                )))
            }
        };
        let git_sha = meta
            .axon_encoder_git_sha
            .clone()
            .filter(|sha| !sha.trim().is_empty());
        Ok(Producer {
            name: "axon-encoder".to_string(),
            version,
            git_sha,
            seed: Some(meta.seed),
            synthetic: meta.synthetic,
            component: Some(meta.encoder.clone()),
            metadata: Some(meta.raw.clone()),
        })
    }
}

impl BundleAdapter for AxonExportAdapter<'_> {
    fn contribute(&self, bundle: &mut BundleBuilder) -> Result<(), BundleError> {
        let (dims, timebase) = (self.dimensions(), self.timebase());
        if *bundle.dimensions() != dims {
            return Err(BundleError::Invalid(format!(
                "{}: export geometry n_steps={} n_neurons={} does not match the bundle's \
                 n_steps={} n_neurons={}",
                self.case.path.display(),
                dims.n_steps,
                dims.n_neurons,
                bundle.dimensions().n_steps,
                bundle.dimensions().n_neurons
            )));
        }
        if bundle.timebase().dt_seconds != timebase.dt_seconds {
            return Err(BundleError::Invalid(format!(
                "{}: export dt_seconds={} does not match the bundle's dt_seconds={}",
                self.case.path.display(),
                timebase.dt_seconds,
                bundle.timebase().dt_seconds
            )));
        }
        bundle.add_producer(&self.producer_id, self.producer()?)?;
        let spec = ChannelSpec::new(ChannelKind::SpikeEvents, self.producer_id.clone())
            .with_description(format!(
                "spikes.npz from an axon-encoder export (encoder {:?}, export schema {})",
                self.case.meta.encoder, self.case.meta.schema_version
            ));
        bundle.add_spike_events(&self.channel_id, spec, self.case.events.clone())
    }
}

/// Start a bundle from one export case: its timebase and geometry become the
/// bundle's shared axes, and the case is contributed through
/// [`AxonExportAdapter`] with the default ids.
pub fn bundle_from_axon_export(case: &AxonExportCase) -> Result<BundleBuilder, BundleError> {
    let adapter = AxonExportAdapter::new(case);
    let mut bundle = BundleBuilder::new(adapter.timebase(), adapter.dimensions())?;
    bundle.contribute(&adapter)?;
    Ok(bundle)
}
