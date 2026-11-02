//! Versioned experiment bundle: the language-neutral artifact contract that
//! lets several neuromorphic crates contribute data to spike-viz without
//! depending on each other (spec: `docs/experiment-bundle.md`).
//!
//! A bundle is a directory holding a `bundle.json` manifest plus one payload
//! file per channel. The manifest carries the required provenance — the
//! shared step timebase, the `(n_steps, n_neurons)` dimensions, and one
//! entry per producer (name, version / git SHA when known, seed, synthetic
//! flag) — and an index of **optional** channels: spike events,
//! neuron/state traces, modulator traces, topology, metrics, diagnostics,
//! hardware telemetry, accelerator/FPGA evidence.
//!
//! Rules this module enforces:
//!
//! - **Versioned, with explicit compatibility.** `bundle_version` is
//!   `MAJOR.MINOR`. A different major fails to open; a newer minor opens as
//!   [`Compatibility::NewerMinor`]. A `requires` list names features a reader
//!   must understand: an unknown one fails the bundle (top level) or makes the
//!   channel [`ChannelLayout::Unsupported`] (channel level).
//! - **Known fields are strictly typed; unknown fields are kept apart.**
//!   Unrecognized keys are recorded verbatim in
//!   [`BundleManifest::unrecognized`] under their JSON pointer and never
//!   mapped onto known fields. Unknown layouts, time axes, or column indexes
//!   make that one channel unsupported rather than misread.
//! - **Channels are optional and independently loadable.** Asking for a
//!   channel the bundle does not have is [`BundleError::ChannelAbsent`],
//!   never empty data; a broken payload fails only its own channel.
//! - **Explicit time/index semantics.** Event and series channels are indexed
//!   by integer step on the shared axis (`time: "step"`); documents are
//!   `time: "static"`. Series columns say whether they are neuron ids or
//!   named signals.
//!
//! The core schema embeds no sibling-crate types. Adapters
//! ([`BundleAdapter`]) translate a source into bundle channels; the
//! axon-encoder export contract is wrapped by [`AxonExportAdapter`].
//!
//! # Example
//!
//! ```
//! use spike_viz_core::bundle::{
//!     BundleBuilder, ChannelKind, ChannelSpec, Columns, Dimensions, ExperimentBundle, Producer,
//!     Series, TimeConvention, Timebase,
//! };
//! use spike_viz_core::SpikeEvents;
//!
//! let timebase = Timebase { dt_seconds: 0.001, convention: TimeConvention::Sampling };
//! let dims = Dimensions { n_steps: 4, n_neurons: 2 };
//! let mut builder = BundleBuilder::new(timebase, dims).unwrap();
//! builder.add_producer("enc", Producer::new("example-encoder")).unwrap();
//!
//! let spikes = SpikeEvents::new(vec![0, 3], vec![1, 0], None).unwrap();
//! builder
//!     .add_spike_events("spikes", ChannelSpec::new(ChannelKind::SpikeEvents, "enc"), spikes.clone())
//!     .unwrap();
//! let trace = Series::new(vec![0, 2], vec![0.5, 0.25], Columns::Labels(vec!["da".into()]), None)
//!     .unwrap();
//! builder
//!     .add_series("dopamine", ChannelSpec::new(ChannelKind::ModulatorTrace, "enc"), trace.clone())
//!     .unwrap();
//!
//! let dir = std::env::temp_dir().join(format!("spike_viz_bundle_doctest_{}", std::process::id()));
//! # let _ = std::fs::remove_dir_all(&dir);
//! let bundle = builder.write(&dir).unwrap();
//! assert_eq!(bundle.load_spike_events("spikes").unwrap(), spikes);
//! assert_eq!(bundle.load_series("dopamine").unwrap(), trace);
//! // A channel the bundle does not have is an error, never empty data.
//! assert!(bundle.load_series("membrane").is_err());
//!
//! let reopened = ExperimentBundle::open(&dir).unwrap();
//! assert_eq!(reopened.manifest().producers["enc"].name, "example-encoder");
//! # std::fs::remove_dir_all(&dir).unwrap();
//! ```

mod adapter;
mod json;
mod manifest;
mod series;
mod store;

use std::io;

use crate::error::SpikeIOError;

pub use adapter::{bundle_from_axon_export, AxonExportAdapter, BundleAdapter};
pub use manifest::{
    BundleManifest, BundleVersion, Channel, ChannelKind, ChannelLayout, Compatibility, Dimensions,
    LayoutKind, Producer, TimeAxis, TimeConvention, Timebase,
};
pub use series::{Columns, Series};
pub use store::{BundleBuilder, ChannelSpec, ExperimentBundle};

/// Value of the manifest's `format` key; identifies a spike-viz bundle.
pub const BUNDLE_FORMAT: &str = "spike-viz.experiment-bundle";

/// Manifest file name at the bundle root.
pub const BUNDLE_MANIFEST_FILE: &str = "bundle.json";

/// The bundle schema version this crate reads fully and writes.
pub const BUNDLE_VERSION: BundleVersion = BundleVersion { major: 1, minor: 0 };

/// `requires` features this reader understands. Bundle version 1.0 defines
/// none, so any listed feature is unknown to this reader.
pub const SUPPORTED_FEATURES: &[&str] = &[];

/// Errors from opening, loading, or writing an experiment bundle.
///
/// The variants keep "not there", "too new for this reader", and "invalid"
/// apart, so a caller never mistakes an absent channel for empty activity.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    /// Filesystem failure while reading or writing bundle files.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// A spike payload failed the shared export loaders (missing file,
    /// invalid arrays, out-of-range events, ...).
    #[error(transparent)]
    Spike(#[from] SpikeIOError),
    /// The manifest or a payload violates the bundle contract.
    #[error("{0}")]
    Invalid(String),
    /// `bundle_version` names a major version this reader does not implement.
    #[error(
        "unsupported bundle_version {found:?}: this reader implements major version {} \
         (up to {})",
        BUNDLE_VERSION.major,
        BUNDLE_VERSION
    )]
    UnsupportedVersion { found: String },
    /// The bundle needs something this reader does not understand
    /// (a top-level `requires` feature or an unknown timebase convention).
    #[error("{0}")]
    Unsupported(String),
    /// The requested channel is not in the manifest.
    #[error("channel {id:?} is not present in this bundle")]
    ChannelAbsent { id: String },
    /// The channel is declared but uses a layout, time axis, column index, or
    /// required feature this reader does not understand.
    #[error("channel {id:?} cannot be loaded by this reader: {reason}")]
    ChannelUnsupported { id: String, reason: String },
    /// The channel exists with a different layout than the one requested.
    #[error("channel {id:?} has layout {found:?}, not {expected:?}")]
    LayoutMismatch {
        id: String,
        expected: &'static str,
        found: String,
    },
}

impl BundleError {
    /// Prefix an `Invalid` message with where it came from (e.g. the manifest path).
    fn context(self, prefix: &str) -> Self {
        match self {
            BundleError::Invalid(message) => BundleError::Invalid(format!("{prefix}: {message}")),
            BundleError::Unsupported(message) => {
                BundleError::Unsupported(format!("{prefix}: {message}"))
            }
            other => other,
        }
    }
}
