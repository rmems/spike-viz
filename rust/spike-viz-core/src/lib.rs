//! Rust-first core for `spike-viz`: load spike exports, paint CPU rasters.
//!
//! Product split (see `docs/CHARTER.md`): [axon-encoder] owns encoding
//! **truth**; spike-viz only **paints**. This crate is the Rust-side port of
//! the Python reference implementation in `src/spike_viz/`, which stays
//! operational until the documented parity gates pass
//! (see `docs/rust-migration.md`). The bridge stays **export-first**: the
//! on-disk contract (`meta.json` + `spikes.npz`) is the only handoff, with
//! no in-process bindings in either direction.
//!
//! Rendering is CPU-only: no CUDA, no GPU requirement.
//!
//! Fail loud, never invent spikes: a missing or invalid export raises
//! [`SpikeIOError`] rather than returning empty activity.
//!
//! # Example
//!
//! Render the checked-in golden fixture without a GPU:
//!
//! ```
//! use spike_viz_core::{load_axon_export, render_raster, write_png};
//!
//! let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
//! let case = load_axon_export(root.join("fixtures/axon-encoder/rate/tiny_synthetic")).unwrap();
//! assert_eq!(case.meta.encoder, "rate");
//! assert_eq!(case.events.len(), 7);
//!
//! let raster = render_raster(
//!     &case.events,
//!     case.meta.n_steps as usize,
//!     case.meta.n_neurons as usize,
//!     1,
//! )
//! .unwrap();
//! assert_eq!((raster.width(), raster.height()), (8, 4));
//!
//! // write_png(&raster, "raster.png") — parent directory must exist.
//! # write_png(&raster, std::env::temp_dir().join("spike_viz_core_doctest.png")).unwrap();
//! ```
//!
//! [axon-encoder]: https://github.com/Limen-Neural/axon-encoder

mod error;
mod events;
mod export;
mod io;
mod meta;
mod render;
mod report;

pub use error::SpikeIOError;
pub use events::SpikeEvents;
pub use export::{load_axon_export, AxonExportCase};
pub use io::load_sparse;
pub use meta::{load_meta, ExportMeta, REQUIRED_META_KEYS, SUPPORTED_SCHEMA_VERSIONS};
pub use render::{render_dense, render_raster, sparse_to_dense, write_png, DenseGrid, Raster};
pub use report::{semantic_report, BinRecord, EventRecord, SemanticReport};
