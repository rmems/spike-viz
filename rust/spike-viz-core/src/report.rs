use serde::Serialize;

use crate::error::SpikeIOError;
use crate::export::AxonExportCase;
use crate::render::{render_raster, Raster};

/// One sparse event, in load order. `amp` is `null` when the export
/// omitted the array.
///
/// `amp` carries the in-memory `f32` amplitude **widened to `f64`**: serde
/// writes the shortest round-trip *f32* decimal for an `f32` field (e.g.
/// `0.1`), which does not equal the `f64` widening of the same value
/// (`0.10000000149011612`) that the Python reference compares against.
/// Widening before serialization keeps the parity comparison exact.
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    pub t: i64,
    pub neuron_id: i64,
    pub amp: Option<f64>,
}

/// One non-black raster bin after normalization: final `u8` intensity for
/// the `(t, neuron_id)` cell (row-major over the `[T, N]` grid).
#[derive(Debug, Clone, Serialize)]
pub struct BinRecord {
    pub t: i64,
    pub neuron_id: i64,
    pub value: u8,
}

/// Semantic summary of one rendered export case.
///
/// This is the Rust/Python parity bridge: the fields compare **semantic
/// output** — the exact event list, grid geometry, peak, and final
/// intensities — without requiring pixel-perfect PNG identity. Written by
/// the `spike-viz` CLI's `--report`; consumed by
/// `tests/test_rust_parity.py`.
#[derive(Debug, Clone, Serialize)]
pub struct SemanticReport {
    /// Always `"spike-viz-core"`.
    pub tool: &'static str,
    /// Export contract version from `meta.json`.
    pub schema_version: String,
    pub encoder: String,
    pub axon_encoder_git_sha: Option<String>,
    pub synthetic: Option<bool>,
    /// Seconds per displayed step (export sampling convention).
    pub dt_seconds: f64,
    pub seed: u64,
    pub n_steps: u64,
    pub n_neurons: u64,
    /// Number of events in the export (length of `events`).
    pub n_events: usize,
    /// Nearest-neighbor upscale factor used for the PNG.
    pub scale: usize,
    /// PNG width in pixels (`n_steps * scale`).
    pub width: usize,
    /// PNG height in pixels (`n_neurons * scale`).
    pub height: usize,
    /// Pre-normalization peak of the frame, widened to `f64` so the JSON
    /// carries the exact `f32` value (see [`EventRecord::amp`]).
    pub peak: f64,
    /// Events in load order.
    pub events: Vec<EventRecord>,
    /// Non-black bins in `(t, neuron_id)` row-major order.
    pub bins: Vec<BinRecord>,
}

/// Build the semantic report for a case by rendering it at `scale`.
pub fn semantic_report(
    case: &AxonExportCase,
    scale: usize,
) -> Result<SemanticReport, SpikeIOError> {
    let raster = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        scale,
    )?;
    SemanticReport::from_render(case, &raster)
}

impl SemanticReport {
    /// Build the semantic report from an already-rendered raster.
    ///
    /// The raster must have been rendered from the same case geometry:
    /// independent `case` and `raster` arguments could silently misreport
    /// otherwise, so a grid mismatch fails loud.
    pub fn from_render(case: &AxonExportCase, raster: &Raster) -> Result<Self, SpikeIOError> {
        let (n_steps, n_neurons) = raster.grid_shape();
        if (n_steps as u64, n_neurons as u64) != (case.meta.n_steps, case.meta.n_neurons) {
            return Err(SpikeIOError::Invalid(format!(
                "raster grid {n_steps}x{n_neurons} (n_steps, n_neurons) does not match case \
                 geometry {}x{}",
                case.meta.n_steps, case.meta.n_neurons
            )));
        }
        let mut bins = Vec::new();
        // Row-major over the [T, N] grid (time-major, neuron-minor).
        for t in 0..n_steps {
            for n in 0..n_neurons {
                let value = raster.pixels()[n * n_steps + t];
                if value != 0 {
                    bins.push(BinRecord {
                        t: t as i64,
                        neuron_id: n as i64,
                        value,
                    });
                }
            }
        }
        let amp_present = case.events.amp().is_some();
        Ok(Self {
            tool: "spike-viz-core",
            schema_version: case.meta.schema_version.clone(),
            encoder: case.meta.encoder.clone(),
            axon_encoder_git_sha: case.meta.axon_encoder_git_sha.clone(),
            synthetic: case.meta.synthetic,
            dt_seconds: case.meta.dt_seconds,
            seed: case.meta.seed,
            n_steps: case.meta.n_steps,
            n_neurons: case.meta.n_neurons,
            n_events: case.events.len(),
            scale: raster.scale(),
            width: raster.width(),
            height: raster.height(),
            peak: raster.peak() as f64,
            events: case
                .events
                .iter()
                .map(|(t, neuron, amp)| EventRecord {
                    t,
                    neuron_id: neuron,
                    amp: if amp_present { Some(amp as f64) } else { None },
                })
                .collect(),
            bins,
        })
    }
}
