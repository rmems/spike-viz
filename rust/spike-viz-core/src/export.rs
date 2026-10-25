use std::path::{Path, PathBuf};

use crate::error::SpikeIOError;
use crate::events::SpikeEvents;
use crate::io::load_sparse;
use crate::meta::{load_meta, ExportMeta};

/// One fixture / export case under the axon-encoder layout.
///
/// Mirrors the Python `AxonExportCase` (`src/spike_viz/export.py`).
#[derive(Debug, Clone)]
pub struct AxonExportCase {
    /// Resolved case directory.
    pub path: PathBuf,
    /// Validated sparse spikes.
    pub events: SpikeEvents,
    /// Validated `meta.json`.
    pub meta: ExportMeta,
    /// Path to `spikes.npz`.
    pub spikes_path: PathBuf,
    /// Path to `meta.json`.
    pub meta_path: PathBuf,
    /// Path to the optional `stimulus.npy` underlay, when present.
    pub stimulus_path: Option<PathBuf>,
}

/// Load a case directory: `spikes.npz` + `meta.json` [+ optional `stimulus.npy`].
///
/// Port of the Python `load_axon_export` (`src/spike_viz/export.py`).
///
/// Layout (see `docs/axon-encoder-export.md`):
///
/// ```text
/// <case>/
///   spikes.npz      # required
///   meta.json       # required
///   stimulus.npy    # optional continuous input for underlays
/// ```
///
/// Does **not** invent spikes if files are missing — raises
/// [`SpikeIOError`]. Events are bounds-checked against the geometry in
/// `meta.json` (`t` in `[0, n_steps)`, `neuron_id` in `[0, n_neurons)`).
pub fn load_axon_export<P: AsRef<Path>>(case_dir: P) -> Result<AxonExportCase, SpikeIOError> {
    // Mirror the Python loader (`_as_path`): expand a leading `~`, then
    // resolve. Canonicalize needs the directory to exist, and the is_dir
    // check below reports a clearer error for a missing case.
    let expanded = expand_home(case_dir.as_ref());
    if !expanded.is_dir() {
        return Err(SpikeIOError::io(format!(
            "export case directory not found: {}",
            expanded.display()
        )));
    }
    let root = expanded
        .canonicalize()
        .map_err(|e| SpikeIOError::io(format!("failed to resolve case directory: {e}")))?;

    let meta_path = root.join("meta.json");
    let meta = load_meta(&meta_path)?;
    let spikes_path = root.join("spikes.npz");
    let events = load_sparse(&spikes_path)?;

    if !events.is_empty() {
        let t_min = events.t().iter().min().expect("non-empty");
        let t_max = events.t().iter().max().expect("non-empty");
        // Compare in u64: `meta.n_steps` is unbounded above i64::MAX, and a
        // wrapping `as i64` cast would reject valid times for huge geometry.
        // The u64 cast is sound because t_min >= 0 is checked first.
        if *t_min < 0 || (*t_max as u64) >= meta.n_steps {
            return Err(SpikeIOError::Invalid(format!(
                "{}: t out of range for n_steps={} (min={t_min}, max={t_max})",
                spikes_path.display(),
                meta.n_steps
            )));
        }
        let n_min = events.neuron_id().iter().min().expect("non-empty");
        let n_max = events.neuron_id().iter().max().expect("non-empty");
        if *n_min < 0 || (*n_max as u64) >= meta.n_neurons {
            return Err(SpikeIOError::Invalid(format!(
                "{}: neuron_id out of range for n_neurons={} (min={n_min}, max={n_max})",
                spikes_path.display(),
                meta.n_neurons
            )));
        }
    }

    let stimulus_path = {
        let candidate = root.join("stimulus.npy");
        if candidate.is_file() {
            Some(candidate)
        } else {
            None
        }
    };

    Ok(AxonExportCase {
        path: root,
        events,
        meta,
        spikes_path,
        meta_path,
        stimulus_path,
    })
}

/// Expand a leading `~` / `~/` to `$HOME`, like Python's `expanduser` for
/// the common cases (`~user` spellings are left untouched).
fn expand_home(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" || text.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let rest = text.strip_prefix('~').unwrap_or_default();
            return Path::new(&home).join(rest.trim_start_matches('/'));
        }
    }
    path.to_path_buf()
}
