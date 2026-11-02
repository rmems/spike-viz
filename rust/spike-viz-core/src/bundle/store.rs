//! Bundle reader ([`ExperimentBundle`]) and writer ([`BundleBuilder`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use npyz::npz::{NpzArchive, NpzWriter};
use npyz::{DType, NpyFile, Order, TypeChar, WriterBuilder};
use serde_json::Value;

use crate::error::SpikeIOError;
use crate::events::SpikeEvents;
use crate::export::check_event_bounds;
use crate::io::{load_sparse, read_index_array};

use super::manifest::{
    check_id, BundleManifest, Channel, ChannelKind, ChannelLayout, Compatibility, Dimensions,
    LayoutKind, Producer, Timebase,
};
use super::series::Series;
use super::{BundleAdapter, BundleError, BUNDLE_MANIFEST_FILE, BUNDLE_VERSION};

/// Directory (relative to the bundle root) the writer puts payloads in.
const CHANNEL_DIR: &str = "channels";

/// An opened experiment bundle: a validated manifest plus lazy,
/// per-channel payload loading.
///
/// Opening reads only `bundle.json`. Each `load_*` call reads and validates
/// one channel, so a missing or corrupt payload fails that channel alone and
/// every other channel stays loadable.
#[derive(Debug, Clone)]
pub struct ExperimentBundle {
    root: PathBuf,
    manifest: BundleManifest,
}

impl ExperimentBundle {
    /// Open a bundle directory and validate its `bundle.json`.
    ///
    /// Fails loud on a missing directory or manifest, invalid JSON, a
    /// manifest that is not a spike-viz bundle, an unsupported major
    /// version or required feature, or any malformed known field.
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self, BundleError> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(not_found(format!(
                "experiment bundle directory not found: {}",
                dir.display()
            )));
        }
        let root = dir.canonicalize()?;
        let manifest_path = root.join(BUNDLE_MANIFEST_FILE);
        if !manifest_path.is_file() {
            return Err(not_found(format!(
                "{BUNDLE_MANIFEST_FILE} not found: {} (an axon-encoder export case has meta.json \
                 instead; wrap it with bundle_from_axon_export)",
                manifest_path.display()
            )));
        }
        let text = fs::read_to_string(&manifest_path)?;
        let value: Value = serde_json::from_str(&text).map_err(|e| {
            BundleError::Invalid(format!("invalid JSON in {}: {e}", manifest_path.display()))
        })?;
        let manifest = BundleManifest::from_json(&value)
            .map_err(|e| e.context(&manifest_path.display().to_string()))?;
        Ok(Self { root, manifest })
    }

    /// Resolved bundle directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The validated manifest.
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// Whether this reader understands every field ([`Compatibility::Full`]).
    pub fn compatibility(&self) -> Compatibility {
        self.manifest.compatibility
    }

    pub fn timebase(&self) -> &Timebase {
        &self.manifest.timebase
    }

    pub fn dimensions(&self) -> &Dimensions {
        &self.manifest.dimensions
    }

    /// A channel's manifest entry; `None` when the bundle does not have it.
    pub fn channel(&self, id: &str) -> Option<&Channel> {
        self.manifest.channels.get(id)
    }

    /// Ids of channels with the given kind, in id order.
    pub fn channels_of_kind<'a>(
        &'a self,
        kind: &'a ChannelKind,
    ) -> impl Iterator<Item = &'a str> + 'a {
        self.manifest
            .channels
            .iter()
            .filter(move |(_, channel)| &channel.kind == kind)
            .map(|(id, _)| id.as_str())
    }

    /// Load an `events` channel, bounds-checked against the dimensions.
    pub fn load_spike_events(&self, id: &str) -> Result<SpikeEvents, BundleError> {
        let path = self.payload(id, LayoutKind::Events)?;
        let events = load_sparse(&path)?;
        let dims = self.dimensions();
        check_event_bounds(&events, dims.n_steps, dims.n_neurons, &path)?;
        Ok(events)
    }

    /// Load a `series` channel, validated against its manifest columns and
    /// the dimensions.
    pub fn load_series(&self, id: &str) -> Result<Series, BundleError> {
        let path = self.payload(id, LayoutKind::Series)?;
        let ChannelLayout::Series { columns, unit, .. } = &self.manifest.channels[id].layout else {
            unreachable!("payload() checked the layout");
        };
        let context = path.display().to_string();
        let (t, values) = read_series_payload(&path, columns.len())?;
        let series = Series::new(t, values, columns.clone(), unit.clone())
            .map_err(|e| e.context(&context))?;
        series
            .check_bounds(self.dimensions())
            .map_err(|e| BundleError::Invalid(format!("{context}: {e}")))?;
        Ok(series)
    }

    /// Load a `document` channel (static JSON).
    pub fn load_document(&self, id: &str) -> Result<Value, BundleError> {
        let path = self.payload(id, LayoutKind::Document)?;
        let text = fs::read_to_string(&path)?;
        serde_json::from_str(&text)
            .map_err(|e| BundleError::Invalid(format!("invalid JSON in {}: {e}", path.display())))
    }

    /// Resolve a channel's payload, keeping absent / unsupported / wrong
    /// layout / missing file / escaping path as distinct, loud errors.
    fn payload(&self, id: &str, expected: LayoutKind) -> Result<PathBuf, BundleError> {
        let channel = self
            .manifest
            .channels
            .get(id)
            .ok_or_else(|| BundleError::ChannelAbsent { id: id.to_string() })?;
        let found = match &channel.layout {
            ChannelLayout::Unsupported { reason, .. } => {
                return Err(BundleError::ChannelUnsupported {
                    id: id.to_string(),
                    reason: reason.clone(),
                })
            }
            layout => layout.kind().expect("supported layout"),
        };
        if found != expected {
            return Err(BundleError::LayoutMismatch {
                id: id.to_string(),
                expected: expected.as_str(),
                found: found.as_str().to_string(),
            });
        }
        let candidate = self
            .root
            .join(channel.layout.path().expect("supported layout"));
        if !candidate.is_file() {
            return Err(not_found(format!(
                "channel {id:?} payload not found: {}",
                candidate.display()
            )));
        }
        let resolved = candidate.canonicalize()?;
        if !resolved.starts_with(&self.root) {
            return Err(BundleError::Invalid(format!(
                "channel {id:?} payload {} resolves outside the bundle directory {}",
                candidate.display(),
                self.root.display()
            )));
        }
        Ok(resolved)
    }
}

/// Describes a channel being added to a [`BundleBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSpec {
    pub kind: ChannelKind,
    /// Id of a producer already added with [`BundleBuilder::add_producer`].
    pub producer: String,
    pub description: Option<String>,
}

impl ChannelSpec {
    pub fn new(kind: ChannelKind, producer: impl Into<String>) -> Self {
        Self {
            kind,
            producer: producer.into(),
            description: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

#[derive(Debug, Clone)]
enum Payload {
    Events(SpikeEvents),
    Series(Series),
    Document(Value),
}

/// Assembles a bundle in memory, validating every channel as it is added,
/// then writes it with [`BundleBuilder::write`].
///
/// The writer emits only what [`BUNDLE_VERSION`] defines: known channel
/// kinds, supported layouts, and no `requires` features.
#[derive(Debug, Clone)]
pub struct BundleBuilder {
    timebase: Timebase,
    dimensions: Dimensions,
    producers: BTreeMap<String, Producer>,
    channels: BTreeMap<String, (ChannelSpec, Payload)>,
    extensions: BTreeMap<String, Value>,
}

impl BundleBuilder {
    /// Start a bundle on a shared timebase and geometry.
    pub fn new(timebase: Timebase, dimensions: Dimensions) -> Result<Self, BundleError> {
        timebase.validate().map_err(BundleError::Invalid)?;
        dimensions.validate().map_err(BundleError::Invalid)?;
        Ok(Self {
            timebase,
            dimensions,
            producers: BTreeMap::new(),
            channels: BTreeMap::new(),
            extensions: BTreeMap::new(),
        })
    }

    pub fn timebase(&self) -> &Timebase {
        &self.timebase
    }

    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    /// Declared producers by id.
    pub fn producers(&self) -> &BTreeMap<String, Producer> {
        &self.producers
    }

    /// Ids of the channels added so far, in id order.
    pub fn channel_ids(&self) -> impl Iterator<Item = &str> {
        self.channels.keys().map(String::as_str)
    }

    /// Declare a producer under a unique id.
    pub fn add_producer(&mut self, id: &str, producer: Producer) -> Result<(), BundleError> {
        check_id(id).map_err(|e| BundleError::Invalid(format!("producer id: {e}")))?;
        if self.producers.contains_key(id) {
            return Err(BundleError::Invalid(format!(
                "producer {id:?} is already declared"
            )));
        }
        producer
            .validate()
            .map_err(|e| BundleError::Invalid(format!("producer {id:?}: {e}")))?;
        self.producers.insert(id.to_string(), producer);
        Ok(())
    }

    /// Add sparse spike events (layout `events`), bounds-checked against the
    /// dimensions. An empty event list is a present, silent channel.
    pub fn add_spike_events(
        &mut self,
        id: &str,
        spec: ChannelSpec,
        events: SpikeEvents,
    ) -> Result<(), BundleError> {
        self.check_channel(id, &spec, LayoutKind::Events)?;
        check_event_bounds(
            &events,
            self.dimensions.n_steps,
            self.dimensions.n_neurons,
            Path::new(&format!("channel {id:?}")),
        )?;
        self.channels
            .insert(id.to_string(), (spec, Payload::Events(events)));
        Ok(())
    }

    /// Add a time series (layout `series`), bounds-checked against the
    /// dimensions.
    pub fn add_series(
        &mut self,
        id: &str,
        spec: ChannelSpec,
        series: Series,
    ) -> Result<(), BundleError> {
        self.check_channel(id, &spec, LayoutKind::Series)?;
        series
            .check_bounds(&self.dimensions)
            .map_err(|e| BundleError::Invalid(format!("channel {id:?}: {e}")))?;
        self.channels
            .insert(id.to_string(), (spec, Payload::Series(series)));
        Ok(())
    }

    /// Add a static JSON document (layout `document`).
    pub fn add_document(
        &mut self,
        id: &str,
        spec: ChannelSpec,
        document: Value,
    ) -> Result<(), BundleError> {
        self.check_channel(id, &spec, LayoutKind::Document)?;
        self.channels
            .insert(id.to_string(), (spec, Payload::Document(document)));
        Ok(())
    }

    /// Attach opaque, namespaced extension data (never interpreted).
    pub fn add_extension(&mut self, namespace: &str, value: Value) -> Result<(), BundleError> {
        if namespace.trim().is_empty() {
            return Err(BundleError::Invalid(
                "extension namespace must be non-blank".to_string(),
            ));
        }
        if self.extensions.contains_key(namespace) {
            return Err(BundleError::Invalid(format!(
                "extension namespace {namespace:?} is already set"
            )));
        }
        self.extensions.insert(namespace.to_string(), value);
        Ok(())
    }

    /// Let an adapter add its producers and channels. All-or-nothing: if the
    /// adapter fails, the builder is left exactly as it was.
    ///
    /// Rollback records only the ids present beforehand, never payloads: the
    /// builder's API can add producers, channels, and extensions but never
    /// replace or remove one, so dropping the ids added during a failed
    /// contribution restores the previous state exactly.
    pub fn contribute<A: BundleAdapter + ?Sized>(
        &mut self,
        adapter: &A,
    ) -> Result<(), BundleError> {
        let producers: BTreeSet<String> = self.producers.keys().cloned().collect();
        let channels: BTreeSet<String> = self.channels.keys().cloned().collect();
        let extensions: BTreeSet<String> = self.extensions.keys().cloned().collect();
        let result = adapter.contribute(self);
        if result.is_err() {
            self.producers.retain(|id, _| producers.contains(id));
            self.channels.retain(|id, _| channels.contains(id));
            self.extensions.retain(|key, _| extensions.contains(key));
        }
        result
    }

    /// The manifest [`BundleBuilder::write`] will emit.
    pub fn manifest(&self) -> BundleManifest {
        let channels = self
            .channels
            .iter()
            .map(|(id, (spec, payload))| {
                let layout = match payload {
                    Payload::Events(_) => ChannelLayout::Events {
                        path: payload_path(id, LayoutKind::Events),
                    },
                    Payload::Series(series) => ChannelLayout::Series {
                        path: payload_path(id, LayoutKind::Series),
                        columns: series.columns().clone(),
                        unit: series.unit().map(str::to_string),
                    },
                    Payload::Document(_) => ChannelLayout::Document {
                        path: payload_path(id, LayoutKind::Document),
                    },
                };
                let channel = Channel {
                    kind: spec.kind.clone(),
                    producer: spec.producer.clone(),
                    description: spec.description.clone(),
                    layout,
                };
                (id.clone(), channel)
            })
            .collect();
        BundleManifest {
            version: BUNDLE_VERSION,
            compatibility: Compatibility::Full,
            requires: Vec::new(),
            timebase: self.timebase,
            dimensions: self.dimensions,
            producers: self.producers.clone(),
            channels,
            extensions: self.extensions.clone(),
            unrecognized: BTreeMap::new(),
        }
    }

    /// Write the bundle into `dir` and reopen it through the reader.
    ///
    /// `dir` must be absent (its parent must exist) or an empty directory:
    /// the writer never mixes its files with stale ones. Payloads are written
    /// first and `bundle.json` last, so an interrupted write never leaves a
    /// directory that opens as a bundle. Output is deterministic (sorted
    /// keys, fixed archive timestamps).
    pub fn write<P: AsRef<Path>>(&self, dir: P) -> Result<ExperimentBundle, BundleError> {
        let dir = dir.as_ref();
        if self.producers.is_empty() {
            return Err(BundleError::Invalid(
                "a bundle needs at least one producer".to_string(),
            ));
        }
        prepare_output_dir(dir)?;
        let manifest = self.manifest();
        if !self.channels.is_empty() {
            fs::create_dir(dir.join(CHANNEL_DIR))?;
        }
        for (id, (_, payload)) in &self.channels {
            let relative = manifest.channels[id]
                .layout
                .path()
                .expect("written layouts are supported");
            let path = dir.join(relative);
            match payload {
                Payload::Events(events) => write_events(&path, events)?,
                Payload::Series(series) => write_series(&path, series)?,
                Payload::Document(document) => write_json(&path, document)?,
            }
        }
        write_json(&dir.join(BUNDLE_MANIFEST_FILE), &manifest.to_json())?;
        ExperimentBundle::open(dir)
    }

    fn check_channel(
        &self,
        id: &str,
        spec: &ChannelSpec,
        layout: LayoutKind,
    ) -> Result<(), BundleError> {
        check_id(id).map_err(|e| BundleError::Invalid(format!("channel id: {e}")))?;
        if self.channels.contains_key(id) {
            return Err(BundleError::Invalid(format!(
                "channel {id:?} is already present"
            )));
        }
        if !self.producers.contains_key(&spec.producer) {
            return Err(BundleError::Invalid(format!(
                "channel {id:?} references undeclared producer {:?}; call add_producer first",
                spec.producer
            )));
        }
        if let ChannelKind::Unrecognized(kind) = &spec.kind {
            return Err(BundleError::Invalid(format!(
                "channel {id:?}: kind {kind:?} is not defined by bundle_version {BUNDLE_VERSION}"
            )));
        }
        if !spec.kind.allowed_layouts().contains(&layout) {
            return Err(BundleError::Invalid(format!(
                "channel {id:?}: kind {:?} does not allow layout {:?}",
                spec.kind.as_str(),
                layout.as_str()
            )));
        }
        if spec
            .description
            .as_deref()
            .is_some_and(|text| text.trim().is_empty())
        {
            return Err(BundleError::Invalid(format!(
                "channel {id:?}: description must be non-blank (use None)"
            )));
        }
        Ok(())
    }
}

fn payload_path(id: &str, layout: LayoutKind) -> String {
    format!("{CHANNEL_DIR}/{id}.{}", layout.extension())
}

fn not_found(message: String) -> BundleError {
    BundleError::Io(io::Error::new(io::ErrorKind::NotFound, message))
}

fn prepare_output_dir(dir: &Path) -> Result<(), BundleError> {
    if dir.exists() {
        if !dir.is_dir() {
            return Err(BundleError::Invalid(format!(
                "bundle output {} exists and is not a directory",
                dir.display()
            )));
        }
        if fs::read_dir(dir)?.next().is_some() {
            return Err(BundleError::Invalid(format!(
                "refusing to write a bundle into non-empty directory {}",
                dir.display()
            )));
        }
        return Ok(());
    }
    fs::create_dir(dir).map_err(|e| {
        BundleError::Io(io::Error::new(
            e.kind(),
            format!(
                "failed to create bundle directory {} (its parent must exist): {e}",
                dir.display()
            ),
        ))
    })
}

fn read_series_payload(path: &Path, n_columns: usize) -> Result<(Vec<i64>, Vec<f64>), BundleError> {
    let read_error = |e: io::Error| {
        SpikeIOError::io(format!(
            "failed to read series from {}: {e}",
            path.display()
        ))
    };
    let mut archive = NpzArchive::open(path).map_err(read_error)?;
    let mut names: Vec<String> = archive.array_names().map(str::to_string).collect();
    names.sort();
    let missing = || {
        BundleError::Invalid(format!(
            "{}: required arrays 't' and 'values' missing; found {names:?}",
            path.display()
        ))
    };
    let t = match archive.by_name("t").map_err(read_error)? {
        Some(npy) => read_index_array(npy, "t")?,
        None => return Err(missing()),
    };
    let values = match archive.by_name("values").map_err(read_error)? {
        Some(npy) => read_values(npy, t.len(), n_columns, path)?,
        None => return Err(missing()),
    };
    Ok((t, values))
}

/// Read `values` as row-major `f64`, accepting C or Fortran order.
fn read_values<R: Read>(
    npy: NpyFile<R>,
    rows: usize,
    cols: usize,
    path: &Path,
) -> Result<Vec<f64>, BundleError> {
    let shape = npy.shape().to_vec();
    if shape != [rows as u64, cols as u64] {
        return Err(BundleError::Invalid(format!(
            "{}: values must have shape [n_samples, n_columns] = [{rows}, {cols}] (n_samples \
             from t, n_columns from the manifest), got {shape:?}",
            path.display()
        )));
    }
    let order = npy.order();
    let read_failed = |e: io::Error| SpikeIOError::io(format!("values: failed to read array: {e}"));
    let flat: Vec<f64> = match npy.dtype() {
        DType::Plain(type_str) => match (type_str.type_char(), type_str.size_field()) {
            (TypeChar::Float, 8) => npy.into_vec::<f64>().map_err(read_failed)?,
            (TypeChar::Float, 4) => npy
                .into_vec::<f32>()
                .map_err(read_failed)?
                .into_iter()
                .map(f64::from)
                .collect(),
            _ => {
                return Err(BundleError::Invalid(format!(
                    "{}: values must be float32 or float64, got dtype {type_str}",
                    path.display()
                )))
            }
        },
        other => {
            return Err(BundleError::Invalid(format!(
                "{}: values must be float32 or float64, got structured dtype {other:?}",
                path.display()
            )))
        }
    };
    Ok(match order {
        Order::C => flat,
        Order::Fortran => {
            let mut row_major = Vec::with_capacity(flat.len());
            for row in 0..rows {
                for col in 0..cols {
                    row_major.push(flat[col * rows + row]);
                }
            }
            row_major
        }
    })
}

type NpzFile = NpzWriter<BufWriter<File>>;

fn write_events(path: &Path, events: &SpikeEvents) -> Result<(), BundleError> {
    let len = [events.len() as u64];
    let mut npz = NpzWriter::create(path)?;
    write_array(&mut npz, "t", events.t(), &len)?;
    write_array(&mut npz, "neuron_id", events.neuron_id(), &len)?;
    if let Some(amp) = events.amp() {
        write_array(&mut npz, "amp", amp, &len)?;
    }
    finish_npz(npz)
}

fn write_series(path: &Path, series: &Series) -> Result<(), BundleError> {
    let mut npz = NpzWriter::create(path)?;
    write_array(&mut npz, "t", series.t(), &[series.n_samples() as u64])?;
    write_array(
        &mut npz,
        "values",
        series.values(),
        &[series.n_samples() as u64, series.n_columns() as u64],
    )?;
    finish_npz(npz)
}

fn write_array<T>(npz: &mut NpzFile, name: &str, values: &[T], shape: &[u64]) -> io::Result<()>
where
    T: npyz::AutoSerialize + Clone,
    npyz::WriteOptions<T>: WriterBuilder<T>,
{
    // A fixed timestamp keeps archives byte-identical across writes.
    let options =
        npyz::zip::write::FileOptions::default().last_modified_time(npyz::zip::DateTime::default());
    let mut writer = npz
        .array::<T>(name, options)?
        .default_dtype()
        .shape(shape)
        .begin_nd()?;
    writer.extend(values.iter().cloned())?;
    writer.finish()
}

fn finish_npz(mut npz: NpzFile) -> Result<(), BundleError> {
    let mut file = npz.zip_writer().finish().map_err(io::Error::from)?;
    file.flush()?;
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<(), BundleError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(io::Error::from)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}
