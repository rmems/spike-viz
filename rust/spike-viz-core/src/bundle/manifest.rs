//! `bundle.json`: versioned manifest, provenance, and the channel index.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{json, Map, Value};

use super::json::{
    child_pointer, invalid, nullable_bool, nullable_text, nullable_uint, positive_number,
    positive_uint, text, text_list, type_name, uint_list, Fields,
};
use super::series::Columns;
use super::{BundleError, BUNDLE_FORMAT, BUNDLE_VERSION, SUPPORTED_FEATURES};

/// Bundle schema version, written as `"MAJOR.MINOR"`.
///
/// Minor versions are additive only (new optional fields, new enum values);
/// anything that changes the meaning of an existing field needs a new major
/// version or a `requires` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BundleVersion {
    pub major: u32,
    pub minor: u32,
}

impl BundleVersion {
    /// Parse `"MAJOR.MINOR"` (ASCII digits only, nothing else).
    pub fn parse(text: &str) -> Option<Self> {
        let (major, minor) = text.split_once('.')?;
        Some(Self {
            major: parse_digits(major)?,
            minor: parse_digits(minor)?,
        })
    }

    /// How a reader implementing `self` treats a bundle written at `found`:
    /// `None` when the majors differ (the bundle must not be opened).
    pub fn compatibility_with(self, found: BundleVersion) -> Option<Compatibility> {
        if found.major != self.major {
            return None;
        }
        Some(if found.minor <= self.minor {
            Compatibility::Full
        } else {
            Compatibility::NewerMinor
        })
    }
}

fn parse_digits(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

impl fmt::Display for BundleVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Outcome of the version check for a bundle that opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    /// Same major, minor at or below this reader's: every field is defined.
    Full,
    /// Same major, newer minor: additive fields or values this reader does
    /// not know may be present. They are reported in
    /// [`BundleManifest::unrecognized`] or as unsupported channels, never
    /// interpreted.
    NewerMinor,
}

/// What `dt_seconds` claims about time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeConvention {
    /// `dt_seconds` is the producer's real, physical tick duration.
    Physical,
    /// `dt_seconds` is the exporter's seconds-per-step sampling convention,
    /// not a physical timebase claimed by the producer.
    Sampling,
}

impl TimeConvention {
    pub fn as_str(self) -> &'static str {
        match self {
            TimeConvention::Physical => "physical",
            TimeConvention::Sampling => "sampling",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "physical" => Some(TimeConvention::Physical),
            "sampling" => Some(TimeConvention::Sampling),
            _ => None,
        }
    }
}

/// The shared step axis: integer steps `0..n_steps`, step `k` starting
/// `k * dt_seconds` after step 0 under `convention`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timebase {
    /// Seconds per step; finite and `> 0`.
    pub dt_seconds: f64,
    pub convention: TimeConvention,
}

impl Timebase {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.dt_seconds.is_finite() || self.dt_seconds <= 0.0 {
            return Err(format!(
                "dt_seconds must be finite and > 0, got {}",
                self.dt_seconds
            ));
        }
        Ok(())
    }
}

/// Shared experiment geometry: `T` steps by `N` neurons (both `>= 1`, even
/// when a channel is silent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub n_steps: u64,
    pub n_neurons: u64,
}

impl Dimensions {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.n_steps == 0 || self.n_neurons == 0 {
            return Err(format!(
                "n_steps and n_neurons must be >= 1, got n_steps={} n_neurons={}",
                self.n_steps, self.n_neurons
            ));
        }
        Ok(())
    }
}

/// Provenance of one producer (a crate or tool that contributed channels).
///
/// `version`, `git_sha`, `seed`, and `synthetic` are **required keys** in
/// the manifest whose value may be `null`: a producer must state "not
/// known" explicitly rather than omit it, and a reader never fills one in.
#[derive(Debug, Clone, PartialEq)]
pub struct Producer {
    /// Producer identity, e.g. `"axon-encoder"`.
    pub name: String,
    /// Release version, when known.
    pub version: Option<String>,
    /// Source commit, when known (recorded in full, never shortened).
    pub git_sha: Option<String>,
    /// RNG seed used for this producer's data, when there was one / it is known.
    pub seed: Option<u64>,
    /// `Some(true)` when the data was not produced by the real producer
    /// (synthetic fixtures); `None` when the producer did not say.
    pub synthetic: Option<bool>,
    /// Optional sub-component, e.g. the axon-encoder encoder id `"rate"`.
    pub component: Option<String>,
    /// Optional opaque producer metadata (configuration, notes, the original
    /// upstream metadata record). Recorded verbatim; never interpreted.
    pub metadata: Option<Value>,
}

impl Producer {
    /// A producer with only a name: every provenance field reads "not known".
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
            git_sha: None,
            seed: None,
            synthetic: None,
            component: None,
            metadata: None,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("producer name must be non-blank".to_string());
        }
        for (key, value) in [
            ("version", &self.version),
            ("git_sha", &self.git_sha),
            ("component", &self.component),
        ] {
            if value.as_deref().is_some_and(|text| text.trim().is_empty()) {
                return Err(format!(
                    "producer {key} must be non-blank (use None when unknown)"
                ));
            }
        }
        Ok(())
    }

    fn from_json(
        value: &Value,
        path: String,
        unrecognized: &mut BTreeMap<String, Value>,
    ) -> Result<Self, BundleError> {
        let mut fields = Fields::new(value, path)?;
        let name = text(fields.required("name")?, &fields.child("name"))?;
        let version = nullable_text(fields.required("version")?, &fields.child("version"))?;
        let git_sha = nullable_text(fields.required("git_sha")?, &fields.child("git_sha"))?;
        let seed = nullable_uint(fields.required("seed")?, &fields.child("seed"))?;
        let synthetic = nullable_bool(fields.required("synthetic")?, &fields.child("synthetic"))?;
        let component = match fields.optional("component") {
            None => None,
            Some(value) => nullable_text(value, &fields.child("component"))?,
        };
        let metadata = match fields.optional("metadata") {
            None | Some(Value::Null) => None,
            Some(value) => Some(value.clone()),
        };
        fields.finish(unrecognized);
        Ok(Self {
            name,
            version,
            git_sha,
            seed,
            synthetic,
            component,
            metadata,
        })
    }

    fn to_json(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("name".into(), json!(self.name));
        obj.insert("version".into(), json!(self.version));
        obj.insert("git_sha".into(), json!(self.git_sha));
        obj.insert("seed".into(), json!(self.seed));
        obj.insert("synthetic".into(), json!(self.synthetic));
        if let Some(component) = &self.component {
            obj.insert("component".into(), json!(component));
        }
        if let Some(metadata) = &self.metadata {
            obj.insert("metadata".into(), metadata.clone());
        }
        Value::Object(obj)
    }
}

/// Semantic role of a channel. The core defines the roles; it never embeds
/// a sibling crate's types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChannelKind {
    /// `spike_events`: sparse spikes (layout `events`).
    SpikeEvents,
    /// `state_trace`: neuron state such as membrane potential (layout `series`).
    StateTrace,
    /// `modulator_trace`: neuromodulator levels (layout `series`).
    ModulatorTrace,
    /// `topology`: connectivity, delays, polarity (layout `document`).
    Topology,
    /// `metrics`: plasticity / evaluation metrics (`series` or `document`).
    Metrics,
    /// `diagnostics`: runtime / transport diagnostics (`series` or `document`).
    Diagnostics,
    /// `telemetry`: hardware telemetry (`series` or `document`).
    Telemetry,
    /// `accelerator_evidence`: accelerator / FPGA evidence (layout `document`).
    AcceleratorEvidence,
    /// A kind this reader does not know (e.g. from a newer minor version).
    /// Kept verbatim; still loadable through its layout, never treated as a
    /// known kind.
    Unrecognized(String),
}

impl ChannelKind {
    /// Every kind defined by bundle version 1.0.
    pub const KNOWN: [ChannelKind; 8] = [
        ChannelKind::SpikeEvents,
        ChannelKind::StateTrace,
        ChannelKind::ModulatorTrace,
        ChannelKind::Topology,
        ChannelKind::Metrics,
        ChannelKind::Diagnostics,
        ChannelKind::Telemetry,
        ChannelKind::AcceleratorEvidence,
    ];

    pub fn parse(text: &str) -> Self {
        Self::KNOWN
            .into_iter()
            .find(|kind| kind.as_str() == text)
            .unwrap_or_else(|| ChannelKind::Unrecognized(text.to_string()))
    }

    pub fn as_str(&self) -> &str {
        match self {
            ChannelKind::SpikeEvents => "spike_events",
            ChannelKind::StateTrace => "state_trace",
            ChannelKind::ModulatorTrace => "modulator_trace",
            ChannelKind::Topology => "topology",
            ChannelKind::Metrics => "metrics",
            ChannelKind::Diagnostics => "diagnostics",
            ChannelKind::Telemetry => "telemetry",
            ChannelKind::AcceleratorEvidence => "accelerator_evidence",
            ChannelKind::Unrecognized(text) => text,
        }
    }

    /// Layouts a known kind may use; an unrecognized kind allows any.
    pub fn allowed_layouts(&self) -> &'static [LayoutKind] {
        use LayoutKind::{Document, Events, Series};
        match self {
            ChannelKind::SpikeEvents => &[Events],
            ChannelKind::StateTrace | ChannelKind::ModulatorTrace => &[Series],
            ChannelKind::Topology | ChannelKind::AcceleratorEvidence => &[Document],
            ChannelKind::Metrics | ChannelKind::Diagnostics | ChannelKind::Telemetry => {
                &[Series, Document]
            }
            ChannelKind::Unrecognized(_) => &[Events, Series, Document],
        }
    }
}

impl fmt::Display for ChannelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// On-disk payload layout of a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutKind {
    /// Sparse spike events: `.npz` with `t`, `neuron_id`, optional `amp`.
    Events,
    /// Dense time series: `.npz` with `t` `[S]` and `values` `[S, C]`.
    Series,
    /// Static JSON document.
    Document,
}

impl LayoutKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutKind::Events => "events",
            LayoutKind::Series => "series",
            LayoutKind::Document => "document",
        }
    }

    /// The time axis this layout is indexed by.
    pub fn time_axis(self) -> TimeAxis {
        match self {
            LayoutKind::Events | LayoutKind::Series => TimeAxis::Step,
            LayoutKind::Document => TimeAxis::Static,
        }
    }

    /// Payload file extension (without the dot).
    pub(crate) fn extension(self) -> &'static str {
        match self {
            LayoutKind::Events | LayoutKind::Series => "npz",
            LayoutKind::Document => "json",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "events" => Some(LayoutKind::Events),
            "series" => Some(LayoutKind::Series),
            "document" => Some(LayoutKind::Document),
            _ => None,
        }
    }
}

/// How a channel relates to time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeAxis {
    /// Indexed by integer step on the bundle's shared step axis.
    Step,
    /// No time axis: describes the whole experiment.
    Static,
}

impl TimeAxis {
    pub fn as_str(self) -> &'static str {
        match self {
            TimeAxis::Step => "step",
            TimeAxis::Static => "static",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "step" => Some(TimeAxis::Step),
            "static" => Some(TimeAxis::Static),
            _ => None,
        }
    }
}

/// One channel entry in the manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    pub kind: ChannelKind,
    /// Id of the producer (key of [`BundleManifest::producers`]) that made it.
    pub producer: String,
    pub description: Option<String>,
    pub layout: ChannelLayout,
}

/// Where a channel's payload lives and how it is indexed.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelLayout {
    /// `time: "step"`; payload path relative to the bundle root.
    Events { path: String },
    /// `time: "step"`; columns and unit are part of the contract.
    Series {
        path: String,
        columns: Columns,
        unit: Option<String>,
    },
    /// `time: "static"`.
    Document { path: String },
    /// A layout, time axis, column index, or `requires` feature this reader
    /// does not understand. The raw manifest entry is kept; the channel is
    /// listed but cannot be loaded, so it is never misread.
    Unsupported { reason: String, raw: Value },
}

impl ChannelLayout {
    /// The layout, or `None` when unsupported.
    pub fn kind(&self) -> Option<LayoutKind> {
        match self {
            ChannelLayout::Events { .. } => Some(LayoutKind::Events),
            ChannelLayout::Series { .. } => Some(LayoutKind::Series),
            ChannelLayout::Document { .. } => Some(LayoutKind::Document),
            ChannelLayout::Unsupported { .. } => None,
        }
    }

    /// The time axis, or `None` when unsupported.
    pub fn time_axis(&self) -> Option<TimeAxis> {
        self.kind().map(LayoutKind::time_axis)
    }

    /// Payload path relative to the bundle root, or `None` when unsupported.
    pub fn path(&self) -> Option<&str> {
        match self {
            ChannelLayout::Events { path }
            | ChannelLayout::Series { path, .. }
            | ChannelLayout::Document { path } => Some(path),
            ChannelLayout::Unsupported { .. } => None,
        }
    }
}

impl Channel {
    fn from_json(
        value: &Value,
        path: String,
        dims: &Dimensions,
        producers: &BTreeMap<String, Producer>,
        unrecognized: &mut BTreeMap<String, Value>,
    ) -> Result<Self, BundleError> {
        let mut fields = Fields::new(value, path)?;
        let kind = ChannelKind::parse(&text(fields.required("kind")?, &fields.child("kind"))?);
        let producer = text(fields.required("producer")?, &fields.child("producer"))?;
        if !producers.contains_key(&producer) {
            return Err(invalid(
                &fields.child("producer"),
                format!(
                    "references undeclared producer {producer:?}; declared: {:?}",
                    producers.keys().collect::<Vec<_>>()
                ),
            ));
        }
        let description = match fields.optional("description") {
            None => None,
            Some(value) => nullable_text(value, &fields.child("description"))?,
        };
        let requires = match fields.optional("requires") {
            None => Vec::new(),
            Some(value) => text_list(value, &fields.child("requires"))?,
        };
        let layout_text = text(fields.required("layout")?, &fields.child("layout"))?;
        let time_text = text(fields.required("time")?, &fields.child("time"))?;

        let unsupported = |reason: String| Channel {
            kind: kind.clone(),
            producer: producer.clone(),
            description: description.clone(),
            layout: ChannelLayout::Unsupported {
                reason,
                raw: value.clone(),
            },
        };
        let unknown: Vec<&String> = requires
            .iter()
            .filter(|feature| !SUPPORTED_FEATURES.contains(&feature.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Ok(unsupported(format!(
                "requires unknown features {unknown:?}"
            )));
        }
        let Some(layout) = LayoutKind::parse(&layout_text) else {
            return Ok(unsupported(format!("unknown layout {layout_text:?}")));
        };
        let Some(time) = TimeAxis::parse(&time_text) else {
            return Ok(unsupported(format!("unknown time axis {time_text:?}")));
        };
        if layout.time_axis() != time {
            return Err(invalid(
                &fields.child("time"),
                format!(
                    "layout {:?} is indexed by time {:?}, got {:?}",
                    layout.as_str(),
                    layout.time_axis().as_str(),
                    time.as_str()
                ),
            ));
        }
        if !kind.allowed_layouts().contains(&layout) {
            return Err(invalid(
                &fields.child("layout"),
                format!(
                    "kind {:?} does not allow layout {:?} (allowed: {:?})",
                    kind.as_str(),
                    layout.as_str(),
                    kind.allowed_layouts()
                        .iter()
                        .map(|layout| layout.as_str())
                        .collect::<Vec<_>>()
                ),
            ));
        }
        let payload = text(fields.required("path")?, &fields.child("path"))?;
        check_payload_path(&payload, layout).map_err(|e| invalid(&fields.child("path"), e))?;

        let layout = match layout {
            LayoutKind::Events => ChannelLayout::Events { path: payload },
            LayoutKind::Document => ChannelLayout::Document { path: payload },
            LayoutKind::Series => {
                let columns_path = fields.child("columns");
                let columns = match parse_columns(
                    fields.required("columns")?,
                    columns_path,
                    dims,
                    unrecognized,
                )? {
                    Ok(columns) => columns,
                    Err(reason) => return Ok(unsupported(reason)),
                };
                let unit = match fields.optional("unit") {
                    None => None,
                    Some(value) => nullable_text(value, &fields.child("unit"))?,
                };
                ChannelLayout::Series {
                    path: payload,
                    columns,
                    unit,
                }
            }
        };
        fields.finish(unrecognized);
        Ok(Channel {
            kind,
            producer,
            description,
            layout,
        })
    }

    fn to_json(&self) -> Value {
        let (layout, path) = match &self.layout {
            ChannelLayout::Unsupported { raw, .. } => return raw.clone(),
            other => (
                other.kind().expect("supported layout"),
                other.path().expect("supported layout"),
            ),
        };
        let mut obj = Map::new();
        obj.insert("kind".into(), json!(self.kind.as_str()));
        obj.insert("producer".into(), json!(self.producer));
        if let Some(description) = &self.description {
            obj.insert("description".into(), json!(description));
        }
        obj.insert("layout".into(), json!(layout.as_str()));
        obj.insert("time".into(), json!(layout.time_axis().as_str()));
        obj.insert("path".into(), json!(path));
        if let ChannelLayout::Series { columns, unit, .. } = &self.layout {
            let columns = match columns {
                Columns::Neurons(ids) => json!({"index": "neuron", "neuron_ids": ids}),
                Columns::Labels(labels) => json!({"index": "label", "labels": labels}),
            };
            obj.insert("columns".into(), columns);
            if let Some(unit) = unit {
                obj.insert("unit".into(), json!(unit));
            }
        }
        Value::Object(obj)
    }
}

/// Parse series columns. `Ok(Err(reason))` means the column index is
/// unknown to this reader (the channel becomes unsupported).
fn parse_columns(
    value: &Value,
    path: String,
    dims: &Dimensions,
    unrecognized: &mut BTreeMap<String, Value>,
) -> Result<Result<Columns, String>, BundleError> {
    let mut fields = Fields::new(value, path.clone())?;
    let index = text(fields.required("index")?, &fields.child("index"))?;
    let columns = match index.as_str() {
        "neuron" => Columns::Neurons(uint_list(
            fields.required("neuron_ids")?,
            &fields.child("neuron_ids"),
        )?),
        "label" => Columns::Labels(text_list(
            fields.required("labels")?,
            &fields.child("labels"),
        )?),
        other => return Ok(Err(format!("unknown column index {other:?}"))),
    };
    columns
        .validate(Some(dims.n_neurons))
        .map_err(|e| invalid(&path, e))?;
    fields.finish(unrecognized);
    Ok(Ok(columns))
}

/// Channel / producer ids: 1-64 lower-case ASCII letters, digits, `_` or
/// `-`, starting with a letter or digit.
///
/// Ids double as payload file names, so upper case is not allowed: on
/// case-insensitive filesystems (APFS, NTFS) `Spikes` and `spikes` would
/// name one file and one channel would silently read the other's data.
pub(crate) fn check_id(id: &str) -> Result<(), String> {
    let allowed = |byte: u8| byte.is_ascii_lowercase() || byte.is_ascii_digit();
    let valid = !id.is_empty()
        && id.len() <= 64
        && allowed(id.as_bytes()[0])
        && id
            .bytes()
            .all(|byte| allowed(byte) || byte == b'_' || byte == b'-');
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid id {id:?}: use 1-64 lower-case ASCII letters, digits, '_' or '-', starting \
             with a letter or digit"
        ))
    }
}

/// Payload paths are `/`-separated, relative, and stay inside the bundle.
fn check_payload_path(path: &str, layout: LayoutKind) -> Result<(), String> {
    if path.contains(['\\', ':', '\0']) {
        return Err(format!(
            "payload path {path:?} must use '/' separators and no ':'"
        ));
    }
    if path.starts_with('/') {
        return Err(format!(
            "payload path {path:?} must be relative to the bundle root"
        ));
    }
    if path
        .split('/')
        .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(format!(
            "payload path {path:?} must not contain empty, '.' or '..' segments"
        ));
    }
    let extension = layout.extension();
    if !path
        .to_ascii_lowercase()
        .ends_with(&format!(".{extension}"))
    {
        return Err(format!(
            "payload path {path:?} for layout {:?} must end in .{extension}",
            layout.as_str()
        ));
    }
    Ok(())
}

/// Parsed and validated `bundle.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleManifest {
    /// Version the bundle was written at.
    pub version: BundleVersion,
    /// How completely this reader understands it.
    pub compatibility: Compatibility,
    /// Top-level `requires` features (all understood by this reader).
    pub requires: Vec<String>,
    pub timebase: Timebase,
    pub dimensions: Dimensions,
    /// Producers by id; at least one.
    pub producers: BTreeMap<String, Producer>,
    /// Channels by id; possibly none. A channel that is not listed is
    /// absent, not empty.
    pub channels: BTreeMap<String, Channel>,
    /// Namespaced, opaque extension data (never interpreted).
    pub extensions: BTreeMap<String, Value>,
    /// Fields this reader did not recognize, keyed by JSON pointer, kept
    /// verbatim and never mapped onto known fields.
    pub unrecognized: BTreeMap<String, Value>,
}

impl BundleManifest {
    /// Validate a parsed `bundle.json` document.
    ///
    /// Order matters: `format` and `bundle_version` are checked first, so a
    /// bundle from another major version reports the version, not whatever
    /// structural difference it has.
    pub fn from_json(value: &Value) -> Result<Self, BundleError> {
        let mut root = Fields::new(value, String::new())?;
        let format = root.required("format")?;
        if format.as_str() != Some(BUNDLE_FORMAT) {
            return Err(invalid(
                "/format",
                format!(
                    "not a spike-viz experiment bundle: expected {BUNDLE_FORMAT:?}, got {format}"
                ),
            ));
        }
        let version_value = root.required("bundle_version")?;
        let version = match version_value {
            Value::String(text) => BundleVersion::parse(text).ok_or_else(|| {
                invalid(
                    "/bundle_version",
                    format!("must be \"MAJOR.MINOR\", got {version_value}"),
                )
            })?,
            other => {
                return Err(invalid(
                    "/bundle_version",
                    format!("must be a \"MAJOR.MINOR\" string, got {}", type_name(other)),
                ))
            }
        };
        let compatibility = BUNDLE_VERSION.compatibility_with(version).ok_or_else(|| {
            BundleError::UnsupportedVersion {
                found: version.to_string(),
            }
        })?;
        let requires = match root.optional("requires") {
            None => Vec::new(),
            Some(value) => text_list(value, "/requires")?,
        };
        let unknown: Vec<&String> = requires
            .iter()
            .filter(|feature| !SUPPORTED_FEATURES.contains(&feature.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(BundleError::Unsupported(format!(
                "/requires: bundle requires features this reader does not understand: {unknown:?}"
            )));
        }

        let mut unrecognized = BTreeMap::new();

        let mut timebase_fields = Fields::new(root.required("timebase")?, "/timebase".into())?;
        let dt_seconds = positive_number(
            timebase_fields.required("dt_seconds")?,
            "/timebase/dt_seconds",
        )?;
        let convention_text = text(
            timebase_fields.required("convention")?,
            "/timebase/convention",
        )?;
        let convention = TimeConvention::parse(&convention_text).ok_or_else(|| {
            BundleError::Unsupported(format!(
                "/timebase/convention: unknown convention {convention_text:?}; this reader \
                 understands \"physical\" and \"sampling\""
            ))
        })?;
        timebase_fields.finish(&mut unrecognized);

        let mut dims_fields = Fields::new(root.required("dimensions")?, "/dimensions".into())?;
        let dimensions = Dimensions {
            n_steps: positive_uint(dims_fields.required("n_steps")?, "/dimensions/n_steps")?,
            n_neurons: positive_uint(dims_fields.required("n_neurons")?, "/dimensions/n_neurons")?,
        };
        dims_fields.finish(&mut unrecognized);

        let producers_value = root.required("producers")?;
        let Value::Object(producer_entries) = producers_value else {
            return Err(invalid(
                "/producers",
                format!("must be an object, got {}", type_name(producers_value)),
            ));
        };
        if producer_entries.is_empty() {
            return Err(invalid("/producers", "must declare at least one producer"));
        }
        let mut producers = BTreeMap::new();
        for (id, entry) in producer_entries {
            let path = child_pointer("/producers", id);
            check_id(id).map_err(|e| invalid(&path, e))?;
            producers.insert(
                id.clone(),
                Producer::from_json(entry, path, &mut unrecognized)?,
            );
        }

        let channels_value = root.required("channels")?;
        let Value::Object(channel_entries) = channels_value else {
            return Err(invalid(
                "/channels",
                format!("must be an object, got {}", type_name(channels_value)),
            ));
        };
        let mut channels = BTreeMap::new();
        for (id, entry) in channel_entries {
            let path = child_pointer("/channels", id);
            check_id(id).map_err(|e| invalid(&path, e))?;
            channels.insert(
                id.clone(),
                Channel::from_json(entry, path, &dimensions, &producers, &mut unrecognized)?,
            );
        }

        let extensions = match root.optional("extensions") {
            None => BTreeMap::new(),
            Some(Value::Object(entries)) => {
                for key in entries.keys() {
                    if key.trim().is_empty() {
                        return Err(invalid("/extensions", "namespace keys must be non-blank"));
                    }
                }
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            }
            Some(other) => {
                return Err(invalid(
                    "/extensions",
                    format!("must be an object, got {}", type_name(other)),
                ))
            }
        };
        root.finish(&mut unrecognized);

        Ok(Self {
            version,
            compatibility,
            requires,
            timebase: Timebase {
                dt_seconds,
                convention,
            },
            dimensions,
            producers,
            channels,
            extensions,
            unrecognized,
        })
    }

    /// The manifest document a writer emits (known fields only).
    pub(crate) fn to_json(&self) -> Value {
        let producers: Map<String, Value> = self
            .producers
            .iter()
            .map(|(id, producer)| (id.clone(), producer.to_json()))
            .collect();
        let channels: Map<String, Value> = self
            .channels
            .iter()
            .map(|(id, channel)| (id.clone(), channel.to_json()))
            .collect();
        json!({
            "format": BUNDLE_FORMAT,
            "bundle_version": self.version.to_string(),
            "requires": self.requires,
            "timebase": {
                "dt_seconds": self.timebase.dt_seconds,
                "convention": self.timebase.convention.as_str(),
            },
            "dimensions": {
                "n_steps": self.dimensions.n_steps,
                "n_neurons": self.dimensions.n_neurons,
            },
            "producers": producers,
            "channels": channels,
            "extensions": self.extensions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse_is_strict() {
        assert_eq!(
            BundleVersion::parse("1.0"),
            Some(BundleVersion { major: 1, minor: 0 })
        );
        assert_eq!(
            BundleVersion::parse("1.12"),
            Some(BundleVersion {
                major: 1,
                minor: 12
            })
        );
        for bad in [
            "1", "1.0.0", "v1.0", "1.", ".1", "1.-1", " 1.0", "1.0 ", "+1.0",
        ] {
            assert_eq!(BundleVersion::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn compatibility_rules() {
        let reader = BundleVersion { major: 1, minor: 2 };
        let at = |major, minor| reader.compatibility_with(BundleVersion { major, minor });
        assert_eq!(at(1, 0), Some(Compatibility::Full));
        assert_eq!(at(1, 2), Some(Compatibility::Full));
        assert_eq!(at(1, 3), Some(Compatibility::NewerMinor));
        assert_eq!(at(2, 0), None);
        assert_eq!(at(0, 9), None);
    }

    #[test]
    fn kinds_round_trip_and_unknown_kinds_are_kept() {
        for kind in ChannelKind::KNOWN {
            assert_eq!(ChannelKind::parse(kind.as_str()), kind);
        }
        assert_eq!(
            ChannelKind::parse("future_kind"),
            ChannelKind::Unrecognized("future_kind".into())
        );
    }

    #[test]
    fn ids_double_as_file_names() {
        for good in ["spikes", "v_mem", "a-1", "0"] {
            assert!(check_id(good).is_ok(), "{good:?}");
        }
        for bad in [
            "",
            "..",
            "a/b",
            "a.b",
            "_x",
            "-x",
            "é",
            "Spikes",
            "sPikes",
            &"x".repeat(65),
        ] {
            assert!(check_id(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn payload_paths_stay_inside_the_bundle() {
        use LayoutKind::{Document, Events};
        assert!(check_payload_path("channels/spikes.npz", Events).is_ok());
        assert!(check_payload_path("topology.json", Document).is_ok());
        for bad in [
            "../spikes.npz",
            "/abs/spikes.npz",
            "channels/./spikes.npz",
            "channels//spikes.npz",
            "channels\\spikes.npz",
            "C:spikes.npz",
            "channels/spikes.json",
        ] {
            assert!(check_payload_path(bad, Events).is_err(), "{bad:?}");
        }
    }
}
