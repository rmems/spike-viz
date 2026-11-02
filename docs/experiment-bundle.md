# Experiment bundle contract (v1.0)

Status: **implemented in the Rust core** (`rust/spike-viz-core/src/bundle/`),
tracked as spike-viz issue #31 ([RM-1925]), parent epic #29. Charter:
[CHARTER.md](CHARTER.md). Rust migration state:
[rust-migration.md](rust-migration.md).

An experiment bundle is the language-neutral artifact that lets several
neuromorphic crates contribute data about **one experiment** to spike-viz
without depending on each other, or on spike-viz. It generalizes the
export-first bridge in [axon-encoder-export.md](axon-encoder-export.md):
producers still own truth and write files; spike-viz only loads and paints.

A bundle carries only the channels that exist for the experiment:

| Channel kind | What it holds | Layouts |
|---|---|---|
| `spike_events` | spike events | `events` |
| `state_trace` | neuron / state traces (e.g. membrane potential) | `series` |
| `modulator_trace` | neuromodulator traces | `series` |
| `topology` | topology / delay / polarity metadata | `document` |
| `metrics` | plasticity / evaluation metrics | `series`, `document` |
| `diagnostics` | runtime / transport diagnostics | `series`, `document` |
| `telemetry` | hardware telemetry | `series`, `document` |
| `accelerator_evidence` | accelerator / FPGA evidence | `document` |

Provenance and source metadata are not a channel. They are required parts
of the manifest. The schema has no sibling-crate types in it: a channel is
a kind (semantic role) plus a layout (on-disk shape), and nothing more.

## Layout on disk

```text
<bundle>/
  bundle.json            # manifest: version, provenance, channel index
  channels/<id>.npz      # events / series payloads (NumPy .npz, no pickle)
  channels/<id>.json     # document payloads
```

`bundle.json` is the single source of truth. A channel exists only when the
manifest lists it, and its `path` (relative, `/`-separated) names its payload.
The `channels/<id>.<ext>` naming is what the spike-viz writer does, not a
rule readers depend on. Payloads are plain JSON and NumPy `.npz`, so any
language can write a bundle.

## Manifest (`bundle.json`)

Adapted from what the writer emits in the round-trip test (trimmed; the
writer sorts keys):

```json
{
  "format": "spike-viz.experiment-bundle",
  "bundle_version": "1.0",
  "requires": [],
  "timebase": { "dt_seconds": 0.001, "convention": "physical" },
  "dimensions": { "n_steps": 16, "n_neurons": 4 },
  "producers": {
    "encoder": {
      "name": "example-encoder", "version": "0.6.0",
      "git_sha": "908a34797c8aaff0a57e0ea4167aa00f3780c940",
      "seed": 42, "synthetic": true, "component": "rate",
      "metadata": { "encoder_config": { "max_rate_hz": 400.0 } }
    },
    "neuro": {
      "name": "example-neuron-model", "version": null,
      "git_sha": null, "seed": null, "synthetic": true
    }
  },
  "channels": {
    "spikes": {
      "kind": "spike_events", "producer": "encoder",
      "layout": "events", "time": "step", "path": "channels/spikes.npz"
    },
    "membrane": {
      "kind": "state_trace", "producer": "neuro",
      "layout": "series", "time": "step", "path": "channels/membrane.npz",
      "columns": { "index": "neuron", "neuron_ids": [2, 0] }, "unit": "mV"
    },
    "modulators": {
      "kind": "modulator_trace", "producer": "neuro",
      "layout": "series", "time": "step", "path": "channels/modulators.npz",
      "columns": { "index": "label", "labels": ["dopamine", "serotonin"] }
    },
    "topology": {
      "kind": "topology", "producer": "neuro",
      "layout": "document", "time": "static", "path": "channels/topology.json"
    }
  },
  "extensions": { "example.org/run": { "operator_notes": "..." } }
}
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `format` | string | **yes** | Must be `"spike-viz.experiment-bundle"` |
| `bundle_version` | `"MAJOR.MINOR"` string | **yes** | Schema version (see compatibility) |
| `requires` | array of strings | no | Features a reader must understand to read the bundle at all |
| `timebase` | object | **yes** | Shared step axis (below) |
| `dimensions` | object | **yes** | `n_steps` (T) and `n_neurons` (N), integers `>= 1` |
| `producers` | object: id → producer | **yes** | At least one producer |
| `channels` | object: id → channel | **yes** | May be empty; an unlisted channel is absent |
| `extensions` | object: namespace → any | no | Opaque, namespaced extra data; preserved, never interpreted |

Producer and channel ids are 1 to 64 **lower-case** ASCII letters, digits,
`_` or `-`, starting with a letter or digit. Ids name payload files, and
upper case would let `Spikes` and `spikes` collide on case-insensitive
filesystems (APFS, NTFS), where one channel would silently read the other's
data.

## Required provenance

**Timebase.** `dt_seconds` is a finite number `> 0`, in seconds per step.
`convention` says what that claims:

- `"physical"`: the producer's real tick duration.
- `"sampling"`: the exporter's seconds-per-displayed-step convention, not a
  physical timebase claimed by the producer. This is what the axon-encoder
  export's `dt_seconds` means for dimensionless encoders.

**Dimensions.** `n_steps` and `n_neurons` define the shared grid every
step-indexed channel lives on. Both must be `>= 1`, even when every channel
is silent.

**Producers.** Each producer (a crate or tool that contributed channels)
records:

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | non-blank string | **yes** | Producer identity, e.g. `"axon-encoder"` |
| `version` | string \| null | **yes** (nullable) | Release version, when known |
| `git_sha` | string \| null | **yes** (nullable) | Source commit, when known, in full |
| `seed` | integer \| null | **yes** (nullable) | RNG seed, when there was one / it is known |
| `synthetic` | boolean \| null | **yes** (nullable) | `true` when not produced by the real producer |
| `component` | string \| null | no | Sub-component, e.g. the encoder id `"rate"` |
| `metadata` | any | no | Opaque producer metadata (config, notes, original upstream record) |

The nullable keys are **required**. A producer that does not know a value
must write `null`, and a reader never fills one in. A blank string is
rejected, so "unknown" can only be written as `null`. Every channel names
its producer, so provenance stays attached to the channel even when the
bundle mixes data from several crates.

## Channels

Every channel has:

| Key | Required | Meaning |
|---|---|---|
| `kind` | **yes** | Semantic role (table at the top) |
| `producer` | **yes** | Id of a declared producer |
| `layout` | **yes** | `events`, `series`, or `document` |
| `time` | **yes** | `step` or `static`; must match the layout |
| `path` | **yes** | Payload path relative to the bundle root |
| `description` | no | Human note (string or null) |
| `requires` | no | Features a reader must understand to load this channel |
| `columns`, `unit` | series only | Column semantics (required) and unit (optional) |

### Time and index semantics

All step-indexed channels share **one step axis**: integer steps
`0 .. n_steps`, where step `k` starts `k * dt_seconds` after step 0 under
the timebase convention.

- **`events`** (`time: "step"`): the export contract's `spikes.npz`
  ([schema.md](schema.md)). `t` is a step index in `[0, n_steps)`,
  `neuron_id` is in `[0, n_neurons)`, and `amp` is optional (default 1.0).
  The dtype rules and the loader are shared with `load_axon_export`.
- **`series`** (`time: "step"`): an `.npz` with `t` (`int64`, shape `[S]`)
  and `values` (`float32`/`float64`, shape `[S, C]`, C or Fortran order).
  - `t` holds strictly increasing step indices in `[0, n_steps)`.
  - Each row is an instantaneous reading at step `t[i]`. Nothing is defined
    between samples, so a renderer must not interpolate or hold values
    silently.
  - Values must be finite. A missing reading is an omitted sample, never a
    NaN. `S = 0` is a valid channel that recorded nothing.
  - `columns` says what column `j` is:
    - `{"index": "neuron", "neuron_ids": [...]}`: neuron `neuron_ids[j]`
      on the neuron axis. The ids are unique and `< n_neurons`, and may be
      a subset of the population in any order.
    - `{"index": "label", "labels": [...]}`: the named signal `labels[j]`
      (unique, non-blank). It is **not** a neuron index, e.g. `"dopamine"`
      or a sensor name.
  - `unit` (e.g. `"mV"`) is optional and is never guessed.
- **`document`** (`time: "static"`): a JSON value that describes the whole
  experiment and has no time axis. Version 1.0 constrains only the
  envelope (kind, producer, static time), not the document's inner schema.
  Per-kind payload schemas can be added later as minor versions.

### Optional, independently loadable

- A channel the manifest does not list is **absent**. Loading it fails with
  `ChannelAbsent`, never empty data. A listed channel with zero events or
  zero samples is honest silence, and the two are never confused.
- Opening a bundle reads only `bundle.json`. Each channel load reads and
  validates only its own payload against the manifest (columns, bounds,
  dtypes). A missing or corrupt payload fails that channel and leaves the
  others loadable.
- Payload paths must be relative, `/`-separated, with no `.`/`..`/empty
  segments and no `\` or `:`. The extension must match the layout (`.npz` or
  `.json`). A payload that resolves outside the bundle directory, for
  example through a symlink, is rejected.

## Versioning and compatibility

`bundle_version` is `MAJOR.MINOR`. A reader implementing `R.r` treats a
bundle at `B.b` as follows:

| Bundle | Reader outcome |
|---|---|
| `B != R` | fails to open: `UnsupportedVersion` |
| `B == R`, `b <= r` | opens with `Compatibility::Full` |
| `B == R`, `b > r` | opens with `Compatibility::NewerMinor`; newer additions are reported, never interpreted |
| malformed (`"1"`, `"1.0.0"`, `1.0`, missing) | invalid |

How each kind of unknown is handled, at any version:

- **Unknown keys** in any known object (root, timebase, dimensions,
  producers, channels, columns) are kept verbatim in
  `BundleManifest::unrecognized`, keyed by JSON pointer (e.g.
  `/channels/membrane/sample_mode`). They are never mapped onto a known
  field.
- **Unknown `kind`** values become `ChannelKind::Unrecognized(...)`. The
  channel can still be loaded through its layout, but a reader that asks
  for a known kind never selects it.
- **Unknown `layout`, `time`, `columns.index`, or channel-level `requires`**
  make that one channel `ChannelLayout::Unsupported`. Its raw entry is kept
  and it is listed, but loading it fails with `ChannelUnsupported`. Every
  other channel still loads.
- **Unknown top-level `requires`** or an **unknown `timebase.convention`**
  fail the whole bundle (`Unsupported`), because nothing in it can be read
  safely.
- **Malformed known fields** fail loud even in a newer-minor bundle. Examples:
  a string `dt_seconds`, `n_steps: 16.0`, a boolean seed, a missing nullable
  key, an undeclared producer, `time` disagreeing with `layout`, a kind
  paired with a layout it does not allow, or an unsafe path. A newer minor
  version may add things, but it may not redefine what a known field means.

Rules for evolving the format:

1. A **minor** version may only add: new optional keys, new kinds, layouts,
   time axes, column indexes, or conventions. Older readers report these
   or refuse them; they never misread them.
2. Anything that changes the meaning of an existing field needs a new
   **major** version, or a `requires` feature at bundle or channel level so
   that older readers refuse instead of misreading.
3. Producer-specific data goes in `extensions` (namespaced) or
   `producers.<id>.metadata`, not in new top-level keys.

Version 1.0 defines no `requires` features.

## Writer guarantees (`BundleBuilder`)

- Every producer and channel is validated when it is added: ids, declared
  producer, kind/layout pairing, bounds against the dimensions, series
  shape, finiteness, and strict `t` order. Invalid data is never written.
- The writer only emits what version 1.0 defines: known kinds, supported
  layouts, and an empty `requires`.
- The target directory must be absent (its parent must exist) or empty.
  Payloads are written first and `bundle.json` last, so an interrupted
  write never leaves a directory that opens as a bundle.
- Output is deterministic: sorted JSON keys and fixed archive timestamps
  make two writes of the same bundle byte-identical.
- Floats round-trip bit-exactly: the writer emits the shortest decimal
  that reads back as the same `f64`, and the reader parses JSON floats
  correctly rounded (serde_json `float_roundtrip`). Comparing `dt_seconds`
  across bundles and exports is therefore an exact equality check.
- `write` reopens the result through the reader before returning it.

## Adapter rule

The bundle is the durable interchange. Direct integration with a crate is
an optional convenience on top of it.

1. **Durable artifact interchange is first-class.** Any producer can write
   a bundle with nothing but JSON and `.npz`. It does not need to depend on
   spike-viz or on any sibling crate, and spike-viz can read every bundle
   without any adapter.
2. **Direct crate integrations may exist behind optional cargo features.**
   An adapter for a sibling crate (e.g. neuromod, plasticity-lab) depends
   on that crate only under a feature that is off by default. The default
   build depends on no sibling crate.
3. **Adapters produce bundle channels, nothing else.** Every adapter
   implements `BundleAdapter` and contributes through `BundleBuilder`.
   Sibling types never appear in the bundle schema or in renderer APIs, and
   renderers consume only bundle types (`SpikeEvents`, `Series`, documents).
4. **Anything a direct adapter contributes must survive a round trip
   through disk.** Its output must be writable as a bundle and loadable
   without the feature, and its tests should compare against that round
   trip. A feature-gated path may never be the only way to get a channel.
5. **Adapters never invent data.** They check their source against the
   bundle's shared timebase and dimensions and fail loud on a mismatch.
   They map unknown provenance to `null`, and they do not turn a field
   without defined time semantics into a channel.
   `BundleBuilder::contribute` is all-or-nothing: if an adapter fails, the
   builder is left unchanged.

## Wrapping the axon-encoder export (#26)

`AxonExportAdapter` / `bundle_from_axon_export` wrap any schema 1.0 export
case (`meta.json` + `spikes.npz`) without losing provenance:

| Export | Bundle |
|---|---|
| `dt_seconds` | `timebase.dt_seconds`, `convention: "sampling"` |
| `n_steps`, `n_neurons` | `dimensions` |
| `axon_encoder_version` | `producers.axon-encoder.version` (absent → `null`; a non-string fails loud) |
| `axon_encoder_git_sha` | `producers.axon-encoder.git_sha` (absent, null, or blank → `null`) |
| `seed` | `producers.axon-encoder.seed` |
| `synthetic` | `producers.axon-encoder.synthetic` (absent → `null`, never `false`) |
| `encoder` | `producers.axon-encoder.component` |
| the whole `meta.json` | `producers.axon-encoder.metadata`, verbatim (incl. `encoder_config`, `notes`, gain/modulator keys) |
| `spikes.npz` | channel `spikes`, kind `spike_events`, layout `events` |

Notes on the mapping:

- The wrap claims only the `sampling` convention. The export's `dt_seconds`
  means "seconds per displayed step" and is not always encoder-owned
  physical time.
- `stimulus.npy` is **not** converted. The export contract gives it no
  time/index semantics, so making it a step-indexed series would invent
  them. It stays next to the original export.
- Several exports can share one bundle when their axes match (for example
  rate and Poisson on the same stimulus), using
  `AxonExportAdapter::with_ids`. Mismatched geometry, `dt_seconds`, or
  convention fails loud. In particular an export cannot join a bundle whose
  timebase claims `physical` time, since the export never states that.

The tests wrap every checked-in fixture, write it, reopen it, and check
that provenance and events are unchanged. They also render
`rate/shared_sine_v1` from the bundle and from the original export, and
check that the pixels are identical.

## Rust API

```rust
use spike_viz_core::bundle::{bundle_from_axon_export, ChannelKind, ExperimentBundle};
use spike_viz_core::load_axon_export;

let case = load_axon_export("fixtures/axon-encoder/rate/shared_sine_v1")?;
bundle_from_axon_export(&case)?.write("out/rate_bundle")?;

let bundle = ExperimentBundle::open("out/rate_bundle")?;
for id in bundle.channels_of_kind(&ChannelKind::SpikeEvents) {
    let events = bundle.load_spike_events(id)?; // ChannelAbsent if missing, never empty
    println!("{id}: {} events", events.len());
}
```

Other entry points: `BundleBuilder` (`add_producer`, `add_spike_events`,
`add_series`, `add_document`, `add_extension`, `contribute`, `write`),
`ExperimentBundle` (`open`, `manifest`, `compatibility`, `channel`,
`load_series`, `load_document`), and `BundleError`. Its variants keep
*absent*, *unsupported*, *wrong layout*, and *invalid* apart.

The Python package does not read bundles. It remains the reference for the
export contract (see [rust-migration.md](rust-migration.md)).

## Non-goals

- No generic SNN runtime and no simulation inside spike-viz.
- No universal ML tensor format: there are three layouts, each with fixed
  time semantics.
- No requirement that sibling crates emit every channel. Bundles carry only
  what exists.
- No Python-side encoding or simulation, and no in-process bindings. The
  bridge stays export-first.

[RM-1925]: https://linear.app/rpd-34/issue/RM-1925
