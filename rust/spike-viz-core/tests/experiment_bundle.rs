//! Experiment bundle contract tests: multi-channel reader/writer round
//! trips, schema compatibility, independent channel loading, explicit
//! time/index semantics, and wrapping the axon-encoder export.

mod common;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use common::*;
use serde_json::{json, Value};
use spike_viz_core::bundle::{
    bundle_from_axon_export, AxonExportAdapter, BundleBuilder, BundleError, ChannelKind,
    ChannelLayout, ChannelSpec, Columns, Compatibility, Dimensions, ExperimentBundle, LayoutKind,
    Producer, Series, TimeAxis, TimeConvention, Timebase, BUNDLE_FORMAT, BUNDLE_MANIFEST_FILE,
};
use spike_viz_core::{load_axon_export, render_raster, SpikeEvents};

// ---------------------------------------------------------------- helpers

/// An empty scratch directory. `temp_dir` paths are reused across
/// `cargo test` runs, and the writer rightly refuses non-empty targets, so
/// clear whatever an earlier run left behind.
fn scratch(tag: &str) -> PathBuf {
    let dir = temp_dir(&format!("bundle-{tag}"));
    fs::remove_dir_all(&dir).unwrap();
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn timebase() -> Timebase {
    Timebase {
        dt_seconds: 0.001,
        convention: TimeConvention::Physical,
    }
}

fn dims() -> Dimensions {
    Dimensions {
        n_steps: 16,
        n_neurons: 4,
    }
}

fn labels(names: &[&str]) -> Columns {
    Columns::Labels(names.iter().map(|name| name.to_string()).collect())
}

fn spec(kind: ChannelKind, producer: &str) -> ChannelSpec {
    ChannelSpec::new(kind, producer)
}

fn spikes() -> SpikeEvents {
    SpikeEvents::new(
        vec![0, 3, 3, 15],
        vec![1, 0, 3, 2],
        Some(vec![1.0, -1.0, 0.5, 0.0]),
    )
    .unwrap()
}

fn unit_spikes() -> SpikeEvents {
    SpikeEvents::new(vec![2, 9], vec![0, 0], None).unwrap()
}

fn membrane() -> Series {
    // A subset of neurons, in a non-sorted order: column j is neuron ids[j].
    Series::new(
        vec![0, 5, 15],
        vec![-65.25, -70.0, -52.5, -64.0, 0.1, -71.125],
        Columns::Neurons(vec![2, 0]),
        Some("mV".to_string()),
    )
    .unwrap()
}

fn modulators() -> Series {
    Series::new(
        vec![0, 1, 2, 3],
        vec![0.1, 0.0, 0.2, 0.05, 0.4, 0.05, 0.3, 0.1],
        labels(&["dopamine", "serotonin"]),
        None,
    )
    .unwrap()
}

fn weight_drift() -> Series {
    Series::new(vec![7], vec![1e-3], labels(&["mean_dw"]), None).unwrap()
}

fn power() -> Series {
    // Present, but no samples were recorded: honestly empty, not absent.
    Series::new(vec![], vec![], labels(&["board_power"]), Some("W".into())).unwrap()
}

fn topology() -> Value {
    json!({"edges": [{"pre": 0, "post": 1, "delay_steps": 2, "polarity": "excitatory"}]})
}

fn eval_metrics() -> Value {
    json!({"accuracy": 0.875, "epochs": 3})
}

fn transport() -> Value {
    json!({"dropped_frames": 0, "queue_depth_max": 12})
}

fn fpga_evidence() -> Value {
    json!({"bitstream": "example", "readback_matches": true})
}

/// A representative multi-producer bundle with every kind 1.0 defines.
fn multi_channel_builder() -> BundleBuilder {
    let mut builder = BundleBuilder::new(timebase(), dims()).unwrap();
    builder
        .add_producer(
            "encoder",
            Producer {
                name: "example-encoder".into(),
                version: Some("0.6.0".into()),
                git_sha: Some("908a34797c8aaff0a57e0ea4167aa00f3780c940".into()),
                seed: Some(u64::MAX),
                synthetic: Some(true),
                component: Some("rate".into()),
                metadata: Some(json!({"encoder_config": {"max_rate_hz": 400.0}})),
            },
        )
        .unwrap();
    builder
        .add_producer(
            "neuro",
            Producer {
                synthetic: Some(true),
                ..Producer::new("example-neuron-model")
            },
        )
        .unwrap();
    builder
        .add_producer("hw", Producer::new("example-board"))
        .unwrap();

    builder
        .add_spike_events(
            "spikes",
            spec(ChannelKind::SpikeEvents, "encoder").with_description("signed spikes"),
            spikes(),
        )
        .unwrap();
    builder
        .add_spike_events(
            "unit_spikes",
            spec(ChannelKind::SpikeEvents, "neuro"),
            unit_spikes(),
        )
        .unwrap();
    builder
        .add_series(
            "membrane",
            spec(ChannelKind::StateTrace, "neuro"),
            membrane(),
        )
        .unwrap();
    builder
        .add_series(
            "modulators",
            spec(ChannelKind::ModulatorTrace, "neuro"),
            modulators(),
        )
        .unwrap();
    builder
        .add_series(
            "weight_drift",
            spec(ChannelKind::Metrics, "neuro"),
            weight_drift(),
        )
        .unwrap();
    builder
        .add_document("eval", spec(ChannelKind::Metrics, "neuro"), eval_metrics())
        .unwrap();
    builder
        .add_document("topology", spec(ChannelKind::Topology, "neuro"), topology())
        .unwrap();
    builder
        .add_document(
            "transport",
            spec(ChannelKind::Diagnostics, "hw"),
            transport(),
        )
        .unwrap();
    builder
        .add_series("power", spec(ChannelKind::Telemetry, "hw"), power())
        .unwrap();
    builder
        .add_document(
            "fpga",
            spec(ChannelKind::AcceleratorEvidence, "hw"),
            fpga_evidence(),
        )
        .unwrap();
    builder
        .add_extension("example.org/run", json!({"operator_notes": "test"}))
        .unwrap();
    builder
}

/// Write the representative bundle into a fresh scratch directory.
fn written_bundle(tag: &str) -> (PathBuf, ExperimentBundle) {
    let dir = scratch(tag).join("bundle");
    let bundle = multi_channel_builder().write(&dir).unwrap();
    (dir, bundle)
}

fn read_manifest(dir: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(dir.join(BUNDLE_MANIFEST_FILE)).unwrap()).unwrap()
}

/// Rewrite `bundle.json` through `edit`.
fn edit_manifest(dir: &Path, edit: impl FnOnce(&mut Value)) {
    let mut manifest = read_manifest(dir);
    edit(&mut manifest);
    fs::write(
        dir.join(BUNDLE_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// A fresh written bundle with an edited manifest, reopened.
fn open_edited(tag: &str, edit: impl FnOnce(&mut Value)) -> Result<ExperimentBundle, BundleError> {
    let (dir, _) = written_bundle(tag);
    edit_manifest(&dir, edit);
    ExperimentBundle::open(&dir)
}

/// One hand-built `.npy` member: (name, descr, fortran_order, shape, data).
type RawNpy<'a> = (&'a str, &'a str, bool, &'a [u64], Vec<u8>);

fn write_raw_npz(path: &Path, members: &[RawNpy]) {
    let mut writer = npyz::npz::NpzWriter::create(path).unwrap();
    for (name, descr, fortran, shape, data) in members {
        let npy = npy_bytes(descr, *fortran, shape, data);
        let zip = writer.zip_writer();
        zip.start_file(
            format!("{name}.npy"),
            npyz::zip::write::FileOptions::default(),
        )
        .unwrap();
        zip.write_all(&npy).unwrap();
    }
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn assert_invalid<T: std::fmt::Debug>(result: Result<T, BundleError>, needle: &str) {
    match result {
        Err(BundleError::Invalid(message)) => {
            assert!(message.contains(needle), "{needle:?} not in {message:?}")
        }
        other => panic!("expected Invalid containing {needle:?}, got {other:?}"),
    }
}

// ------------------------------------------------------------ round trips

#[test]
fn multi_channel_bundle_round_trips() {
    let builder = multi_channel_builder();
    let (_, bundle) = written_bundle("round-trip");

    assert_eq!(bundle.compatibility(), Compatibility::Full);
    assert!(bundle.manifest().unrecognized.is_empty());
    // Every provenance field and channel entry survives the trip exactly.
    assert_eq!(bundle.manifest(), &builder.manifest());
    assert_eq!(bundle.manifest().producers["encoder"].seed, Some(u64::MAX));

    assert_eq!(bundle.load_spike_events("spikes").unwrap(), spikes());
    assert_eq!(
        bundle.load_spike_events("unit_spikes").unwrap(),
        unit_spikes()
    );
    assert!(bundle
        .load_spike_events("unit_spikes")
        .unwrap()
        .amp()
        .is_none());
    assert_eq!(bundle.load_series("membrane").unwrap(), membrane());
    assert_eq!(bundle.load_series("modulators").unwrap(), modulators());
    assert_eq!(bundle.load_series("weight_drift").unwrap(), weight_drift());
    assert_eq!(bundle.load_series("power").unwrap(), power());
    assert_eq!(bundle.load_document("eval").unwrap(), eval_metrics());
    assert_eq!(bundle.load_document("topology").unwrap(), topology());
    assert_eq!(bundle.load_document("transport").unwrap(), transport());
    assert_eq!(bundle.load_document("fpga").unwrap(), fpga_evidence());
    assert_eq!(
        bundle.manifest().extensions["example.org/run"],
        json!({"operator_notes": "test"})
    );
}

#[test]
fn time_and_index_semantics_are_explicit_on_disk() {
    let (dir, bundle) = written_bundle("semantics");
    let manifest = read_manifest(&dir);
    assert_eq!(manifest["format"], BUNDLE_FORMAT);
    assert_eq!(manifest["bundle_version"], "1.0");
    assert_eq!(manifest["requires"], json!([]));
    assert_eq!(
        manifest["timebase"],
        json!({"dt_seconds": 0.001, "convention": "physical"})
    );
    assert_eq!(
        manifest["dimensions"],
        json!({"n_steps": 16, "n_neurons": 4})
    );

    let channels = &manifest["channels"];
    assert_eq!(channels["spikes"]["layout"], "events");
    assert_eq!(channels["spikes"]["time"], "step");
    assert_eq!(channels["membrane"]["time"], "step");
    assert_eq!(
        channels["membrane"]["columns"],
        json!({"index": "neuron", "neuron_ids": [2, 0]})
    );
    assert_eq!(channels["membrane"]["unit"], "mV");
    assert_eq!(
        channels["modulators"]["columns"],
        json!({"index": "label", "labels": ["dopamine", "serotonin"]})
    );
    assert_eq!(channels["topology"]["layout"], "document");
    assert_eq!(channels["topology"]["time"], "static");

    // Unknown provenance is stated as explicit null, never omitted or guessed.
    let neuro = &manifest["producers"]["neuro"];
    for key in ["version", "git_sha", "seed"] {
        assert_eq!(neuro.get(key), Some(&Value::Null), "{key}");
    }
    assert_eq!(neuro["synthetic"], true);

    let topology = bundle.channel("topology").unwrap();
    assert_eq!(topology.layout.time_axis(), Some(TimeAxis::Static));
    assert_eq!(
        bundle.channel("membrane").unwrap().layout.kind(),
        Some(LayoutKind::Series)
    );
}

#[test]
fn writes_are_deterministic() {
    let builder = multi_channel_builder();
    let first = scratch("deterministic").join("a");
    let second = scratch("deterministic").join("b");
    builder.write(&first).unwrap();
    builder.write(&second).unwrap();
    let mut files = vec![PathBuf::from(BUNDLE_MANIFEST_FILE)];
    for entry in fs::read_dir(first.join("channels")).unwrap() {
        files.push(Path::new("channels").join(entry.unwrap().file_name()));
    }
    assert_eq!(files.len(), 11);
    for file in files {
        assert_eq!(
            fs::read(first.join(&file)).unwrap(),
            fs::read(second.join(&file)).unwrap(),
            "{}",
            file.display()
        );
    }
}

// ------------------------------------------- optional, independent channels

#[test]
fn absent_channel_is_an_error_not_empty_data() {
    let (_, bundle) = written_bundle("absent");
    assert!(bundle.channel("dendrites").is_none());
    for result in [
        bundle.load_series("dendrites").map(|_| ()),
        bundle.load_spike_events("dendrites").map(|_| ()),
        bundle.load_document("dendrites").map(|_| ()),
    ] {
        match result {
            Err(BundleError::ChannelAbsent { id }) => assert_eq!(id, "dendrites"),
            other => panic!("expected ChannelAbsent, got {other:?}"),
        }
    }
}

#[test]
fn present_silent_channel_differs_from_absent() {
    let mut builder = BundleBuilder::new(timebase(), dims()).unwrap();
    builder.add_producer("p", Producer::new("example")).unwrap();
    builder
        .add_spike_events(
            "spikes",
            spec(ChannelKind::SpikeEvents, "p"),
            SpikeEvents::new(vec![], vec![], None).unwrap(),
        )
        .unwrap();
    let bundle = builder.write(scratch("silent").join("bundle")).unwrap();
    assert!(bundle.load_spike_events("spikes").unwrap().is_empty());
    assert!(matches!(
        bundle.load_series("membrane"),
        Err(BundleError::ChannelAbsent { .. })
    ));
}

#[test]
fn channels_load_independently() {
    let (dir, _) = written_bundle("independent");
    fs::remove_file(dir.join("channels/membrane.npz")).unwrap();
    fs::write(dir.join("channels/topology.json"), "{not json").unwrap();
    fs::write(dir.join("channels/spikes.npz"), b"not a zip").unwrap();

    // Opening reads only the manifest.
    let bundle = ExperimentBundle::open(&dir).unwrap();
    match bundle.load_series("membrane") {
        Err(BundleError::Io(err)) => {
            assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
            assert!(err.to_string().contains("\"membrane\""), "{err}");
        }
        other => panic!("expected a NotFound Io error, got {other:?}"),
    }
    assert_invalid(bundle.load_document("topology"), "invalid JSON");
    assert!(matches!(
        bundle.load_spike_events("spikes"),
        Err(BundleError::Spike(_))
    ));

    // Every other channel still loads.
    assert_eq!(bundle.load_series("modulators").unwrap(), modulators());
    assert_eq!(
        bundle.load_spike_events("unit_spikes").unwrap(),
        unit_spikes()
    );
    assert_eq!(bundle.load_document("fpga").unwrap(), fpga_evidence());
}

#[test]
fn layout_mismatch_is_reported() {
    let (_, bundle) = written_bundle("mismatch");
    match bundle.load_series("spikes") {
        Err(BundleError::LayoutMismatch {
            id,
            expected,
            found,
        }) => {
            assert_eq!(
                (id.as_str(), expected, found.as_str()),
                ("spikes", "series", "events")
            );
        }
        other => panic!("expected LayoutMismatch, got {other:?}"),
    }
    assert!(matches!(
        bundle.load_spike_events("topology"),
        Err(BundleError::LayoutMismatch { .. })
    ));
    assert!(matches!(
        bundle.load_document("membrane"),
        Err(BundleError::LayoutMismatch { .. })
    ));
}

#[test]
fn channels_are_selected_by_kind() {
    let (_, bundle) = written_bundle("by-kind");
    let ids = |kind| {
        bundle
            .channels_of_kind(&kind)
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(ChannelKind::SpikeEvents), ["spikes", "unit_spikes"]);
    assert_eq!(ids(ChannelKind::Metrics), ["eval", "weight_drift"]);
    assert_eq!(ids(ChannelKind::StateTrace), ["membrane"]);
}

// ------------------------------------------------- schema compatibility

#[test]
fn newer_minor_opens_and_keeps_unknown_fields_apart() {
    let original = multi_channel_builder().manifest();
    let bundle = open_edited("newer-minor", |manifest| {
        manifest["bundle_version"] = json!("1.4");
        manifest["provenance_signature"] = json!({"alg": "ed25519"});
        manifest["timebase"]["epoch"] = json!("2026-01-01T00:00:00Z");
        manifest["producers"]["encoder"]["license"] = json!("Apache-2.0");
        manifest["channels"]["membrane"]["sample_mode"] = json!("hold");
        manifest["channels"]["membrane"]["columns"]["groups"] = json!([0, 0]);
    })
    .unwrap();

    assert_eq!(bundle.compatibility(), Compatibility::NewerMinor);
    let unrecognized = &bundle.manifest().unrecognized;
    assert_eq!(
        unrecognized.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "/channels/membrane/columns/groups",
            "/channels/membrane/sample_mode",
            "/producers/encoder/license",
            "/provenance_signature",
            "/timebase/epoch",
        ]
    );
    assert_eq!(unrecognized["/channels/membrane/sample_mode"], "hold");

    // Known fields read exactly as before; the unknown ones change nothing.
    assert_eq!(bundle.manifest().producers, original.producers);
    assert_eq!(bundle.manifest().channels, original.channels);
    assert_eq!(bundle.manifest().timebase, original.timebase);
    assert_eq!(bundle.load_series("membrane").unwrap(), membrane());
}

#[test]
fn unknown_fields_at_the_current_version_are_kept_apart_too() {
    let bundle = open_edited("current-minor", |manifest| {
        manifest["producers"]["hw"]["firmware"] = json!("3.1");
    })
    .unwrap();
    assert_eq!(bundle.compatibility(), Compatibility::Full);
    assert_eq!(
        bundle.manifest().unrecognized["/producers/hw/firmware"],
        "3.1"
    );
    assert_eq!(
        bundle.manifest().producers["hw"],
        Producer::new("example-board")
    );
}

#[test]
fn other_major_versions_are_rejected() {
    for version in ["2.0", "0.9", "10.0"] {
        let result = open_edited("major", |manifest| {
            manifest["bundle_version"] = json!(version);
        });
        match result {
            Err(BundleError::UnsupportedVersion { found }) => assert_eq!(found, version),
            other => panic!("{version}: expected UnsupportedVersion, got {other:?}"),
        }
    }
}

#[test]
fn malformed_versions_are_rejected() {
    for version in [
        json!("1"),
        json!("1.0.0"),
        json!("v1.0"),
        json!(""),
        json!(1.0),
    ] {
        let result = open_edited("bad-version", |manifest| {
            manifest["bundle_version"] = version.clone();
        });
        assert_invalid(result, "/bundle_version");
    }
    let result = open_edited("no-version", |manifest| {
        manifest.as_object_mut().unwrap().remove("bundle_version");
    });
    assert_invalid(result, "missing required key \"bundle_version\"");
}

#[test]
fn non_bundles_are_rejected() {
    assert_invalid(
        open_edited("format", |manifest| {
            manifest["format"] = json!("something-else");
        }),
        "not a spike-viz experiment bundle",
    );
    assert_invalid(
        open_edited("format-missing", |manifest| {
            manifest.as_object_mut().unwrap().remove("format");
        }),
        "missing required key \"format\"",
    );
    // An export case directory is not a bundle; it must be wrapped.
    match ExperimentBundle::open(fixture("rate", "tiny_synthetic")) {
        Err(BundleError::Io(err)) => {
            assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
            assert!(err.to_string().contains("bundle_from_axon_export"), "{err}");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
    assert!(matches!(
        ExperimentBundle::open(scratch("no-bundle").join("missing")),
        Err(BundleError::Io(_))
    ));
}

#[test]
fn unknown_required_feature_fails_the_bundle() {
    let result = open_edited("requires", |manifest| {
        manifest["requires"] = json!(["x-irregular-time"]);
    });
    match result {
        Err(BundleError::Unsupported(message)) => {
            assert!(message.contains("x-irregular-time"), "{message}")
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn channel_level_unknowns_make_only_that_channel_unsupported() {
    let bundle = open_edited("unsupported-channel", |manifest| {
        let channels = &mut manifest["channels"];
        channels["membrane"]["requires"] = json!(["x-masked-samples"]);
        channels["modulators"]["time"] = json!("wall_seconds");
        channels["topology"]["layout"] = json!("graph_v2");
        channels["weight_drift"]["columns"] = json!({"index": "synapse", "synapse_ids": [4]});
    })
    .unwrap();

    for (id, needle) in [
        ("membrane", "x-masked-samples"),
        ("modulators", "wall_seconds"),
        ("topology", "graph_v2"),
        ("weight_drift", "synapse"),
    ] {
        let channel = bundle.channel(id).unwrap();
        match &channel.layout {
            ChannelLayout::Unsupported { reason, raw } => {
                assert!(reason.contains(needle), "{id}: {reason}");
                assert!(raw.is_object(), "{id}: raw entry kept");
            }
            other => panic!("{id}: expected Unsupported, got {other:?}"),
        }
        assert_eq!(channel.layout.time_axis(), None);
        let result = match id {
            "topology" => bundle.load_document(id).map(|_| ()),
            _ => bundle.load_series(id).map(|_| ()),
        };
        assert!(
            matches!(result, Err(BundleError::ChannelUnsupported { .. })),
            "{id}: {result:?}"
        );
    }
    // Unsupported entries are not double-reported as unrecognized fields.
    assert!(bundle.manifest().unrecognized.is_empty());
    // The rest of the bundle is unaffected.
    assert_eq!(bundle.load_spike_events("spikes").unwrap(), spikes());
    assert_eq!(bundle.load_series("power").unwrap(), power());
}

#[test]
fn unknown_kind_is_kept_verbatim_and_loadable_by_layout() {
    let bundle = open_edited("unknown-kind", |manifest| {
        manifest["channels"]["topology"]["kind"] = json!("nir_graph");
    })
    .unwrap();
    let channel = bundle.channel("topology").unwrap();
    assert_eq!(channel.kind, ChannelKind::Unrecognized("nir_graph".into()));
    assert_eq!(bundle.load_document("topology").unwrap(), topology());
    // A reader asking for a known kind never picks it up.
    assert_eq!(bundle.channels_of_kind(&ChannelKind::Topology).count(), 0);
}

#[test]
fn malformed_known_fields_fail_loud_even_in_a_newer_minor() {
    type Edit = fn(&mut Value);
    let cases: Vec<(&str, Edit)> = vec![
        ("/timebase/dt_seconds", |m| {
            m["timebase"]["dt_seconds"] = json!("0.001")
        }),
        ("/timebase/dt_seconds", |m| {
            m["timebase"]["dt_seconds"] = json!(0)
        }),
        ("/dimensions/n_steps", |m| {
            m["dimensions"]["n_steps"] = json!(16.0)
        }),
        ("/dimensions/n_neurons", |m| {
            m["dimensions"]["n_neurons"] = json!(0)
        }),
        ("/producers/encoder/seed", |m| {
            m["producers"]["encoder"]["seed"] = json!(true)
        }),
        ("/producers/encoder/seed", |m| {
            m["producers"]["encoder"]["seed"] = json!(-1)
        }),
        ("/producers/encoder/git_sha", |m| {
            m["producers"]["encoder"]["git_sha"] = json!(123)
        }),
        ("/producers/encoder/git_sha", |m| {
            m["producers"]["encoder"]["git_sha"] = json!("")
        }),
        ("missing required key \"version\"", |m| {
            m["producers"]["neuro"]
                .as_object_mut()
                .unwrap()
                .remove("version");
        }),
        ("/producers/neuro/synthetic", |m| {
            m["producers"]["neuro"]["synthetic"] = json!("no")
        }),
        ("missing required key \"name\"", |m| {
            m["producers"]["hw"].as_object_mut().unwrap().remove("name");
        }),
        ("/producers", |m| m["producers"] = json!({})),
        ("/producers/bad~1id", |m| {
            m["producers"]["bad/id"] = m["producers"]["hw"].clone()
        }),
        ("undeclared producer", |m| {
            m["channels"]["spikes"]["producer"] = json!("ghost")
        }),
        ("/channels/spikes/kind", |m| {
            m["channels"]["spikes"]["kind"] = json!(5)
        }),
        ("is indexed by time", |m| {
            m["channels"]["spikes"]["time"] = json!("static")
        }),
        ("does not allow layout", |m| {
            m["channels"]["membrane"]["kind"] = json!("topology")
        }),
        ("must not contain", |m| {
            m["channels"]["spikes"]["path"] = json!("../escape.npz")
        }),
        ("must be relative", |m| {
            m["channels"]["spikes"]["path"] = json!("/abs.npz")
        }),
        ("must end in .npz", |m| {
            m["channels"]["spikes"]["path"] = json!("channels/spikes.json")
        }),
        ("out of range for n_neurons", |m| {
            m["channels"]["membrane"]["columns"]["neuron_ids"] = json!([7])
        }),
        ("duplicate", |m| {
            m["channels"]["modulators"]["columns"]["labels"] = json!(["da", "da"])
        }),
        ("/channels/membrane/unit", |m| {
            m["channels"]["membrane"]["unit"] = json!(3)
        }),
        ("/channels", |m| m["channels"] = json!([])),
        ("/extensions", |m| m["extensions"] = json!([])),
    ];
    for (needle, edit) in cases {
        let result = open_edited("malformed", |manifest| {
            manifest["bundle_version"] = json!("1.9");
            edit(manifest);
        });
        assert_invalid(result, needle);
    }

    let result = open_edited("convention", |manifest| {
        manifest["timebase"]["convention"] = json!("wallclock");
    });
    match result {
        Err(BundleError::Unsupported(message)) => assert!(message.contains("wallclock")),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn symlinked_payload_cannot_escape_the_bundle() {
    let (dir, _) = written_bundle("symlink");
    let outside = dir.parent().unwrap().join("outside.json");
    fs::write(&outside, "{}").unwrap();
    fs::remove_file(dir.join("channels/topology.json")).unwrap();
    std::os::unix::fs::symlink(&outside, dir.join("channels/topology.json")).unwrap();
    let bundle = ExperimentBundle::open(&dir).unwrap();
    assert_invalid(bundle.load_document("topology"), "outside the bundle");
}

// -------------------------------------------------- payload validation

#[test]
fn payloads_are_bounds_checked_against_the_manifest() {
    // Shrink the declared geometry under existing payloads.
    let bundle = open_edited("bounds", |manifest| {
        manifest["dimensions"]["n_steps"] = json!(10);
    })
    .unwrap();
    match bundle.load_spike_events("spikes") {
        Err(BundleError::Spike(err)) => {
            assert!(err.to_string().contains("t out of range"), "{err}")
        }
        other => panic!("expected an out-of-range spike error, got {other:?}"),
    }
    assert_invalid(
        bundle.load_series("membrane"),
        "out of range for n_steps=10",
    );
    assert_eq!(bundle.load_series("modulators").unwrap(), modulators());
}

#[test]
fn fortran_ordered_values_are_read_row_major() {
    let (dir, _) = written_bundle("fortran");
    // membrane is [[-65.25, -70.0], [-52.5, -64.0], [0.1, -71.125]];
    // Fortran order stores it column by column.
    write_raw_npz(
        &dir.join("channels/membrane.npz"),
        &[
            ("t", "<i8", false, &[3], i64_bytes(&[0, 5, 15])),
            (
                "values",
                "<f8",
                true,
                &[3, 2],
                f64_bytes(&[-65.25, -52.5, 0.1, -70.0, -64.0, -71.125]),
            ),
        ],
    );
    let bundle = ExperimentBundle::open(&dir).unwrap();
    assert_eq!(bundle.load_series("membrane").unwrap(), membrane());
}

#[test]
fn series_payloads_are_validated() {
    let t = || ("t", "<i8", false, &[2u64][..], i64_bytes(&[0, 1]));
    let cases: Vec<(&str, Vec<RawNpy>)> = vec![
        (
            "[n_samples, n_columns]",
            vec![t(), ("values", "<f8", false, &[4], f64_bytes(&[0.0; 4]))],
        ),
        (
            "[n_samples, n_columns]",
            vec![t(), ("values", "<f8", false, &[2, 3], f64_bytes(&[0.0; 6]))],
        ),
        (
            "float32 or float64",
            vec![t(), ("values", "<i8", false, &[2, 2], i64_bytes(&[0; 4]))],
        ),
        ("required arrays", vec![t()]),
        (
            "strictly increasing",
            vec![
                ("t", "<i8", false, &[2], i64_bytes(&[1, 1])),
                ("values", "<f8", false, &[2, 2], f64_bytes(&[0.0; 4])),
            ],
        ),
        (
            "finite",
            vec![
                t(),
                (
                    "values",
                    "<f8",
                    false,
                    &[2, 2],
                    f64_bytes(&[0.0, f64::NAN, 0.0, 0.0]),
                ),
            ],
        ),
    ];
    for (needle, members) in cases {
        let (dir, _) = written_bundle("series-payload");
        write_raw_npz(&dir.join("channels/modulators.npz"), &members);
        let bundle = ExperimentBundle::open(&dir).unwrap();
        assert_invalid(bundle.load_series("modulators"), needle);
    }
}

// ------------------------------------------------------ builder rules

#[test]
fn builder_rejects_invalid_channels() {
    let mut builder = multi_channel_builder();
    let events = || SpikeEvents::new(vec![0], vec![0], None).unwrap();
    let series = || Series::new(vec![0], vec![0.0], labels(&["x"]), None).unwrap();

    assert_invalid(
        builder.add_spike_events(
            "spikes",
            spec(ChannelKind::SpikeEvents, "encoder"),
            events(),
        ),
        "already present",
    );
    for bad_id in ["", "../x", "a/b", "a.b", "_x"] {
        assert_invalid(
            builder.add_document(bad_id, spec(ChannelKind::Topology, "neuro"), json!({})),
            "invalid id",
        );
    }
    assert_invalid(
        builder.add_document("t2", spec(ChannelKind::Topology, "ghost"), json!({})),
        "undeclared producer",
    );
    assert_invalid(
        builder.add_document(
            "t2",
            spec(ChannelKind::Unrecognized("nir_graph".into()), "neuro"),
            json!({}),
        ),
        "not defined by bundle_version 1.0",
    );
    assert_invalid(
        builder.add_document("t2", spec(ChannelKind::StateTrace, "neuro"), json!({})),
        "does not allow layout",
    );
    assert_invalid(
        builder.add_series("s2", spec(ChannelKind::SpikeEvents, "neuro"), series()),
        "does not allow layout",
    );
    assert_invalid(
        builder.add_document(
            "t2",
            spec(ChannelKind::Topology, "neuro").with_description(" "),
            json!({}),
        ),
        "description must be non-blank",
    );
    let late = SpikeEvents::new(vec![16], vec![0], None).unwrap();
    assert!(matches!(
        builder.add_spike_events("late", spec(ChannelKind::SpikeEvents, "neuro"), late),
        Err(BundleError::Spike(_))
    ));
    let late = Series::new(vec![16], vec![0.0], labels(&["x"]), None).unwrap();
    assert_invalid(
        builder.add_series("late", spec(ChannelKind::StateTrace, "neuro"), late),
        "out of range for n_steps=16",
    );
    let wide = Series::new(vec![0], vec![0.0], Columns::Neurons(vec![4]), None).unwrap();
    assert_invalid(
        builder.add_series("wide", spec(ChannelKind::StateTrace, "neuro"), wide),
        "out of range for n_neurons=4",
    );

    assert_invalid(
        builder.add_producer("neuro", Producer::new("again")),
        "already declared",
    );
    assert_invalid(builder.add_producer("p2", Producer::new(" ")), "non-blank");
    assert_invalid(
        builder.add_producer(
            "p2",
            Producer {
                git_sha: Some(String::new()),
                ..Producer::new("x")
            },
        ),
        "git_sha must be non-blank",
    );
    assert_invalid(
        builder.add_extension("example.org/run", json!(1)),
        "already set",
    );

    // Nothing above changed the bundle.
    assert_eq!(builder.manifest(), multi_channel_builder().manifest());
}

#[test]
fn builder_requires_valid_axes() {
    for dt_seconds in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let timebase = Timebase {
            dt_seconds,
            convention: TimeConvention::Sampling,
        };
        assert_invalid(BundleBuilder::new(timebase, dims()), "dt_seconds");
    }
    for (n_steps, n_neurons) in [(0, 1), (1, 0)] {
        let dims = Dimensions { n_steps, n_neurons };
        assert_invalid(BundleBuilder::new(timebase(), dims), ">= 1");
    }
}

#[test]
fn writer_refuses_unsafe_output() {
    let builder = multi_channel_builder();

    let occupied = scratch("occupied");
    fs::write(occupied.join("stale.npz"), b"old").unwrap();
    assert_invalid(builder.write(&occupied), "non-empty directory");

    let file = scratch("file-target").join("bundle");
    fs::write(&file, b"").unwrap();
    assert_invalid(builder.write(&file), "not a directory");

    assert!(matches!(
        builder.write(scratch("orphan").join("missing/parent/bundle")),
        Err(BundleError::Io(_))
    ));

    let empty = BundleBuilder::new(timebase(), dims()).unwrap();
    assert_invalid(
        empty.write(scratch("no-producers").join("b")),
        "at least one producer",
    );

    // An existing empty directory is fine.
    let empty_dir = scratch("empty-target");
    assert!(builder.write(&empty_dir).is_ok());
}

// ------------------------------------------ axon-encoder export wrapping

#[test]
fn every_fixture_wraps_without_losing_provenance() {
    for case_dir in all_fixture_cases() {
        let case = load_axon_export(&case_dir).unwrap();
        let out = scratch("wrap").join("bundle");
        let bundle = bundle_from_axon_export(&case).unwrap().write(&out).unwrap();
        let name = case_dir.display();

        assert_eq!(
            *bundle.timebase(),
            Timebase {
                dt_seconds: case.meta.dt_seconds,
                convention: TimeConvention::Sampling,
            },
            "{name}"
        );
        assert_eq!(
            *bundle.dimensions(),
            Dimensions {
                n_steps: case.meta.n_steps,
                n_neurons: case.meta.n_neurons,
            },
            "{name}"
        );

        let producer = &bundle.manifest().producers[AxonExportAdapter::DEFAULT_PRODUCER_ID];
        assert_eq!(producer.name, "axon-encoder");
        assert_eq!(producer.git_sha, case.meta.axon_encoder_git_sha, "{name}");
        assert_eq!(
            producer.version.as_deref(),
            case.meta
                .raw
                .get("axon_encoder_version")
                .and_then(Value::as_str),
            "{name}"
        );
        assert_eq!(producer.seed, Some(case.meta.seed));
        assert_eq!(producer.synthetic, case.meta.synthetic);
        assert_eq!(
            producer.component.as_deref(),
            Some(case.meta.encoder.as_str())
        );
        // The original meta.json is carried verbatim, extra keys and all.
        assert_eq!(producer.metadata.as_ref(), Some(&case.meta.raw), "{name}");

        let channel = bundle
            .channel(AxonExportAdapter::DEFAULT_CHANNEL_ID)
            .unwrap();
        assert_eq!(channel.kind, ChannelKind::SpikeEvents);
        assert_eq!(channel.producer, AxonExportAdapter::DEFAULT_PRODUCER_ID);
        assert_eq!(
            bundle
                .load_spike_events(AxonExportAdapter::DEFAULT_CHANNEL_ID)
                .unwrap(),
            case.events,
            "{name}"
        );
        // Nothing beyond the export is invented.
        assert_eq!(bundle.manifest().channels.len(), 1, "{name}");
    }
}

#[test]
fn wrapped_export_renders_identically() {
    let case = load_axon_export(fixture("rate", "shared_sine_v1")).unwrap();
    let bundle = bundle_from_axon_export(&case)
        .unwrap()
        .write(scratch("wrap-render").join("bundle"))
        .unwrap();
    let events = bundle.load_spike_events("spikes").unwrap();
    let dims = *bundle.dimensions();
    let from_bundle =
        render_raster(&events, dims.n_steps as usize, dims.n_neurons as usize, 4).unwrap();
    let from_export = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        4,
    )
    .unwrap();
    assert_eq!(from_bundle.pixels(), from_export.pixels());
    assert_eq!(from_bundle.peak(), from_export.peak());
}

#[test]
fn silent_export_wraps_as_a_present_empty_channel() {
    let case = load_axon_export(fixture("rate", "shared_sine_gain_0_v1")).unwrap();
    let bundle = bundle_from_axon_export(&case)
        .unwrap()
        .write(scratch("wrap-silent").join("bundle"))
        .unwrap();
    assert!(bundle.channel("spikes").is_some());
    assert!(bundle.load_spike_events("spikes").unwrap().is_empty());
    assert_eq!(
        bundle.manifest().producers["axon-encoder"].synthetic,
        Some(false)
    );
}

#[test]
fn two_exports_share_one_bundle() {
    let rate = load_axon_export(fixture("rate", "shared_sine_v1")).unwrap();
    let poisson = load_axon_export(fixture("poisson", "shared_sine_v1")).unwrap();
    let mut builder = bundle_from_axon_export(&rate).unwrap();
    builder
        .contribute(&AxonExportAdapter::new(&poisson).with_ids("axon-poisson", "poisson_spikes"))
        .unwrap();
    let bundle = builder.write(scratch("wrap-two").join("bundle")).unwrap();

    let producers = &bundle.manifest().producers;
    assert_eq!(producers["axon-encoder"].component.as_deref(), Some("rate"));
    assert_eq!(
        producers["axon-poisson"].component.as_deref(),
        Some("poisson")
    );
    assert_eq!(producers["axon-poisson"].seed, Some(poisson.meta.seed));
    assert_eq!(
        bundle.channel("poisson_spikes").unwrap().producer,
        "axon-poisson"
    );
    assert_eq!(bundle.load_spike_events("spikes").unwrap(), rate.events);
    assert_eq!(
        bundle.load_spike_events("poisson_spikes").unwrap(),
        poisson.events
    );
}

#[test]
fn adapter_rejects_mismatched_axes_and_rolls_back() {
    let rate = load_axon_export(fixture("rate", "shared_sine_v1")).unwrap();
    let population = load_axon_export(fixture("population", "shared_sine_v1")).unwrap();
    let mut builder = bundle_from_axon_export(&rate).unwrap();
    let before = builder.manifest();
    assert_invalid(
        builder.contribute(&AxonExportAdapter::new(&population).with_ids("pop", "pop_spikes")),
        "does not match the bundle",
    );
    assert_eq!(builder.manifest(), before);

    let other_dt = Timebase {
        dt_seconds: 0.002,
        convention: TimeConvention::Sampling,
    };
    let mut builder = BundleBuilder::new(
        other_dt,
        Dimensions {
            n_steps: 64,
            n_neurons: 8,
        },
    )
    .unwrap();
    assert_invalid(
        builder.contribute(&AxonExportAdapter::new(&rate)),
        "dt_seconds",
    );
    assert!(builder.producers().is_empty());
    assert_eq!(builder.channel_ids().count(), 0);

    // Re-contributing under the same ids is a conflict, not a silent merge.
    let mut builder = bundle_from_axon_export(&rate).unwrap();
    assert_invalid(
        builder.contribute(&AxonExportAdapter::new(&rate)),
        "already declared",
    );
}

/// Export case with the tiny synthetic spikes and an edited `meta.json`.
fn edited_case(tag: &str, edit: impl FnOnce(&mut Value)) -> spike_viz_core::AxonExportCase {
    let dir = scratch(tag).join("case");
    let mut meta = valid_meta();
    edit(&mut meta);
    write_case(&dir, &meta, &[0, 1], &[1, 0], None);
    load_axon_export(&dir).unwrap()
}

#[test]
fn unknown_export_provenance_stays_unknown() {
    // No synthetic flag, no SHA, no version: all stay null, never guessed.
    let case = edited_case("wrap-unknown", |meta| {
        meta.as_object_mut().unwrap().remove("synthetic");
    });
    let producer = AxonExportAdapter::new(&case).producer().unwrap();
    assert_eq!(
        (producer.synthetic, producer.git_sha, producer.version),
        (None, None, None)
    );

    let case = edited_case("wrap-blank-sha", |meta| {
        meta["axon_encoder_git_sha"] = json!("  ");
    });
    assert_eq!(
        AxonExportAdapter::new(&case).producer().unwrap().git_sha,
        None
    );

    let bundle = bundle_from_axon_export(&case)
        .unwrap()
        .write(scratch("wrap-unknown-write").join("bundle"))
        .unwrap();
    let manifest = read_manifest(bundle.root());
    assert_eq!(
        manifest["producers"]["axon-encoder"]["git_sha"],
        Value::Null
    );
    // The blank original is still in the verbatim metadata.
    assert_eq!(
        manifest["producers"]["axon-encoder"]["metadata"]["axon_encoder_git_sha"],
        "  "
    );
}

#[test]
fn malformed_export_version_fails_loud() {
    let case = edited_case("wrap-bad-version", |meta| {
        meta["axon_encoder_version"] = json!(6);
    });
    assert_invalid(bundle_from_axon_export(&case), "axon_encoder_version");
}

/// A timestep whose shortest decimal form serde_json's default
/// (best-effort) float parser does not read back bit-exactly; the crate
/// enables `float_roundtrip` so every JSON float parses correctly rounded.
const AWKWARD_DT: f64 = 7.753519050239737e-8;

#[test]
fn dt_seconds_round_trips_bit_exactly() {
    let timebase = Timebase {
        dt_seconds: AWKWARD_DT,
        convention: TimeConvention::Physical,
    };
    let mut builder = BundleBuilder::new(timebase, dims()).unwrap();
    builder.add_producer("p", Producer::new("example")).unwrap();
    let bundle = builder.write(scratch("awkward-dt").join("bundle")).unwrap();
    assert_eq!(bundle.timebase().dt_seconds.to_bits(), AWKWARD_DT.to_bits());
}

#[test]
fn adapter_matches_a_literal_timebase_against_parsed_meta() {
    // meta.json carries the dt as JSON text; the bundle axes come from a
    // Rust literal. Correctly rounded parsing makes them the same f64.
    let case = edited_case("wrap-awkward-dt", |meta| {
        meta["dt_seconds"] = json!(AWKWARD_DT);
    });
    assert_eq!(case.meta.dt_seconds.to_bits(), AWKWARD_DT.to_bits());
    let timebase = Timebase {
        dt_seconds: AWKWARD_DT,
        convention: TimeConvention::Sampling,
    };
    let dims = Dimensions {
        n_steps: 2,
        n_neurons: 2,
    };
    let mut builder = BundleBuilder::new(timebase, dims).unwrap();
    builder.contribute(&AxonExportAdapter::new(&case)).unwrap();
    assert!(builder.channel_ids().any(|id| id == "spikes"));
}
