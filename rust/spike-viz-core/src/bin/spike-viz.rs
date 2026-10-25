//! Thin parity-bridge CLI for the Rust-first spike-viz core.
//!
//! Intentionally minimal (`render` only): the library is the reusable core,
//! and this binary exists so the documented Rust/Python parity gates
//! (see `docs/rust-migration.md` and `tests/test_rust_parity.py`) can run
//! the Rust render end-to-end. No live UI, no video, no GPU requirement.

use std::path::PathBuf;
use std::process::ExitCode;

use spike_viz_core::{load_axon_export, render_raster, write_png, SemanticReport};

const USAGE: &str = "spike-viz (Rust core) — render axon-encoder spike exports to CPU rasters

USAGE:
    spike-viz render <CASE_DIR> [--out <PNG>] [--scale <N>] [--report <JSON>]

ARGS:
    <CASE_DIR>      Export case directory holding meta.json + spikes.npz
                    (optional stimulus.npy is ignored by the raster path).

OPTIONS:
    --out <PNG>     Write the rendered raster as an 8-bit RGB PNG.
                    The parent directory must already exist.
    --scale <N>     Nearest-neighbor upscale factor (integer >= 1). Default 1.
    --report <JSON> Write the semantic parity report (events, grid geometry,
                    peak, final intensities) as JSON. Must differ from --out.

At least one of --out / --report is required. Rendering is CPU-only.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(message)) => {
            eprintln!("error: {message}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(CliError::Spike(err)) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

enum CliError {
    Usage(String),
    Spike(spike_viz_core::SpikeIOError),
}

impl From<spike_viz_core::SpikeIOError> for CliError {
    fn from(err: spike_viz_core::SpikeIOError) -> Self {
        CliError::Spike(err)
    }
}

impl From<std::io::Error> for CliError {
    fn from(err: std::io::Error) -> Self {
        CliError::Spike(spike_viz_core::SpikeIOError::Io(err))
    }
}

impl From<serde_json::Error> for CliError {
    fn from(err: serde_json::Error) -> Self {
        CliError::Spike(spike_viz_core::SpikeIOError::io(format!(
            "failed to write report: {err}"
        )))
    }
}

struct RenderArgs {
    case_dir: PathBuf,
    out: Option<PathBuf>,
    report: Option<PathBuf>,
    scale: usize,
}

fn run(args: &[String]) -> Result<(), CliError> {
    let parsed = parse_args(args)?;
    let case = load_axon_export(&parsed.case_dir)?;
    let raster = render_raster(
        &case.events,
        case.meta.n_steps as usize,
        case.meta.n_neurons as usize,
        parsed.scale,
    )?;

    if let Some(out) = &parsed.out {
        write_png(&raster, out)?;
    }
    if let Some(report_path) = &parsed.report {
        let report = SemanticReport::from_render(&case, &raster)?;
        let file = std::fs::File::create(report_path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        // One trailing newline keeps the report plain-text diffable.
        use std::io::Write;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }

    println!(
        "encoder={} | N={} | T={} | events={} | scale={} | peak={} | out={} | report={}",
        case.meta.encoder,
        case.meta.n_neurons,
        case.meta.n_steps,
        case.events.len(),
        parsed.scale,
        raster.peak(),
        parsed
            .out
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        parsed
            .report
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    Ok(())
}

/// Parsed flag values before the cross-field validations run.
struct FlagValues {
    case_dir: Option<PathBuf>,
    out: Option<PathBuf>,
    report: Option<PathBuf>,
    scale: usize,
}

fn parse_args(args: &[String]) -> Result<RenderArgs, CliError> {
    let mut iter = args.iter();
    let subcommand = iter
        .next()
        .ok_or_else(|| usage("missing subcommand; expected `render <CASE_DIR>`"))?;
    if subcommand != "render" {
        return Err(usage(format!(
            "unknown subcommand {subcommand:?}; expected `render <CASE_DIR>`"
        )));
    }
    finalize(parse_flags(&mut iter)?)
}

/// Validate the parsed flags and assemble the render arguments.
fn finalize(flags: FlagValues) -> Result<RenderArgs, CliError> {
    let case_dir = flags
        .case_dir
        .ok_or_else(|| usage("missing <CASE_DIR> argument"))?;
    if flags.out.is_none() && flags.report.is_none() {
        return Err(usage(
            "nothing to do: pass --out <PNG> and/or --report <JSON>",
        ));
    }
    if let (Some(out), Some(report)) = (&flags.out, &flags.report) {
        if same_target(out, report) {
            return Err(usage(
                "--out and --report must be different paths: writing the report \
                 would truncate the just-written PNG",
            ));
        }
    }
    Ok(RenderArgs {
        case_dir,
        out: flags.out,
        report: flags.report,
        scale: flags.scale,
    })
}

/// Consume the flags after the `render` subcommand.
fn parse_flags<'a>(mut iter: impl Iterator<Item = &'a String>) -> Result<FlagValues, CliError> {
    let mut flags = FlagValues {
        case_dir: None,
        out: None,
        report: None,
        scale: 1,
    };
    while let Some(arg) = iter.next() {
        let (flag, inline_value) = match arg.split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (arg.clone(), None),
        };
        let mut value = |flag: &str| -> Result<String, CliError> {
            inline_value
                .clone()
                .or_else(|| iter.next().cloned())
                .ok_or_else(|| usage(format!("missing value for {flag}")))
        };
        match flag.as_str() {
            "--out" => flags.out = Some(PathBuf::from(value("--out")?)),
            "--report" => flags.report = Some(PathBuf::from(value("--report")?)),
            "--scale" => {
                let raw = value("--scale")?;
                flags.scale = raw
                    .parse()
                    .map_err(|_| usage(format!("--scale must be an integer >= 1, got {raw:?}")))?;
            }
            _ if flag.starts_with("--") => {
                return Err(usage(format!("unknown option {flag:?}")));
            }
            _ => {
                if flags.case_dir.is_some() {
                    return Err(usage(format!("unexpected argument {arg:?}")));
                }
                flags.case_dir = Some(PathBuf::from(arg));
            }
        }
    }
    Ok(flags)
}

/// Build a usage error (exit code 2).
fn usage(message: impl Into<String>) -> CliError {
    CliError::Usage(message.into())
}

/// True when both paths point at the same file.
///
/// When both paths exist, fully resolved paths are compared (covers
/// symlinks and differently-spelled aliases). Otherwise — the common case,
/// since `--out` / `--report` targets usually do not exist yet — the paths
/// are lexically absolutized and compared without touching the filesystem.
fn same_target(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => std::path::absolute(a).ok() == std::path::absolute(b).ok(),
    }
}
