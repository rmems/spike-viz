# Rust-first core migration (in progress)

Status: **in progress** (tracked as [RM-1924] / spike-viz issue #30, parent epic
[#29] / issue #29). The Rust core established here is the **future primary**
spike-viz implementation. The Python package in `src/spike_viz/` is **not
retired**: it stays the operational **reference implementation** until the
parity gates below pass and a later issue flips the primary path.

Charter: [CHARTER.md](CHARTER.md). Export contract:
[axon-encoder-export.md](axon-encoder-export.md).

## What this means for the product split

- **Truth stays with axon-encoder.** Nothing here encodes spikes.
- **Paint migrates to Rust.** The Rust core loads the same export contract
  (`meta.json` + `spikes.npz`) and paints CPU rasters.
- **The bridge stays export-first.** No pyo3, no in-process bindings in either
  direction; the on-disk files remain the only handoff. This does **not**
  change the charter's "no pyo3" non-goal — the bridge is a subprocess CLI
  plus a JSON report, not a binding.
- **CPU always works.** The Rust build has no CUDA/NVIDIA dependency at all;
  rendering is pure CPU. CUDA remains an optional Python acceleration path.

## Layout

```text
Cargo.toml                 # workspace root (repo root); `cargo test` just works
rust/spike-viz-core/       # the reusable Rust library + thin CLI
  src/lib.rs               # public API: load_axon_export, render_raster, ...
  src/events.rs            # SpikeEvents (COO sparse schema)
  src/io.rs                # load_sparse: .npz arrays + dtype rules
  src/meta.rs              # meta.json validation (schema 1.0)
  src/export.rs            # AxonExportCase: case directory loader
  src/render.rs            # sparse_to_dense + CPU raster + PNG encode
  src/report.rs            # SemanticReport (parity bridge payload)
  src/bin/spike-viz.rs     # `spike-viz render` CLI
  tests/                   # contract, loader, raster, report, CLI tests
```

Python (`src/spike_viz/`) and Rust coexist during parity work; no Python
module was deleted or moved. Rust and Python never share state in-process.

## What is ported, what is not

Ported to Rust (parity with the Python reference):

- `load_sparse` (`.npz` member reading, dtype rules incl. integral floats,
  unsigned/bool widening, loud rejection of complex/object/structured dtypes)
- `load_meta` / `load_axon_export` (required keys, strict types, `schema_version`
  limited to `"1.0"`, geometry `>= 1`, finite positive `dt_seconds`,
  event-vs-geometry bounds)
- `sparse_to_dense` / `render_raster` (accumulate, zero-amp → unit-amp at
  render time only, peak normalization, clip to `[0, 1]`, NumPy-style
  half-to-even rounding, nearest-neighbor upscaling, black background)
- PNG encode (8-bit RGB, same mode as the Python renderer)
- `SemanticReport`: machine-readable events / geometry / peak / final
  intensities for the parity gate

Still Python-only (port in later issues, one concern per issue):

- `load_dense` (`.npy` / `.npz` dense grids)
- bloom / tone-map, hero stills, gain triptych, encoder gallery,
  direction ring, provenance captions (`caption.py`)

Non-goals for this migration step (unchanged from the charter): no sibling
adapters to other ecosystem crates, no live UI, no encoder/neuron/plasticity
semantics, no Python deletion, no crates.io publishing.

## Parity gates

Three gates must all be green before any "primary" flip; they compare
**semantic output** — event lists, grid geometry (N, T, width, height), and
intensity rules (peak normalization, final `u8` intensities). Pixel-perfect
PNG byte identity is *not* a gate (compressors differ); decoded pixels are
compared within ±1 intensity level.

| Gate | Command | What it proves |
|------|---------|----------------|
| Rust invariants | `cargo test --workspace` | contract loading, raster truth rules, invalid data fails loud, empty/silent activity, CLI |
| Python regression | `pytest -q` | the reference implementation is unchanged and still green |
| Cross-language parity | `pytest tests/test_rust_parity.py -q` | Rust renders the checked-in rate fixtures (`tiny_synthetic`, `shared_sine_v1`, silent `shared_sine_gain_0_v1`) with the same events, geometry, and intensities as Python |

The parity test runs the CLI end-to-end:

```bash
cargo run --bin spike-viz -- render fixtures/axon-encoder/rate/shared_sine_v1 \
    --out out/rate.png --scale 8 --report out/rate.json
```

The `--report` JSON is the parity bridge consumed by
`tests/test_rust_parity.py`. The gate skips cleanly when no Rust toolchain
is present, the same way CUDA tests skip without a GPU.

## Documented divergences

The export contract itself is language-neutral; these loader-level edges are
known differences, kept loud on both sides:

- **Seeds** are `u64` in Rust; Python's arbitrary-precision ints would accept
  values above `u64::MAX`. Such seeds fail the Rust load.
- **Very large `dt_seconds`** (integers beyond float range) are rejected by
  both, but Rust reports a JSON parse error rather than a field error.
- **`float16`/`float128` arrays** are rejected by the Rust reader (the npy
  reader covers f32/f64, all integer widths, unsigned, bool); Python accepts
  them when integral. The contract fixtures use `int64`/`float32`.
- **`allow_pickle`**: object-dtype arrays fail to load in both languages; the
  contract has no pickle channel.
- **PNG bytes** may differ (compression), so parity compares decoded pixels,
  not file hashes.

[RM-1924]: https://linear.app/rpd-34/issue/RM-1924/rustcore-establish-rust-first-spike-viz-with-python-parity-bridge
[#29]: https://linear.app/rpd-34/issue/RM-1923/epic-rust-first-spike-viz-and-neuromorphic-ecosystem-adapters
