# spike-viz

[![CI](https://github.com/rmems/spike-viz/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/spike-viz/actions/workflows/ci.yml)

PyTorch + CUDA toolkit for visualizing spiking neural network encodings and activity.

**Product split:** [axon-encoder](https://github.com/Limen-Neural/axon-encoder) owns
encoding **truth**; **spike-viz** only **paints** (load spike exports → rasters / PNG).
Bridge is export-first (`.npz` / `.npy`). See the charter for goals and non-goals.

## Rust-first core (migration in progress)

The paint path is migrating to a **Rust-first core** ([RM-1924], spike-viz
issue #30): a Cargo workspace with the crate `rust/spike-viz-core`
(reusable library + thin `spike-viz` CLI) that loads the same
`meta.json` + `spikes.npz` contract and paints the same truth-preserving CPU
raster. **The Python package below is not retired** — it stays the
operational reference implementation until the documented parity gates pass.
Rendering stays CPU-capable: the Rust build has no CUDA/NVIDIA dependency
whatsoever, and there are no in-process bindings (the bridge is still
export-first). Details: [docs/rust-migration.md](docs/rust-migration.md).

```bash
cargo test                                   # Rust core tests (CPU only)
mkdir -p out
cargo run --bin spike-viz -- render fixtures/axon-encoder/rate/shared_sine_v1 \
    --out out/rate.png --scale 8 --report out/rate.json
pytest tests/test_rust_parity.py -q          # Rust/Python semantic parity gate
```

[RM-1924]: https://linear.app/rpd-34/issue/RM-1924/rustcore-establish-rust-first-spike-viz-with-python-parity-bridge

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**CUDA is optional.** Install a CPU-only or CUDA build of PyTorch for your machine;
spike-viz must import and run loaders on CPU without a GPU. GPU paths (bloom, dense
acceleration) are additive later — never required for install.

```bash
python -c "import spike_viz; print(spike_viz.__version__)"
pytest -q
```

## Docs

| Doc | Purpose |
|-----|---------|
| [docs/CHARTER.md](docs/CHARTER.md) | Purpose, truth/paint split, non-goals |
| [docs/rust-migration.md](docs/rust-migration.md) | Rust-first core state, layout, parity gates |
| [docs/schema.md](docs/schema.md) | Sparse/dense spike schema |
| [docs/axon-encoder-export.md](docs/axon-encoder-export.md) | Export layout + golden fixtures |
| [docs/encoder-gallery.md](docs/encoder-gallery.md) | Seven-encoder gallery on one shared stimulus (Phase pending) |
| [docs/gain-triptych.md](docs/gain-triptych.md) | Real rate gain fixtures → captioned silence / normal / elevated PNG |
| [AGENTS.md](AGENTS.md) | Rules for coding agents |
| [REVIEW.md](REVIEW.md) | Local quality gate before merge |

## Quickstart: fixture → raster PNG

Run from the repository root after the install step above. This loads the checked-in
golden fixture and renders one CPU raster (no GPU needed; CUDA stays optional):

```python
from pathlib import Path

from spike_viz import load_axon_export, render_raster

case = load_axon_export("fixtures/axon-encoder/rate/tiny_synthetic")
print(case.meta["encoder"], len(case.events))

Path("out").mkdir(exist_ok=True)  # create the output directory on a clean checkout
render_raster(
    case.events,
    n_steps=case.meta["n_steps"],
    n_neurons=case.meta["n_neurons"],
    scale=16,  # the tiny fixture is 8 x 4 pixels; upscale so it is visible
    out_path="out/raster.png",
)
```

This writes `out/raster.png` (time horizontal, neuron vertical). The fixture is
synthetic and only exercises the schema; see
[docs/axon-encoder-export.md](docs/axon-encoder-export.md) for the export contract and
[docs/schema.md](docs/schema.md) for the spike layout.

## Rate vs Poisson hero

![Rate deterministic streaming vs Poisson stochastic raster](docs/images/rate-vs-poisson.png)

The same provenance-pinned Rust stimulus, shown with shared time/neuron scales:
deterministic **streaming** Rate vs seeded stochastic Poisson. Captions include
encoder, seed, `dt_seconds`, N, T, and the full axon-encoder commit. These fixtures
use different rate mappings, so this is not a parameter-matched experiment.

See [the hero guide](docs/rate-vs-poisson-hero.md) for the rendering command,
exact axon-encoder source, and fixture regeneration instructions. No Python
re-encoding, synthetic fallback, or GPU required.

## Encoder gallery

![Seven-slot axon-encoder gallery](docs/images/encoder-gallery.png)

`render_encoder_gallery` stacks Rate, Poisson, Latency, Population, Temporal,
Predictive and Phase rasters on one shared stimulus-sample axis, each panel with
a visible mode and provenance caption. The Phase slot is explicitly **pending**:
no PhaseEncoder export exists yet, and nothing is painted for it. See
[the gallery guide](docs/encoder-gallery.md).

## Rate gain triptych

The [rate gain triptych](docs/gain-triptych.md) (`render_gain_triptych`) is three
provenance-pinned Rust exports of one stimulus: an honestly empty silence panel,
identity gain, and elevated gain, with visible gain and provenance captions.

## Preferred-direction ring (explicit geometry required)

```python
from spike_viz import load_axon_export, render_direction_ring

# A source-supplied directional export; not the current scalar population fixture.
case = load_axon_export("path/to/directional-export")
image = render_direction_ring(case, window=(0, 8), glow=True, out_path="ring.png")
```

The export must include `meta["preferred_angles_radians"]`: N finite angles
indexed by neuron id, in radians (zero east, counterclockwise positive).
No angles or activity are inferred. The CPU/Pillow renderer uses event counts
in `[start, stop)` for dot size/brightness, including negative/zero-polarity
events. Omitting `window` displays the last export step, even if silent;
`glow=False` disables neon halos. Muted dots indicate zero events. Visible
and embedded PNG provenance includes encoder, dt, seed, N, T, source commit
(or explicitly `unknown`), synthetic status, and the displayed step interval.
For dimensionless encoders, dt is the export sampling convention.

**#17 remains fixture-blocked:** the checked-in PopulationEncoder export
has linear scalar Gaussian tuning, not preferred directions. It is rejected
rather than silently mapped onto a ring. See the
[export contract and remaining prerequisite](docs/axon-encoder-export.md#preferred-direction-geometry-optional-renderer-extension).

Package renderers and hero stills land under later `v0.1` / `v0.2` issues
([tracker](https://github.com/rmems/spike-viz/issues)).
