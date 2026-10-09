# spike-viz

[![CI](https://github.com/rmems/spike-viz/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/spike-viz/actions/workflows/ci.yml)

PyTorch + CUDA toolkit for visualizing spiking neural network encodings and activity.

**Product split:** [axon-encoder](https://github.com/Limen-Neural/axon-encoder) owns
encoding **truth**; **spike-viz** only **paints** (load spike exports → rasters / PNG).
Bridge is export-first (`.npz` / `.npy`). See the charter for goals and non-goals.

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
| [docs/schema.md](docs/schema.md) | Sparse/dense spike schema |
| [docs/axon-encoder-export.md](docs/axon-encoder-export.md) | Export layout + golden fixtures |
| [AGENTS.md](AGENTS.md) | Rules for coding agents |
| [REVIEW.md](REVIEW.md) | Local quality gate before merge |

## Quick load (export case)

```python
from spike_viz import load_axon_export

case = load_axon_export("fixtures/axon-encoder/rate/tiny_synthetic")
print(case.meta["encoder"], len(case.events))
```

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
