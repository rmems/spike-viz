# spike-viz charter

Epic tracker: [rmems/spike-viz#2](https://github.com/rmems/spike-viz/issues/2)

`spike-viz` is a **PyTorch + CUDA** toolkit for visualizing spiking neural network
encodings and activity. Primary consumer for v0 / v0.2:
**[Limen-Neural/axon-encoder](https://github.com/Limen-Neural/axon-encoder)** hero
stills (wiki / README / product figures).

This charter locks product boundaries so implementation issues stay aligned.

## Product split

| Layer | Owner | Role |
|-------|--------|------|
| **Truth** | [axon-encoder](https://github.com/Limen-Neural/axon-encoder) (Rust) | Encode stimuli → spikes. Source of ground truth. |
| **Paint** | **spike-viz** (PyTorch + optional CUDA) | Load spike events/arrays → neon rasters / panels / PNG. |
| **Bridge (v0)** | Export contract | Rust writes sparse events (`.npz` / `.npy`); spike-viz **only loads + renders**. |

**Bridge default is export-first** until a later explicit decision changes it.
Do not open work that assumes in-process Rust bindings or a second encoder in Python.

## Goals (v0.1–v0.2)

- Honest visualizations of axon-encoder outputs (rate, Poisson, latency, population,
  temporal, predictive; neuromod gains).
- CPU path always works; CUDA optional for bloom / dense grids on machines with GPU
  (e.g. RTX 5080).
- Provenance on every figure: encoder, `dt`, seed, N, T, axon-encoder commit when known.
- Fail loud on missing/invalid data — **no silent synthetic spikes** when load fails.

## Non-goals (do not open work for these in v0)

- Second encoder implementation in Python
- pyo3 / in-process Rust bindings
- Interactive live UI (Makie-style; that is XAIDissect_Viz / Surrogate_Viz territory)
- MP4 / WebM video export
- PyPI release automation
- Porting CUDA.jl kernels from Surrogate_Viz / XAIDissect_Viz

## Milestones (see GitHub)

| Milestone | Intent |
|-----------|--------|
| `v0.1-foundation` | Package layout, schema, I/O, CPU raster, CI |
| `v0.2-hero-stills` | Bloom, hero panels, provenance captions |

## How to use this document

- Implementation issues should respect non-goals above; open a new epic if a
  non-goal becomes intentional product scope.
- README and `AGENTS.md` link here as the durable product boundary.
- Export format details live in the export-contract issue / docs when added (#6).
