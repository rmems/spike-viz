# AGENTS.md

Instructions for coding agents working in **spike-viz**.

## Project identity

`spike-viz` is a **PyTorch + optional CUDA** toolkit that **paints** spike activity:
load axon-encoder (or contract-compatible) exports → neon rasters / panels / PNG.

It is **not** an encoder. Ground-truth encoding lives in
[Limen-Neural/axon-encoder](https://github.com/Limen-Neural/axon-encoder) (Rust).

Read **[docs/CHARTER.md](docs/CHARTER.md)** before expanding scope.

## Product split (non-negotiable)

| Layer | Owner | Role |
|-------|--------|------|
| Truth | axon-encoder | Encode stimuli → spikes |
| Paint | spike-viz | Load + render only |
| Bridge (v0) | Export files | `.npz` / `.npy` (or documented layout); no pyo3 |

Default bridge is **export-first**. Do not add in-process Rust bindings, a Python
encoder, or silent synthetic data when load fails.

## Stack

- Language: Python 3.10+ (`src/spike_viz/` layout)
- Core deps: `numpy`, `torch`, `pillow` — **CUDA optional** (CPU torch is fine)
- Tests: `pytest` on CPU (`pip install -e ".[dev]"`); CUDA tests must skip cleanly when no GPU
- Export layout: `docs/axon-encoder-export.md` + `fixtures/axon-encoder/`
- Interactive Makie-style UI is **out of scope** (other viz repos)

## Non-negotiable rules

1. **Respect the charter** — especially non-goals in `docs/CHARTER.md`.
2. **Fail loud** on missing/invalid spike data; never invent spikes to “make a pretty PNG.”
3. **Provenance** — figures that claim scientific meaning need encoder / `dt` / seed / N / T
   (and axon-encoder commit when known) once provenance work lands.
4. **CPU always works** — CUDA is an optional acceleration path, not a requirement to render.
5. **Small, reviewable diffs** — one issue / concern per PR when practical.
6. **No secrets** — do not commit tokens, absolute machine-only paths into library code,
   or private telemetry.
7. **Do not port** CUDA.jl / Surrogate_Viz / XAIDissect_Viz kernels wholesale; paint paths
   here are PyTorch-first.
8. **No video export, no PyPI automation, no live UI** in v0 unless the charter is updated.

## Working style

- Prefer GitHub issues in this repo for tracked work (`phase:v0.1` / `phase:v0.2`, area labels).
- Link PRs with `Fixes #N` when an issue is fully done.
- Before claiming done, run the quality bar in **[REVIEW.md](REVIEW.md)**.
- When package / tests do not exist yet, do not invent a full app structure beyond the
  issue you are implementing (skeleton is its own issue).

## Suggested v0.1 order

Charter (this foundation) → export contract → package skeleton → schema/loaders →
CPU raster → README quickstart → CI.

Do not jump to v0.2 hero stills until v0.1 I/O and CPU raster exist unless the user
explicitly prioritizes otherwise.

## Related repos

- Truth: https://github.com/Limen-Neural/axon-encoder
- Interactive / other viz (not this repo): Surrogate_Viz, XAIDissect_Viz
