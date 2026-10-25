@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Project identity, stack, product split, and non-negotiable rules are in `AGENTS.md`
(imported above); do not duplicate them here.

## Commands

Run from the repo root inside a venv. "Verified" means it ran successfully in a fresh checkout on Python 3.13 (Linux, CPU-only torch, no GPU).

| Command | Status |
|---|---|
| `pip install --index-url https://download.pytorch.org/whl/cpu torch` then `pip install -e ".[dev]"` | verified (via `uv pip`) |
| `python -c "import spike_viz; print(spike_viz.__version__)"` | verified |
| `python -c "from spike_viz import load_axon_export; print(load_axon_export('fixtures/axon-encoder/rate/tiny_synthetic').meta['encoder'])"` | verified |
| `pytest -q -rs` | verified: 177 passed, 2 skipped (CUDA) |
| `cargo test --workspace` (from the repo root) | verified: 101 passed (Rust core; CPU-only, no CUDA needed) |
| `pytest tests/test_rust_parity.py -q` | verified: 4 passed (renders the checked-in rate fixtures with the Rust CLI; skips cleanly without `cargo`) |
| CUDA-marked tests (`tests/test_cuda.py`, one case in `tests/test_bloom.py`) | not run here (no GPU); they skip cleanly |
| `scripts/generate_fixtures.sh`, `scripts/generate_gain_fixtures.sh` | not run here (regenerate fixtures from axon-encoder; need the Rust toolchain and an axon-encoder checkout) |

No linter or formatter is configured (`REVIEW.md` leaves ruff optional), so there is no lint gate to run.
CI (`.github/workflows/ci.yml`) runs the import smoke test, the golden fixture load, and `pytest -q -rs` on Python 3.10 and 3.12 with CPU torch; a `rust` job runs `cargo test --workspace`.
Before claiming done, follow the gate in `REVIEW.md`.

## Architecture map

- `src/spike_viz/events.py`: in-memory sparse spike event schema. `docs/schema.md` defines the sparse/dense layout.
- `src/spike_viz/io.py`: loads sparse/dense spike arrays from disk. `export.py`: `load_axon_export` for the axon-encoder export layout.
- `src/spike_viz/render.py`: CPU raster (`render_raster`); time is horizontal, neuron is vertical. `bloom.py`: neon bloom and tone-map for dense grids.
- `src/spike_viz/caption.py`: deterministic provenance captions from export metadata.
- `src/spike_viz/hero.py`, `gain_triptych.py`, `gain_captions.py`, `direction_ring.py`: provenance-pinned stills (Rate vs Poisson hero, gain triptych, preferred-direction ring).
- `rust/spike-viz-core/`: Rust-first core (workspace member; lib + `spike-viz` CLI) — export-contract load, CPU raster, PNG, `SemanticReport` parity bridge. Python stays the reference until the parity gates pass (see `docs/rust-migration.md`).
- `fixtures/axon-encoder/<encoder>/<case>/`: golden exports (`meta.json`, `spikes.npz`, optional `stimulus.npy`). Layout contract is in `docs/axon-encoder-export.md`.
- `tests/`: CPU pytest suite; `tests/conftest.py` skips `cuda`-marked tests when no GPU is present.
- `scripts/`: fixture generation and packing helpers (`pack_axon_export.py`, `make_tiny_synthetic_fixture.py`, `make_tiny_fractional_amp_fixture.py`, `axon_gain_export.patch`).
- `.agents/setup`: Amp orb bootstrap script, not a project build step.

## Invariants and non-goals (from `AGENTS.md`, `docs/CHARTER.md`, `README.md`)

- Preferred-direction rings need explicit `meta["preferred_angles_radians"]`; angles are never inferred. Issue #17 stays fixture-blocked (see README).
- Do not weaken tests to hide load errors; those failures are product bugs.

## Gotchas

- Source files in this repo have no SPDX headers; do not add them.
- `pyproject.toml` uses `setuptools` `packages.find` with no `exclude`/`files` list and there is no `MANIFEST.in`, so nothing needs updating when adding agent docs.
- `out/` and `.venv` are gitignored; the README quickstart writes `out/raster.png`.
- Never publish to PyPI (the charter rules out PyPI automation).
- Edit only what the task needs.
