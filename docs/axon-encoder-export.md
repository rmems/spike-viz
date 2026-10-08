# axon-encoder export contract

spike-viz side of the **export-first** bridge. Rust
[axon-encoder](https://github.com/Limen-Neural/axon-encoder) is encoding truth;
this document defines what files spike-viz will load.

Generation path: axon-encoder ships `examples/export_spike_viz.rs`, which
writes a staging layout (`t.npy` / `neuron_id.npy` / `amp.npy` /
`stimulus.npy` / `meta.json`); `scripts/pack_axon_export.py` in this repo
assembles that staging into `spikes.npz` and validates the result through
`load_axon_export`. `scripts/generate_fixtures.sh` regenerates every
checked-in case from a pinned axon-encoder checkout.

Schema details: [schema.md](schema.md). Charter: [CHARTER.md](CHARTER.md).

## Layout

```text
fixtures/axon-encoder/<encoder>/<case>/
  spikes.npz      # required
  meta.json       # required
  stimulus.npy    # optional continuous input for underlays
```

Example checked-in golden (synthetic, schema-valid):

```text
fixtures/axon-encoder/rate/tiny_synthetic/
```

| File | Content |
|------|---------|
| `spikes.npz` | Sparse arrays `t`, `neuron_id`, optional `amp` |
| `meta.json` | Provenance + geometry (see below) |
| `stimulus.npy` | Optional float array of continuous stimulus |

## `spikes.npz` field types

| Key | NumPy dtype | Required | Notes |
|-----|-------------|----------|-------|
| `t` | `int64` | yes | Step index in `[0, n_steps)` |
| `neuron_id` | `int64` | yes | Channel/neuron in `[0, n_neurons)` |
| `amp` | `float32` | no | Default 1.0 if omitted |

`allow_pickle` must not be required. spike-viz loads with `allow_pickle=False`.

### axon-encoder field map (0.5.x)

| axon-encoder `SpikeEvent` | export array |
|---------------------------|--------------|
| `timestamp: TickOffset` → `TimeCursor::absolute(...)` | `t` |
| `channel: u16` | `neuron_id` |
| `polarity: bool` | `amp` ∈ `{1.0, -1.0}` when written |

#### `TickOffset` → `t` mapping

Since axon-encoder 0.5, `SpikeEvent::timestamp` is a `TickOffset`: ticks
**relative to the start of the call that emitted the spike**, never absolute.
Exporters must normalize it through a [`TimeCursor`] advanced once per
`encode`/`encode_step` call:

```rust
let mut cursor = TimeCursor::new(encoder.time_model());
for step in &stimulus {
    let out = encoder.encode_step(step);
    for spike in &out.spikes {
        let t = cursor.absolute(spike.timestamp); // absolute encoder tick
    }
    cursor.advance();
}
```

Exporting `offset.ticks()` directly collapses every streaming spike onto
the call-local range (`0..span_ticks`) — usually `t = 0` for instant encoders.
`tests/` cover this regression.

The encoder's `TimeModel` defines the span:
- `TimeModel::INSTANT` (rate, population, temporal, predictive, …): one tick
  per call; `n_steps` = number of calls.
- `TimeModel::window(span)` (latency): each call is one presentation;
  `n_steps` = calls × span.
- `TimeModel::overlapping` (phase): `n_steps` must include trailing
  `span − step` slack for the last call's offsets.

#### `dt_seconds` semantics

For encoders configured in physical units (e.g. `RateEncoder::try_new(..,
dt_seconds)`), `dt_seconds` is the real tick duration reported by the
encoder's `Timebase`. For dimensionless encoders (latency, population,
temporal, predictive, phase) the encoder owns no physical time: `dt_seconds`
in `meta.json` is then the **export/render sampling convention** chosen by
the exporter — consumers should treat it as "seconds per displayed step",
not an encoder-owned physical claim.

## `meta.json` fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `schema_version` | string | **yes** | Contract version; **supported: `"1.0"` only** (unknown versions fail load) |
| `encoder` | non-empty string | **yes** | Encoder id, e.g. `"rate"`, `"poisson"` |
| `dt_seconds` | number | **yes** | Seconds per step; must be finite and **> 0** |
| `seed` | integer | **yes** | RNG seed used for the export (`>= 0`) |
| `n_neurons` | integer | **yes** | Geometry `N`; must be **>= 1** even if spikes empty |
| `n_steps` | integer | **yes** | Geometry `T`; must be **>= 1** even if spikes empty |
| `axon_encoder_git_sha` | string \| null | no | Commit of axon-encoder when known |
| `synthetic` | boolean | no | `true` if not produced by real axon-encoder |
| `stimulus_notes` | string | no | Human notes about the stimulus |
| `notes` | string | no | Free-form case notes |

Loaders **require** the yes-column keys and reject unknown geometry mismatches
(e.g. `t >= n_steps`).

## Loader API

```python
from spike_viz import load_axon_export

case = load_axon_export("fixtures/axon-encoder/rate/tiny_synthetic")
case.events   # SpikeEvents
case.meta     # dict
case.stimulus_path  # Path | None
```

Missing `spikes.npz` / `meta.json` → `SpikeIOError` (not empty spikes).

## Golden fixtures policy

1. Real exports from a pinned `axon_encoder_git_sha` live under
   `fixtures/axon-encoder/<encoder>/shared_sine_v1/`, all generated from one
   shared stimulus by `scripts/generate_fixtures.sh` (`synthetic: false`).
   Stochastic encoders use seeded `*_with_rng` surfaces, so re-running against
   the recorded SHA reproduces them bit-for-bit.
2. **Tiny synthetic** fixtures may be checked in if:
   - `synthetic: true` is set in `meta.json`
   - arrays match this schema
   - they are small and deterministic
3. spike-viz must never invent spikes when load fails.

## Non-goals (this contract)

- pyo3 / in-process Rust bindings
- Re-encoding inside Python
- Defining axon-encoder’s public Rust API (only the on-disk handoff)
