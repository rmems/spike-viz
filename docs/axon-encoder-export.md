# axon-encoder export contract

spike-viz side of the **export-first** bridge. Rust
[axon-encoder](https://github.com/Limen-Neural/axon-encoder) is encoding truth;
this document defines what files spike-viz will load. A cargo `example` or small
export binary may live in axon-encoder later — **this issue only defines the
contract on the spike-viz side**.

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

### axon-encoder field map

| axon-encoder `SpikeEvent` | export array |
|---------------------------|--------------|
| `timestamp: u64` | `t` |
| `channel: u16` | `neuron_id` |
| `polarity: bool` | `amp` ∈ `{0.0, 1.0}` when written |

## `meta.json` fields

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `schema_version` | string | **yes** | Contract version; current: `"1.0"` |
| `encoder` | string | **yes** | Encoder id, e.g. `"rate"`, `"poisson"` |
| `dt_seconds` | number | **yes** | Seconds per step (provenance) |
| `seed` | integer | **yes** | RNG seed used for the export |
| `n_neurons` | integer | **yes** | Geometry `N` |
| `n_steps` | integer | **yes** | Geometry `T` |
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

1. Prefer real exports from a pinned `axon_encoder_git_sha` when available.
2. Until then, **tiny synthetic** fixtures may be checked in if:
   - `synthetic: true` is set in `meta.json`
   - arrays match this schema
   - they are small and deterministic
3. spike-viz must never invent spikes when load fails.

## Non-goals (this contract)

- pyo3 / in-process Rust bindings
- Re-encoding inside Python
- Defining axon-encoder’s public Rust API (only the on-disk handoff)
