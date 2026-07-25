# Spike event schema

In-memory and on-disk shapes used by **spike-viz**. Related: export layout in
[axon-encoder-export.md](axon-encoder-export.md), product boundary in
[CHARTER.md](CHARTER.md).

## Sparse (COO-like)

Parallel arrays of equal length `E` (number of events):

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `t` | `int64` | yes | Encoder **step** index (default), not seconds |
| `neuron_id` | `int64` | yes | Neuron / channel index |
| `amp` | `float32` | no | Amplitude; default **1.0** when absent |

### Mapping from axon-encoder `SpikeEvent`

| axon-encoder | spike-viz |
|--------------|-----------|
| `timestamp` (`u64`) | `t` |
| `channel` (`u16`) | `neuron_id` |
| `polarity` (`bool`) | `amp` as `1.0` / `0.0` when exported |

## Dense

| Shape | Type | Meaning |
|-------|------|---------|
| `[T, N]` | `float32` or `bool` | Time × neuron grid |

Helpers: `sparse_to_dense(events, n_steps, n_neurons)` → `[T, N]` float32.

## On-disk sparse

**`.npz`** with arrays:

- required: `t`, `neuron_id`
- optional: `amp`

No pickle. Missing files or keys raise clear errors (`SpikeIOError`) — **never**
return empty success that looks like “no spikes” when the file is absent.

## On-disk dense

- `.npy` — raw `[T, N]` array
- `.npz` — array named `dense`, or a single unnamed array

## API

```python
from spike_viz import SpikeEvents, load_sparse, load_dense, sparse_to_dense

events = load_sparse("spikes.npz")
grid = sparse_to_dense(events, n_steps=64, n_neurons=32)
dense = load_dense("grid.npy")
```
