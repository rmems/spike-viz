"""spike-viz: paint spike activity (load exports → rasters / PNG)."""

from __future__ import annotations

__version__ = "0.1.0a0"

from spike_viz.events import SpikeEvents
from spike_viz.export import AxonExportCase, load_axon_export
from spike_viz.io import SpikeIOError, load_dense, load_sparse, sparse_to_dense
from spike_viz.render import render_raster

__all__ = [
    "AxonExportCase",
    "SpikeEvents",
    "SpikeIOError",
    "load_axon_export",
    "load_dense",
    "load_sparse",
    "render_raster",
    "sparse_to_dense",
    "__version__",
]
