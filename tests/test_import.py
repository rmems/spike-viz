"""Import smoke test: the installed `spike_viz` package must import on CPU."""

from __future__ import annotations


def test_import_spike_viz() -> None:
    import spike_viz

    assert isinstance(spike_viz.__version__, str) and spike_viz.__version__
    for name in (
        "AxonExportCase",
        "SpikeEvents",
        "SpikeIOError",
        "load_axon_export",
        "load_dense",
        "load_sparse",
        "render_raster",
        "sparse_to_dense",
    ):
        assert hasattr(spike_viz, name), f"spike_viz missing export: {name}"
