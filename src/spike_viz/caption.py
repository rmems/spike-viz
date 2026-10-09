"""Deterministic human-readable provenance from export metadata."""

from __future__ import annotations

import json
from pathlib import Path

from spike_viz.export import PathLike, load_meta
from spike_viz.io import SpikeIOError


def provenance_caption(meta_path: PathLike, *, out_path: PathLike | None = None) -> str:
    """Build a caption from a schema-valid ``meta.json``, optionally saving it.

    Fields appear in a fixed order: encoder, dt_seconds, seed, N, T, and
    axon_encoder_git_sha (full value, or ``unknown`` if absent/null/blank).
    Synthetic exports additionally carry ``synthetic=true``. String values
    are JSON-quoted so embedded newlines cannot split the caption into lines.

    ``dt_seconds`` retains the export's seconds-per-displayed-step convention;
    it is not necessarily an encoder-owned physical timebase. No provenance
    is inferred from local Git state or the environment.

    If ``out_path`` is supplied, write UTF-8 plain text with one trailing
    newline (suitable for a ``.txt`` sidecar). Parent directories must exist.
    Return the caption without a trailing newline. Invalid metadata raises
    :class:`SpikeIOError` before any sidecar is written; write failures raise
    :class:`OSError`.
    """
    meta = load_meta(meta_path)
    sha = meta.get("axon_encoder_git_sha")
    if sha is not None and not isinstance(sha, str):
        raise SpikeIOError(
            f"{meta_path}: 'axon_encoder_git_sha' must be a string or null"
        )
    commit = json.dumps(sha, ensure_ascii=False) if sha and sha.strip() else "unknown"
    caption = (
        f"encoder={json.dumps(meta['encoder'], ensure_ascii=False)} | "
        f"dt_seconds={meta['dt_seconds']} | seed={meta['seed']} | "
        f"N={meta['n_neurons']} | T={meta['n_steps']} | "
        f"axon_encoder_git_sha={commit}"
    )
    if meta.get("synthetic") is True:
        caption += " | synthetic=true"
    if out_path is not None:
        Path(out_path).expanduser().write_text(caption + "\n", encoding="utf-8")
    return caption
