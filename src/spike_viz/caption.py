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

    If ``out_path`` is supplied, expand ``~`` and resolve ``..`` to one
    absolute path before metadata is read. The parent directory must already
    exist; otherwise :class:`OSError` is raised and nothing is written. The
    sidecar is UTF-8 plain text with one trailing newline. Return the caption
    without a trailing newline. When the parent exists, invalid metadata
    raises :class:`SpikeIOError` before any sidecar is written. Other write
    failures raise :class:`OSError`.
    """
    destination = _resolved_sidecar(out_path) if out_path is not None else None
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
    if destination is not None:
        destination.write_text(caption + "\n", encoding="utf-8")
    return caption


def _resolved_sidecar(out_path: PathLike) -> Path:
    """Canonicalize a sidecar path and require its parent directory.

    ``expanduser`` then ``resolve`` collapses ``~`` and ``..`` before the
    write, so the target is that absolute path rather than a relative
    traversal left for the OS to interpret.
    """
    resolved = Path(out_path).expanduser().resolve()
    if not resolved.parent.is_dir():
        raise OSError(f"Parent directory does not exist: {resolved.parent}")
    return resolved
