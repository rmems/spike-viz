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
    Synthetic exports additionally carry ``synthetic=true``. A present
    ``synthetic`` value that is not a boolean raises :class:`SpikeIOError`.
    String values are JSON-quoted with non-ASCII escaped, so newlines,
    U+2028, U+2029, and lone surrogates stay on one UTF-8 line.

    ``dt_seconds`` retains the export's seconds-per-displayed-step convention;
    it is not necessarily an encoder-owned physical timebase. No provenance
    is inferred from local Git state or the environment.

    If ``out_path`` is supplied, expand ``~`` and resolve ``..`` to one
    absolute path before metadata is read. The parent directory must already
    exist; otherwise :class:`OSError` is raised and nothing is written. The
    sidecar is UTF-8 plain text with one trailing newline. Return the caption
    without a trailing newline. When the parent exists, invalid metadata
    raises :class:`SpikeIOError` before any sidecar is written. If
    ``out_path`` resolves to the metadata file, including through a symlink
    or hard link, :class:`SpikeIOError` is raised and that file is left
    unchanged. Other write failures raise :class:`OSError`.
    """
    destination = _resolved_sidecar(out_path) if out_path is not None else None
    _reject_metadata_overwrite(meta_path, destination)
    meta = load_meta(meta_path)
    sha = meta.get("axon_encoder_git_sha")
    if sha is not None and not isinstance(sha, str):
        raise SpikeIOError(
            f"{meta_path}: 'axon_encoder_git_sha' must be a string or null"
        )
    commit = _quoted(sha) if sha and sha.strip() else "unknown"
    caption = (
        f"encoder={_quoted(meta['encoder'])} | "
        f"dt_seconds={meta['dt_seconds']} | seed={meta['seed']} | "
        f"N={meta['n_neurons']} | T={meta['n_steps']} | "
        f"axon_encoder_git_sha={commit}"
    )
    if "synthetic" in meta:
        synthetic = meta["synthetic"]
        if not isinstance(synthetic, bool):
            raise SpikeIOError(f"{meta_path}: 'synthetic' must be a boolean")
        if synthetic:
            caption += " | synthetic=true"
    if destination is not None:
        destination.write_text(caption + "\n", encoding="utf-8")
    return caption


def _quoted(value: str) -> str:
    """JSON-quote a caption field, escaping non-ASCII and line separators."""
    return json.dumps(value)


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


def _reject_metadata_overwrite(meta_path: PathLike, destination: Path | None) -> None:
    """Refuse a sidecar that would replace the metadata file."""
    if destination is None:
        return
    source = Path(meta_path).expanduser().resolve()
    same = destination == source
    if not same:
        try:
            same = destination.samefile(source)
        except OSError:
            same = False
    if same:
        raise SpikeIOError(
            f"{destination}: out_path resolves to the metadata file; "
            f"refusing to overwrite {source}"
        )
