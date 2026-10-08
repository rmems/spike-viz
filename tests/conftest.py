"""Shared pytest hooks: skip CUDA-marked tests when no GPU is present."""

from __future__ import annotations

import pytest

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # torch missing/broken: treat CUDA as unavailable, not a CI failure
    _HAS_CUDA = False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _HAS_CUDA:
        return
    skip_cuda = pytest.mark.skip(
        reason="requires CUDA GPU (torch.cuda.is_available() is False)"
    )
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
