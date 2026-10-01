"""Shared pytest hooks: skip CUDA-marked tests when no GPU is present."""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip_cuda = pytest.mark.skipif(
        not _HAS_CUDA,
        reason="requires CUDA GPU (torch.cuda.is_available() is False)",
    )
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
