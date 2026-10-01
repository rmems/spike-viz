"""CUDA-only coverage: must skip cleanly on CPU hosts (no GPU failure)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="requires CUDA GPU (torch.cuda.is_available() is False)",
    ),
]


def test_cuda_tensor_roundtrip() -> None:
    x = torch.zeros((2, 2), device="cuda")
    assert x.device.type == "cuda"
    assert x.cpu().shape == (2, 2)
