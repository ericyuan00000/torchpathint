from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def dtype(request) -> torch.dtype:
    return request.param
