from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")
