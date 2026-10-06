# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Seed the simulator process before each shard."""

from __future__ import annotations

import os
import random

import numpy as np


def apply_determinism(seed: int) -> None:
    """Seed Python, numpy, and torch, and request deterministic kernels."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
