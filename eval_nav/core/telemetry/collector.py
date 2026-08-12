# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Batched raw-state collector for envs that expose ``get_raw_state``."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class RawTelemetryCollector:
    """Collect per-step raw state from ``env.get_raw_state()`` into per-env buffers.

    One host sync per step (the batched ``get_raw_state`` call). Done envs are
    skipped when appending so buffers match live episode length.
    """

    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self._buffers: dict[int, dict[str, list[np.ndarray]]] = {
            i: defaultdict(list) for i in range(self.num_envs)
        }
        self._metadata: dict[int, dict[str, Any] | None] = {i: None for i in range(self.num_envs)}

    def set_metadata(self, env_idx: int, metadata: dict[str, Any] | None) -> None:
        self._metadata[int(env_idx)] = metadata

    def get_metadata(self, env_idx: int) -> dict[str, Any] | None:
        return self._metadata.get(int(env_idx))

    def collect_step(self, env: Any, done_per_env: list[bool] | None = None) -> None:
        """Append one raw sample for every live env."""
        raw = env.get_raw_state()
        if not raw:
            return
        n = self.num_envs
        for env_idx in range(n):
            if done_per_env is not None and done_per_env[env_idx]:
                continue
            buf = self._buffers[env_idx]
            for key, arr in raw.items():
                sample = np.asarray(arr[env_idx])
                buf[key].append(sample.copy() if sample.ndim > 0 else sample)

    def series(self, env_idx: int) -> dict[str, np.ndarray]:
        """Stack one env's per-step lists into arrays (leading dim = T)."""
        buf = self._buffers.get(int(env_idx), {})
        out: dict[str, np.ndarray] = {}
        for key, samples in buf.items():
            if not samples:
                continue
            try:
                out[key] = np.stack(samples, axis=0)
            except ValueError:
                out[key] = np.asarray(samples)
        return out

    def reset_env(self, env_idx: int) -> None:
        self._buffers[int(env_idx)] = defaultdict(list)
        self._metadata[int(env_idx)] = None
