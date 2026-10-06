# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""In-process runtime for a checkpoint loaded by the existing policy loader."""

from __future__ import annotations

from typing import Any

import numpy as np


class InProcessRuntime:
    """Call a policy that maps an observation to one action."""

    def __init__(self, policy: Any):
        self.policy = policy

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        return None

    def act(self, obs: dict[str, np.ndarray], step: int, seed: int) -> np.ndarray:
        action = np.asarray(self.policy(obs), dtype=np.float32)
        if action.ndim == 2:
            action = action[:, None, :]
        return action

    def end_episode(self, episode_ids: list[str]) -> None:
        return None
