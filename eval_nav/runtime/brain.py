# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Brain runtime. The client lives in nepher-brain-comm, which brain-mode installs."""

from __future__ import annotations

import numpy as np


class BrainRuntime:
    """Adapt ``BrainClient`` to the lockstep runtime."""

    def __init__(self, client):
        self.client = client

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        self.client.begin_episode(episode_ids, seeds, instructions)

    def act(self, obs: dict[str, np.ndarray], step: int, seed: int) -> np.ndarray:
        return self.client.act(obs, step, seed)

    def end_episode(self, episode_ids: list[str]) -> None:
        self.client.end_episode(episode_ids)
