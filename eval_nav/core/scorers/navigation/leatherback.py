# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Leatherback navigation scorer (v1).

Covers leatherback waypoint navigation and ANYmal B waypoint navigation —
tasks that share the same simple success + time evaluation logic and do not
expose locomotion telemetry.

Formula
-------
    score = 0.7 × success_rate + 0.3 × time_efficiency

``time_efficiency`` = 1 − (mean_successful_steps / max_episode_steps),
clipped to 0 when the ratio exceeds ``max_normalized_time``.

Used by
-------
- ``task_type: "navigation.leatherback"``, ``scoring_version: "v1"``
  → task-leatherback-waypointnav.yaml
  → task-animal-nav.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer


class LeatherbackNavScorer(BaseScorer):
    """Leatherback (and ANYmal B) waypoint navigation scorer.

    Success rate is the primary signal; faster completion is rewarded as a
    secondary signal.  No locomotion telemetry is required.

    Weights
    -------
    - Success rate   : 0.70
    - Time efficiency : 0.30
    """

    VERSION: str = "v1"
    W_SUCCESS: float = 0.70
    W_TIME: float = 0.30

    def __init__(self, max_normalized_time: float = 1.0) -> None:
        """
        Args:
            max_normalized_time: Episodes whose step-ratio exceeds this value
                receive a time score of 0.  Default 1.0 (full episode budget).
        """
        self.max_normalized_time = max_normalized_time

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,  # noqa: ARG002
    ) -> float:
        """Compute success + time score.

        Returns:
            Score in [0, 1].
        """
        time_component = self._time_component(metrics, max_episode_steps, episodes)
        return float(self.W_SUCCESS * metrics.success_rate + self.W_TIME * time_component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.leatherback",
            "scoring_version": self.VERSION,
            "weights": {
                "success_rate": self.W_SUCCESS,
                "time_efficiency": self.W_TIME,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _time_component(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None,
    ) -> float:
        if metrics.successful_episodes > 0 and episodes:
            successful_steps = [e.steps for e in episodes if e.success]
            if successful_steps:
                norm = float(np.mean(successful_steps)) / max_episode_steps
                if norm > self.max_normalized_time:
                    return 0.0
                return 1.0 - norm / self.max_normalized_time

        if metrics.mean_completion_time is not None:
            norm = metrics.mean_completion_time / max_episode_steps
            if norm > self.max_normalized_time:
                return 0.0
            return 1.0 - norm / self.max_normalized_time

        return 0.0
