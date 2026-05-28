# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Pick-and-place manipulation scorer — success rate + time efficiency.

Suitable for arm manipulation tasks (e.g. Franka high-level pick-and-place)
that do not expose locomotion telemetry.

Formula
-------
    score = 0.7 × success_rate + 0.3 × time_efficiency

``time_efficiency`` is 1 − (mean_successful_steps / max_episode_steps),
clipped to 0 when the ratio exceeds ``max_normalized_time``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer


class PickPlaceScorer(BaseScorer):
    """Manipulation pick-and-place scorer: task success + completion speed.

    Mirrors the logic of ``SimpleNavScorer`` but is named and documented
    for manipulation contexts, making the task-type intention explicit.

    Weights
    -------
    - Task success rate : 0.70
    - Time efficiency   : 0.30
    """

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
        """Compute task-success + time-efficiency score.

        Args:
            metrics: Aggregate metrics from all episodes.
            max_episode_steps: Step budget used for time normalization.
            episodes: Individual episode records (used for mean successful steps).
            max_episode_time_s: Accepted for API compatibility; not used here.

        Returns:
            Score in [0, 1].
        """
        success_component = metrics.success_rate
        time_component = self._time_component(metrics, max_episode_steps, episodes)
        return float(self.W_SUCCESS * success_component + self.W_TIME * time_component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.pick_place",
            "max_normalized_time": self.max_normalized_time,
            "weights": {
                "task_success_rate": self.W_SUCCESS,
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
