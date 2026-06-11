# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Pick-and-place manipulation scorers — v1 and v2.

Suitable for arm manipulation tasks (e.g. Franka high-level pick-and-place)
that do not expose locomotion telemetry.

v1 (deprecated)
---------------
    score = 0.7 × success_rate + 0.3 × time_efficiency

Flaw: time efficiency can compensate for a lower success rate, allowing a
submission with fewer successes to outscore one with more.

v2
--
    score = success_rate × (W_SUCCESS_BONUS + (1 − W_SUCCESS_BONUS) × time_efficiency)

``success_rate`` is a first-class **multiplier**, mirroring the v4 pattern
from navigation.spot.  A submission can only improve its score via time
efficiency up to the ceiling set by its own success rate — higher success rate
always yields a higher score ceiling.

``W_SUCCESS_BONUS`` (0.75) means 75% of the score is locked to success rate
alone; only the remaining 25% is modulated by time efficiency.

``time_efficiency`` is 1 − (mean_successful_steps / max_episode_steps),
clipped to 0 when the ratio exceeds ``max_normalized_time``.

Used by
-------
- ``task_type: "manipulation.pick_place"``, ``scoring_version: "v1"`` — legacy
- ``task_type: "manipulation.pick_place"``, ``scoring_version: "v2"``
  → task-franka-mani-hl.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer


class PickPlaceScorer(BaseScorer):
    """Manipulation pick-and-place scorer: task success + completion speed (v1).

    Weights
    -------
    - Task success rate : 0.70
    - Time efficiency   : 0.30
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
        """Compute task-success + time-efficiency score.

        Returns:
            Score in [0, 1].
        """
        time_component = self._time_component(metrics, max_episode_steps, episodes)
        return float(self.W_SUCCESS * metrics.success_rate + self.W_TIME * time_component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.pick_place",
            "scoring_version": self.VERSION,
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


class PickPlaceScorerV2(PickPlaceScorer):
    """Manipulation pick-and-place scorer: success-rate-amplified + time efficiency (v2).

    Adopts the v4 navigation pattern: ``success_rate`` is a first-class
    multiplier rather than an additive term, so a higher success rate always
    produces a higher score regardless of time efficiency.

    Formula
    -------
    ``score = success_rate × (W_SUCCESS_BONUS + (1 − W_SUCCESS_BONUS) × time_efficiency)``

    Weights
    -------
    - Success bonus (floor credit) : 0.75
    - Time efficiency (ceiling lift): 0.25
    """

    VERSION: str = "v2"
    W_SUCCESS_BONUS: float = 0.75

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,  # noqa: ARG002
    ) -> float:
        """Compute success-rate-amplified score.

        Returns:
            Score in [0, 1].
        """
        time_component = self._time_component(metrics, max_episode_steps, episodes)
        quality = self.W_SUCCESS_BONUS + (1.0 - self.W_SUCCESS_BONUS) * time_component
        return float(metrics.success_rate * quality)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.pick_place",
            "scoring_version": self.VERSION,
            "weights": {
                "success_bonus": self.W_SUCCESS_BONUS,
                "time_efficiency": 1.0 - self.W_SUCCESS_BONUS,
                "formula": "success_rate × (W_SUCCESS_BONUS + (1 − W_SUCCESS_BONUS) × time_efficiency)",
            },
        }
