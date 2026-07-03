# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Humanoid waypoint race scorer (v1).

Evaluates a G1 humanoid navigating a deterministic waypoint benchmark.
The scorer is success-rate-amplified and ranks successful runs by completion
time only.

Formula
-------
    score = success_rate × (BASE + (1 − BASE) × mean_quality)

    BASE = 0.25   (non-zero floor even for slow but correct runs)

    quality (per successful episode):
        = time_eff

    time_eff:
        Uses physical time when max_episode_time_s is provided
        (decimation-invariant); falls back to steps/max_steps otherwise.

            norm = completion_time / max_episode_time_s
            time_eff = max(0,  1 − norm)

Used by
-------
    ``task_type: "navigation.humanoid"``, ``scoring_version: "v1"``
    → eval-nav/configs/task-humanoid-race.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer


class HumanoidRaceScorer(BaseScorer):
    """Humanoid waypoint race scorer (v1).

    Success-rate-amplified score that rewards fast completion times.

    Parameters
    ----------
    max_normalized_time : float
        Episodes whose normalised time fraction exceeds this value receive a
        time score of 0.  Default 1.0 (full episode budget).
    """

    VERSION: str = "v1"
    BASE: float = 0.25

    def __init__(self, max_normalized_time: float = 1.0) -> None:
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
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute the humanoid race score ∈ [0, 1].

        Returns 0.0 immediately if no episodes succeeded.
        """
        if not episodes or metrics.successful_episodes == 0:
            return 0.0

        success_rate = metrics.success_rate
        mean_quality = self._mean_quality(
            episodes, max_episode_steps, max_episode_time_s=max_episode_time_s
        )

        return float(success_rate * (self.BASE + (1.0 - self.BASE) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.humanoid",
            "scoring_version": self.VERSION,
            "weights": {
                "time_efficiency": 1.0,
            },
        }

    # ------------------------------------------------------------------
    # Per-episode quality
    # ------------------------------------------------------------------

    def _mean_quality(
        self,
        episodes: list[EpisodeMetrics],
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
        """Average quality over successful episodes only."""
        qualities = [
            self._episode_quality(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            for ep in episodes
            if ep.success
        ]
        return float(np.mean(qualities)) if qualities else 0.0

    def _episode_quality(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
        """Quality score ∈ [0, 1] for a single successful episode."""
        return self._time_efficiency(ep, max_episode_steps, max_episode_time_s)

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    def _time_efficiency(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float:
        """Normalised time score ∈ [0, 1]; faster = higher."""
        if max_episode_time_s and max_episode_time_s > 0 and ep.completion_time is not None:
            norm = ep.completion_time / max_episode_time_s
        elif max_episode_steps > 0:
            norm = ep.steps / max_episode_steps
        else:
            return 0.0

        if norm > self.max_normalized_time:
            return 0.0
        return float(max(0.0, 1.0 - norm / self.max_normalized_time))
