# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Humanoid RunJump course scorer (v1).

Evaluates a G1 high-level policy on the EnvHub obstacle-course benchmark.
Success-rate-amplified score that blends completion time with AMP human-style
similarity (organizer ``run_discriminator.pt`` / ``jump_discriminator.pt``).

Formula
-------
    score = success_rate × (BASE + (1 − BASE) × mean_quality)

    BASE = 0.25

    quality (per successful episode):
        = 0.5 × time_eff + 0.5 × style

    time_eff:
        Uses physical time when max_episode_time_s is provided; else steps.

            time_eff = max(0, 1 − completion_time / max_episode_time_s)

    style:
        Episode mean of σ(D_mode(amp_obs)) from ``extra["mean_style"]``.
        Missing style → 0.0 for that episode.

Used by
-------
    ``task_type: "navigation.humanoid.runjump"``, ``scoring_version: "v1"``
    → eval-nav/configs/task-humanoid-run-jump.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer


class HumanoidRunJumpScorer(BaseScorer):
    """Humanoid RunJump course scorer (v1): SR × (time + AMP style)."""

    VERSION: str = "v1"
    BASE: float = 0.25
    W_TIME: float = 0.5
    W_STYLE: float = 0.5

    def __init__(self, max_normalized_time: float = 1.0) -> None:
        self.max_normalized_time = max_normalized_time

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute the RunJump score ∈ [0, 1]."""
        if not episodes or metrics.successful_episodes == 0:
            return 0.0

        success_rate = metrics.success_rate
        mean_quality = self._mean_quality(
            episodes, max_episode_steps, max_episode_time_s=max_episode_time_s
        )
        return float(success_rate * (self.BASE + (1.0 - self.BASE) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.humanoid.runjump",
            "scoring_version": self.VERSION,
            "weights": {
                "time_efficiency": self.W_TIME,
                "style": self.W_STYLE,
            },
            "base": self.BASE,
        }

    def _mean_quality(
        self,
        episodes: list[EpisodeMetrics],
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
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
        time_eff = self._time_efficiency(ep, max_episode_steps, max_episode_time_s)
        style = self._style(ep)
        return float(self.W_TIME * time_eff + self.W_STYLE * style)

    def _time_efficiency(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float:
        if max_episode_time_s and max_episode_time_s > 0 and ep.completion_time is not None:
            norm = ep.completion_time / max_episode_time_s
        elif max_episode_steps > 0:
            norm = ep.steps / max_episode_steps
        else:
            return 0.0

        if norm > self.max_normalized_time:
            return 0.0
        return float(max(0.0, 1.0 - norm / self.max_normalized_time))

    @staticmethod
    def _style(ep: EpisodeMetrics) -> float:
        raw = ep.extra.get("mean_style") if ep.extra else None
        if raw is None:
            return 0.0
        return float(max(0.0, min(1.0, float(raw))))
