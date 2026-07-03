# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Humanoid waypoint race scorer (v1).

Evaluates a G1 humanoid navigating a deterministic waypoint benchmark.
The scorer is success-rate-amplified and additionally rewards stable,
efficient locomotion — critical constraints for a biped that can fall.

Formula
-------
    score = success_rate × (BASE + (1 − BASE) × mean_quality)

    BASE = 0.25   (non-zero floor even for slow but correct runs)

    quality (per successful episode):
        = W_TIME      × time_eff
        + W_SPEED     × speed_compliance
        + W_STABILITY × stability

    time_eff:
        Uses physical time when max_episode_time_s is provided
        (decimation-invariant); falls back to steps/max_steps otherwise.

            norm = completion_time / max_episode_time_s
            time_eff = max(0,  1 − norm)

    speed_compliance:
        Reward for staying at or below MAX_SPEED (2.0 m/s — G1 race vx_range
        upper limit).  Two-times slope so that exceeding the limit by 50%
        drives the component to zero.

            1.0  if max_speed ≤ MAX_SPEED
            else max(0,  1 − 2 × (max_speed − MAX_SPEED) / MAX_SPEED)

    stability:
        Reward for low roll/pitch angular velocity (proxy for bipedal
        stability).  Falls to zero when max_roll_pitch_rate ≥ MAX_ROLL_PITCH_RATE
        (2.0 rad/s — vigorous wobble / near-fall boundary).

            1.0  if max_roll_pitch_rate ≤ MAX_ROLL_PITCH_RATE
            else max(0,  1 − 2 × (max_roll_pitch_rate − MAX_ROLL_PITCH_RATE) / MAX_ROLL_PITCH_RATE)

    If locomotion telemetry is absent from episode.extra the corresponding
    component defaults to 1.0 (benefit of the doubt).

Weights
-------
    W_TIME      = 0.50   fastest complete solutions score highest
    W_SPEED     = 0.25   speed limit (2.0 m/s)
    W_STABILITY = 0.25   roll/pitch rate limit (2.0 rad/s)

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

# ---------------------------------------------------------------------------
# Limits (tuned for G1 humanoid race task)
# ---------------------------------------------------------------------------

MAX_SPEED:          float = 2.0   # m/s  — G1 race vx_range upper bound
MAX_ROLL_PITCH_RATE: float = 2.0  # rad/s — vigorous wobble / near-fall boundary


class HumanoidRaceScorer(BaseScorer):
    """Humanoid waypoint race scorer (v1).

    Success-rate-amplified score that rewards fast, stable, within-limit runs.

    Parameters
    ----------
    max_normalized_time : float
        Episodes whose normalised time fraction exceeds this value receive a
        time score of 0.  Default 1.0 (full episode budget).
    """

    VERSION:      str   = "v1"
    BASE:         float = 0.25
    W_TIME:       float = 0.50
    W_SPEED:      float = 0.25
    W_STABILITY:  float = 0.25

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
            "task_type":       "navigation.humanoid",
            "scoring_version": self.VERSION,
            "weights": {
                "time_efficiency":  self.W_TIME,
                "speed_compliance": self.W_SPEED,
                "stability":        self.W_STABILITY,
            },
            "limits": {
                "max_speed_m_s":           MAX_SPEED,
                "max_roll_pitch_rate_rad": MAX_ROLL_PITCH_RATE,
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
        time_eff   = self._time_efficiency(ep, max_episode_steps, max_episode_time_s)
        speed_ok   = self._speed_compliance(ep.extra)
        stable     = self._stability(ep.extra)

        return float(
            self.W_TIME      * time_eff
            + self.W_SPEED     * speed_ok
            + self.W_STABILITY * stable
        )

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

    @staticmethod
    def _speed_compliance(extra: dict[str, Any]) -> float:
        """1.0 if max_speed ≤ MAX_SPEED, else linearly decreasing penalty."""
        max_spd = extra.get("max_speed") if extra else None
        if max_spd is None:
            return 1.0
        if max_spd <= MAX_SPEED:
            return 1.0
        excess = (max_spd - MAX_SPEED) / MAX_SPEED
        return float(max(0.0, 1.0 - 2.0 * excess))

    @staticmethod
    def _stability(extra: dict[str, Any]) -> float:
        """1.0 if max_roll_pitch_rate ≤ MAX_ROLL_PITCH_RATE, else penalty."""
        max_rp = extra.get("max_roll_pitch_rate") if extra else None
        if max_rp is None:
            return 1.0
        if max_rp <= MAX_ROLL_PITCH_RATE:
            return 1.0
        excess = (max_rp - MAX_ROLL_PITCH_RATE) / MAX_ROLL_PITCH_RATE
        return float(max(0.0, 1.0 - 2.0 * excess))
