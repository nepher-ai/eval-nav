# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Leatherback maze navigation scorer (v2).

Evaluates a wheeled robot navigating a deterministic perfect-maze benchmark.
Unlike the plain leatherback v1 scorer, this version is success-rate-amplified
and additionally penalises exceeding a maximum speed or yaw-rate limit —
critical constraints for a physical Ackermann-drive robot operating in narrow
maze corridors (2 m width).

Formula
-------
    score = success_rate × (BASE + (1 − BASE) × mean_quality)

    BASE = 0.25   (ensures a non-zero score even for slow but correct runs)

    quality (per successful episode):
        = W_TIME × time_eff
        + W_SPEED × speed_compliance
        + W_YAW   × yaw_compliance

    time_eff:
        Uses physical time when max_episode_time_s is provided (decimation-
        invariant); falls back to steps/max_steps otherwise.

            norm = completion_time / max_episode_time_s   (or steps/max_steps)
            time_eff = max(0,  1 − norm)

    speed_compliance:
        1.0  if max_speed ≤ MAX_SPEED (3.0 m/s)
        else max(0,  1 − 2 × (max_speed − MAX_SPEED) / MAX_SPEED)

        Two-times penalty slope: exceeding the limit by 50% of MAX_SPEED
        drives the component to 0.  Chosen so a brief burst just above the
        limit costs less than consistently exceeding it.

    yaw_compliance:
        1.0  if max_yaw_rate ≤ MAX_YAW_RATE (2.0 rad/s)
        else max(0,  1 − 2 × (max_yaw_rate − MAX_YAW_RATE) / MAX_YAW_RATE)

        Rationale for 2.0 rad/s: at the 3 m/s speed cap the minimum turn
        radius is r = v/ω = 1.5 m, which keeps the robot (wheelbase 0.35 m)
        centred in a 2 m corridor.  Higher yaw-at-speed risks wall contact.
        It also sits safely below the Ackermann action's hard ceiling of
        3.0 rad/s so it acts as a comfort/safety limit, not a physical cap.

    If locomotion telemetry (max_speed / max_yaw_rate) is absent from
    episode.extra, the compliance components default to 1.0 (benefit of
    the doubt — no penalty for missing data).

Weights
-------
    W_TIME  = 0.50   fastest complete solutions score highest overall
    W_SPEED = 0.25   speed limit (3 m/s)
    W_YAW   = 0.25   yaw-rate limit (2.0 rad/s)

Used by
-------
    ``task_type: "navigation.leatherback"``, ``scoring_version: "v2"``
    → task-leatherback-lidar-maze.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer

# ---------------------------------------------------------------------------
# Limits (kept in sync with navigation_env_cfg_maze.py / ActionsCfg)
# ---------------------------------------------------------------------------

MAX_SPEED:    float = 3.0    # m/s — matches AckermannActionCfg.max_lin_vel
MAX_YAW_RATE: float = 2.0   # rad/s — tightest safe turn in 2 m corridor at 3 m/s


class LeatherbackMazeScorer(BaseScorer):
    """Leatherback maze navigation scorer (v2).

    Success-rate-amplified score that rewards fast, limit-respecting runs.

    Parameters
    ----------
    max_normalized_time : float
        Episodes whose normalised step fraction exceeds this value receive a
        time score of 0.  Default 1.0 (full episode budget).
    """

    VERSION:        str   = "v2"
    BASE:           float = 0.25
    W_TIME:         float = 0.50
    W_SPEED:        float = 0.25
    W_YAW:          float = 0.25

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
        """Compute the maze navigation score ∈ [0, 1].

        Returns 0.0 immediately if no episodes succeeded.
        """
        if not episodes or metrics.successful_episodes == 0:
            return 0.0

        success_rate  = metrics.success_rate
        mean_quality  = self._mean_quality(
            episodes, max_episode_steps, max_episode_time_s=max_episode_time_s
        )

        return float(success_rate * (self.BASE + (1.0 - self.BASE) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type":       "navigation.leatherback",
            "scoring_version": self.VERSION,
            "weights": {
                "time_efficiency":  self.W_TIME,
                "speed_compliance": self.W_SPEED,
                "yaw_compliance":   self.W_YAW,
            },
            "limits": {
                "max_speed_m_s":    MAX_SPEED,
                "max_yaw_rate_rad": MAX_YAW_RATE,
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
        time_eff = self._time_efficiency(ep, max_episode_steps, max_episode_time_s)
        speed_ok = self._speed_compliance(ep.extra)
        yaw_ok   = self._yaw_compliance(ep.extra)

        return float(
            self.W_TIME  * time_eff
            + self.W_SPEED * speed_ok
            + self.W_YAW   * yaw_ok
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
        # Prefer physical time (decimation-invariant) over step count
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
            return 1.0  # no telemetry → benefit of the doubt
        if max_spd <= MAX_SPEED:
            return 1.0
        # Zero at max_spd = MAX_SPEED * 1.5
        excess = (max_spd - MAX_SPEED) / MAX_SPEED
        return float(max(0.0, 1.0 - 2.0 * excess))

    @staticmethod
    def _yaw_compliance(extra: dict[str, Any]) -> float:
        """1.0 if max_yaw_rate ≤ MAX_YAW_RATE, else linearly decreasing penalty."""
        max_yaw = extra.get("max_yaw_rate") if extra else None
        if max_yaw is None:
            return 1.0  # no telemetry → benefit of the doubt
        if max_yaw <= MAX_YAW_RATE:
            return 1.0
        excess = (max_yaw - MAX_YAW_RATE) / MAX_YAW_RATE
        return float(max(0.0, 1.0 - 2.0 * excess))
