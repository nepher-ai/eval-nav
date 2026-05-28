# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Spot waypoint navigation scorer — v2.

Designed for the Spot waypoint benchmark.  Adds locomotion quality on top of
success and time so that gait regularity and speed compliance are rewarded.

Formula
-------
    score = 0.50 × success_rate
          + 0.20 × time_efficiency
          + 0.30 × locomotion_quality

``locomotion_quality`` blends:
  - Gait regularity (aerial-phase fraction)   : 0.35  (when available)
  - Speed compliance (walking envelope)        : 0.25
  - Motion smoothness (speed variance)         : 0.20
  - Body stability (vertical/roll-pitch rate)  : 0.20

When locomotion telemetry is absent the scorer returns 0.0.

Used by
-------
- ``task_type: "navigation.spot"``, ``scoring_version: "v2"``
  → task-spot-waypointnav.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .....domain.metrics import AggregateMetrics, EpisodeMetrics
from ...base import BaseScorer


class SpotWaypointScorer(BaseScorer):
    """Spot waypoint benchmark scorer with locomotion quality (v2).

    Weights
    -------
    - Success rate       : 0.50
    - Time efficiency    : 0.20
    - Locomotion quality : 0.30
    """

    VERSION: str = "v2"

    W_SUCCESS: float = 0.50
    W_TIME: float = 0.20
    W_LOCOMOTION: float = 0.30

    # Spot walking speed envelope (m/s).
    MAX_WALKING_SPEED: float = 1.6
    MIN_USEFUL_SPEED: float = 0.05
    SIM_SPEED_TOLERANCE: float = 0.1

    MAX_SPEED_STD: float = 0.8
    MAX_VERTICAL_SPEED: float = 0.3
    MAX_ROLL_PITCH_RATE: float = 1.0
    MAX_AERIAL_PHASE: float = 0.05

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
        max_episode_time_s: float | None = None,  # noqa: ARG002
    ) -> float:
        """Compute success + time + locomotion score.

        Returns:
            Score in [0, 1], or 0.0 when locomotion telemetry is unavailable.
        """
        loco = self._locomotion_component(metrics)
        if loco is None:
            return 0.0
        return float(
            self.W_SUCCESS * metrics.success_rate
            + self.W_TIME * self._time_component(metrics, max_episode_steps, episodes)
            + self.W_LOCOMOTION * loco
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.spot",
            "scoring_version": self.VERSION,
            "weights": {
                "success_rate": self.W_SUCCESS,
                "time_efficiency": self.W_TIME,
                "locomotion_quality": self.W_LOCOMOTION,
            },
            "locomotion_thresholds": {
                "max_walking_speed": self.MAX_WALKING_SPEED,
                "min_useful_speed": self.MIN_USEFUL_SPEED,
                "sim_speed_tolerance": self.SIM_SPEED_TOLERANCE,
                "max_speed_std": self.MAX_SPEED_STD,
                "max_vertical_speed": self.MAX_VERTICAL_SPEED,
                "max_roll_pitch_rate": self.MAX_ROLL_PITCH_RATE,
                "max_aerial_phase": self.MAX_AERIAL_PHASE,
            },
        }

    # ------------------------------------------------------------------
    # Sub-components
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
                return 0.0 if norm > self.max_normalized_time else 1.0 - norm / self.max_normalized_time
        if metrics.mean_completion_time is not None:
            norm = metrics.mean_completion_time / max_episode_steps
            return 0.0 if norm > self.max_normalized_time else 1.0 - norm / self.max_normalized_time
        return 0.0

    def _locomotion_component(self, metrics: AggregateMetrics) -> float | None:
        """Compute aggregate locomotion quality in [0, 1]; None when absent."""
        ex = metrics.extra
        if not ex or "mean_speed" not in ex:
            return None
        return self._locomotion_from_extra(ex)

    def _locomotion_from_extra(self, ex: dict[str, Any]) -> float:
        """Blend locomotion sub-scores from an extra dict."""
        slope_factor = self._slope_speed_factor(ex.get("mean_slope_deg", 0.0))
        speed_score = self._speed_compliance(ex.get("mean_speed", 0.0), ex.get("max_speed", 0.0), slope_factor)
        smoothness_score = self._smoothness(ex.get("speed_std", 0.0))
        stability_score = self._body_stability(ex.get("mean_vertical_speed", 0.0), ex.get("mean_roll_pitch_rate", 0.0))
        gait_score = self._gait_quality(ex.get("aerial_phase_fraction"))
        if gait_score is not None:
            return float(0.35 * gait_score + 0.25 * speed_score + 0.20 * smoothness_score + 0.20 * stability_score)
        return float(0.40 * speed_score + 0.30 * smoothness_score + 0.30 * stability_score)

    # ------------------------------------------------------------------
    # Locomotion sub-scores (all [0, 1])
    # ------------------------------------------------------------------

    def _gait_quality(self, aerial_phase_fraction: float | None) -> float | None:
        if aerial_phase_fraction is None:
            return None
        if aerial_phase_fraction <= self.MAX_AERIAL_PHASE:
            return 1.0
        overshoot = (aerial_phase_fraction - self.MAX_AERIAL_PHASE) / (1.0 - self.MAX_AERIAL_PHASE)
        return max(0.0, 1.0 - overshoot)

    @staticmethod
    def _slope_speed_factor(mean_slope_deg: float) -> float:
        slope_rad = np.radians(np.clip(mean_slope_deg, 0.0, 60.0))
        return float(np.clip(np.cos(slope_rad), 0.3, 1.0))

    def _speed_compliance(self, mean_speed: float, max_speed: float, slope_factor: float = 1.0) -> float:
        effective_max = self.MAX_WALKING_SPEED * slope_factor
        if mean_speed < self.MIN_USEFUL_SPEED:
            return 0.0
        mean_ok = 1.0 if mean_speed <= effective_max else max(0.0, 1.0 - (mean_speed - effective_max) / effective_max * 2.0)
        peak_limit = effective_max + self.SIM_SPEED_TOLERANCE
        peak_ok = 1.0 if max_speed <= peak_limit else max(0.0, 1.0 - (max_speed - peak_limit) / effective_max * 5.0)
        return 0.7 * mean_ok + 0.3 * peak_ok

    def _smoothness(self, speed_std: float) -> float:
        return 0.0 if speed_std >= self.MAX_SPEED_STD else 1.0 - speed_std / self.MAX_SPEED_STD

    def _body_stability(self, mean_vert_speed: float, mean_rp_rate: float) -> float:
        vert = max(0.0, 1.0 - mean_vert_speed / self.MAX_VERTICAL_SPEED) if self.MAX_VERTICAL_SPEED > 0 else 1.0
        rp = max(0.0, 1.0 - mean_rp_rate / self.MAX_ROLL_PITCH_RATE) if self.MAX_ROLL_PITCH_RATE > 0 else 1.0
        return 0.5 * vert + 0.5 * rp
