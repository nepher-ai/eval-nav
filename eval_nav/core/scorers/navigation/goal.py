# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Goal navigation scorer — success-rate-amplified quality with directness.

Designed for obstacle-terrain goal navigation (e.g. Spot student policy)
where success rate, locomotion stability, path directness, and time
efficiency all matter.

Formula
-------
Per successful episode:
    quality = W_TIME × time_efficiency
            + W_STABILITY × stability_quality
            + W_DIRECTNESS × directness_quality

Final score:
    score = success_rate × (W_SUCCESS_BONUS + (1 − W_SUCCESS_BONUS) × mean_quality)

Failed episodes score 0 and do not contribute to ``mean_quality``.
``W_SUCCESS_BONUS`` gives a guaranteed floor credit for each success so
that success rate has first-class influence over the final score.

When locomotion telemetry is absent the scorer returns 0.0.  When
``max_episode_time_s`` is provided, time efficiency is computed from
physical seconds (decimation-invariant).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from .waypoint import WaypointNavScorer


class GoalNavScorer(WaypointNavScorer):
    """Obstacle-terrain goal navigation scorer.

    Builds on ``WaypointNavScorer``'s locomotion helpers and restructures
    aggregation so success rate is an explicit multiplier.

    Episode weights
    ---------------
    - Time efficiency      : 0.40
    - Stability quality    : 0.40
    - Path directness      : 0.20

    Aggregation
    -----------
    ``score = success_rate × (0.25 + 0.75 × mean_quality_over_successes)``
    """

    W_EPISODE_TIME: float = 0.40
    W_EPISODE_STABILITY: float = 0.40
    W_EPISODE_DIRECTNESS: float = 0.20

    W_SUCCESS_BONUS: float = 0.25
    MAX_LATERAL_SPEED: float = 0.5  # Spot's lateral velocity limit (m/s)

    # Stability blend weights (with gait data)
    _STAB_W_STABILITY: float = 0.40
    _STAB_W_GAIT: float = 0.25
    _STAB_W_SMOOTHNESS: float = 0.20
    _STAB_W_SPEED: float = 0.15

    # Stability blend weights (no gait data)
    _STAB_W_STABILITY_NOGAIT: float = 0.50
    _STAB_W_SMOOTHNESS_NOGAIT: float = 0.30
    _STAB_W_SPEED_NOGAIT: float = 0.20

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
        """Compute per-episode quality then apply success-rate amplification.

        Args:
            metrics: Aggregate metrics from all episodes.
            max_episode_steps: Step budget for time normalization fallback.
            episodes: Individual episode records (required for per-episode scoring).
            max_episode_time_s: Physical episode time budget in seconds.

        Returns:
            Score in [0, 1], or 0.0 when telemetry is absent or no episodes given.
        """
        if not episodes:
            return 0.0

        agg_stability = self._aggregate_stability(metrics)
        if agg_stability is None:
            return 0.0

        agg_directness = self._directness_quality(metrics.extra or {})

        success_quality: list[float] = []
        for ep in episodes:
            if not ep.success:
                continue
            time_eff = self._episode_time_efficiency(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            stab = self._stability_quality_from_extra(ep.extra)
            if stab is None:
                stab = agg_stability
            direct = self._directness_quality(ep.extra)
            if direct is None:
                direct = agg_directness if agg_directness is not None else 1.0
            q = (
                self.W_EPISODE_TIME * time_eff
                + self.W_EPISODE_STABILITY * stab
                + self.W_EPISODE_DIRECTNESS * direct
            )
            success_quality.append(float(q))

        if not success_quality:
            return 0.0

        success_rate = len(success_quality) / len(episodes)
        mean_quality = float(np.mean(success_quality))
        return float(success_rate * (self.W_SUCCESS_BONUS + (1.0 - self.W_SUCCESS_BONUS) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["task_type"] = "navigation.goal"
        d["weights"] = {
            "success_bonus": self.W_SUCCESS_BONUS,
            "episode_time_efficiency": self.W_EPISODE_TIME,
            "episode_stability": self.W_EPISODE_STABILITY,
            "episode_directness": self.W_EPISODE_DIRECTNESS,
            "formula": "success_rate × (W_SUCCESS_BONUS + (1 − W_SUCCESS_BONUS) × mean_quality)",
        }
        d["locomotion_thresholds"]["max_lateral_speed"] = self.MAX_LATERAL_SPEED
        return d

    # ------------------------------------------------------------------
    # Per-episode helpers
    # ------------------------------------------------------------------

    def _episode_time_efficiency(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Time efficiency in [0, 1], preferring physical-second normalization."""
        if max_episode_time_s and ep.completion_time is not None:
            norm = ep.completion_time / max_episode_time_s
        else:
            norm = ep.steps / max_episode_steps
        if norm > self.max_normalized_time:
            return 0.0
        return 1.0 - norm / self.max_normalized_time

    def _aggregate_stability(self, metrics: AggregateMetrics) -> float | None:
        """Stability score from aggregate extra; None when telemetry absent."""
        return self._stability_quality_from_extra(metrics.extra or {})

    def _stability_quality_from_extra(self, ex: dict[str, Any]) -> float | None:
        """Stability blend from a single episode or aggregate ``extra`` dict."""
        if not ex or "mean_speed" not in ex:
            return None

        slope_factor = self._slope_speed_factor(ex.get("mean_slope_deg", 0.0))
        speed_score = self._speed_compliance(ex.get("mean_speed", 0.0), ex.get("max_speed", 0.0), slope_factor)
        smoothness_score = self._smoothness(ex.get("speed_std", 0.0))
        stability_score = self._body_stability(ex.get("mean_vertical_speed", 0.0), ex.get("mean_roll_pitch_rate", 0.0))
        gait_score = self._gait_quality(ex.get("aerial_phase_fraction"))

        if gait_score is not None:
            return float(
                self._STAB_W_STABILITY * stability_score
                + self._STAB_W_GAIT * gait_score
                + self._STAB_W_SMOOTHNESS * smoothness_score
                + self._STAB_W_SPEED * speed_score
            )
        return float(
            self._STAB_W_STABILITY_NOGAIT * stability_score
            + self._STAB_W_SMOOTHNESS_NOGAIT * smoothness_score
            + self._STAB_W_SPEED_NOGAIT * speed_score
        )

    def _directness_quality(self, ex: dict[str, Any]) -> float | None:
        """1.0 when purely forward, 0.0 at max lateral speed; None if data absent."""
        mean_lat = ex.get("mean_lateral_speed")
        if mean_lat is None:
            return None
        if self.MAX_LATERAL_SPEED <= 0:
            return 1.0
        return max(0.0, 1.0 - mean_lat / self.MAX_LATERAL_SPEED)
