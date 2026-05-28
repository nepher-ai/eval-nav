# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Spot goal navigation scorer — v3.

Per-episode scoring with a binary success gate:
- Failed episode → score = 0
- Successful episode → ``0.5 × time_efficiency + 0.5 × stability_quality``

Final score = mean of per-episode scores.

When ``max_episode_time_s`` is provided, time efficiency is computed from
physical seconds (decimation-invariant).  Falls back to step-based
normalization when seconds data is unavailable.

When locomotion aggregates are absent the scorer returns 0.0.

Used by
-------
- ``task_type: "navigation.spot"``, ``scoring_version: "v3"``
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .....domain.metrics import AggregateMetrics, EpisodeMetrics
from .v2 import SpotWaypointScorer


class SpotGoalScorerV3(SpotWaypointScorer):
    """Spot obstacle-terrain goal navigation scorer — per-episode binary gate (v3).

    Inherits all locomotion helpers from ``SpotWaypointScorer``.

    Episode weights (successful only)
    ----------------------------------
    - Time efficiency   : 0.50
    - Stability quality : 0.50

    Aggregation
    -----------
    ``score = mean(per_episode_scores)``   (failed episodes score 0)
    """

    VERSION: str = "v3"

    W_EPISODE_TIME: float = 0.50
    W_EPISODE_STABILITY: float = 0.50

    # Stability blend weights (with gait data)
    _STAB_W_STABILITY: float = 0.40
    _STAB_W_GAIT: float = 0.25
    _STAB_W_SMOOTHNESS: float = 0.20
    _STAB_W_SPEED: float = 0.15
    # Stability blend weights (no gait data)
    _STAB_W_STABILITY_NG: float = 0.50
    _STAB_W_SMOOTHNESS_NG: float = 0.30
    _STAB_W_SPEED_NG: float = 0.20

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
        """Compute per-episode mean with binary success gate.

        Returns:
            Score in [0, 1], or 0.0 when telemetry is absent.
        """
        if not episodes:
            return 0.0

        agg_stability = self._stability_from_extra(metrics.extra or {})
        if agg_stability is None:
            return 0.0

        episode_scores: list[float] = []
        for ep in episodes:
            if not ep.success:
                episode_scores.append(0.0)
                continue
            time_eff = self._episode_time_efficiency(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            stab = self._stability_from_extra(ep.extra)
            if stab is None:
                stab = agg_stability
            episode_scores.append(float(self.W_EPISODE_TIME * time_eff + self.W_EPISODE_STABILITY * stab))

        return float(np.mean(episode_scores)) if episode_scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["scoring_version"] = self.VERSION
        d["weights"] = {
            "episode_time_efficiency": self.W_EPISODE_TIME,
            "episode_stability": self.W_EPISODE_STABILITY,
            "formula": "mean(per_episode_scores)  [failed=0]",
        }
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
        return 0.0 if norm > self.max_normalized_time else 1.0 - norm / self.max_normalized_time

    def _stability_from_extra(self, ex: dict[str, Any]) -> float | None:
        """Stability blend from an episode or aggregate ``extra`` dict; None when absent."""
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
            self._STAB_W_STABILITY_NG * stability_score
            + self._STAB_W_SMOOTHNESS_NG * smoothness_score
            + self._STAB_W_SPEED_NG * speed_score
        )
