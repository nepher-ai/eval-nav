# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Spot goal navigation scorer — v4.

Extends v3 by adding **path directness** as an explicit per-episode signal
and restructuring aggregation so success rate is a **first-class multiplier**
rather than only implicitly lowering the mean via zeros.

Formula
-------
Per successful episode:
    quality = W_TIME × time_efficiency
            + W_STABILITY × stability_quality
            + W_DIRECTNESS × directness_quality

Final score:
    score = success_rate × (W_SUCCESS_BONUS + (1 − W_SUCCESS_BONUS) × mean_quality)

Failed episodes are excluded from ``mean_quality``.
``W_SUCCESS_BONUS`` (0.25) gives a guaranteed floor credit per success so
that success rate's marginal influence is comparable to quality's.

Used by
-------
- ``task_type: "navigation.spot"``, ``scoring_version: "v4"``
  → task-spot-nav.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .....domain.metrics import AggregateMetrics, EpisodeMetrics
from .v3 import SpotGoalScorerV3


class SpotGoalScorerV4(SpotGoalScorerV3):
    """Spot obstacle-terrain goal navigation scorer — success-amplified + directness (v4).

    Episode weights (successful only)
    ----------------------------------
    - Time efficiency   : 0.40
    - Stability quality : 0.40
    - Path directness   : 0.20

    Aggregation
    -----------
    ``score = success_rate × (0.25 + 0.75 × mean_quality_over_successes)``
    """

    VERSION: str = "v4"

    W_EPISODE_TIME: float = 0.40
    W_EPISODE_STABILITY: float = 0.40
    W_EPISODE_DIRECTNESS: float = 0.20

    W_SUCCESS_BONUS: float = 0.25
    MAX_LATERAL_SPEED: float = 0.5  # Spot's lateral velocity limit (m/s)

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
        """Compute success-rate-amplified quality with directness.

        Returns:
            Score in [0, 1], or 0.0 when telemetry is absent or no successes.
        """
        if not episodes:
            return 0.0

        agg_stability = self._stability_from_extra(metrics.extra or {})
        if agg_stability is None:
            return 0.0

        agg_directness = self._directness_quality(metrics.extra or {})

        success_quality: list[float] = []
        for ep in episodes:
            if not ep.success:
                continue
            time_eff = self._episode_time_efficiency(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            stab = self._stability_from_extra(ep.extra)
            if stab is None:
                stab = agg_stability
            direct = self._directness_quality(ep.extra)
            if direct is None:
                direct = agg_directness if agg_directness is not None else 1.0
            success_quality.append(float(
                self.W_EPISODE_TIME * time_eff
                + self.W_EPISODE_STABILITY * stab
                + self.W_EPISODE_DIRECTNESS * direct
            ))

        if not success_quality:
            return 0.0

        success_rate = len(success_quality) / len(episodes)
        mean_quality = float(np.mean(success_quality))
        return float(success_rate * (self.W_SUCCESS_BONUS + (1.0 - self.W_SUCCESS_BONUS) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["scoring_version"] = self.VERSION
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
    # Directness helper
    # ------------------------------------------------------------------

    def _directness_quality(self, ex: dict[str, Any]) -> float | None:
        """1.0 when motion is purely forward, 0.0 at max lateral speed; None if absent."""
        mean_lat = ex.get("mean_lateral_speed")
        if mean_lat is None:
            return None
        if self.MAX_LATERAL_SPEED <= 0:
            return 1.0
        return max(0.0, 1.0 - mean_lat / self.MAX_LATERAL_SPEED)
