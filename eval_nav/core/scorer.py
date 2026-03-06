# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Scoring system for navigation evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..domain.metrics import AggregateMetrics, EpisodeMetrics


class V1Scorer:
    """V1 Scoring System - MVP version.
    
    Focuses on fundamental navigation capability:
    - Success rate (primary metric)
    - Normalized time-to-completion (secondary metric)
    
    Final score is normalized to [0, 1] range.
    Failures are heavily penalized.
    """
    
    def __init__(self, max_normalized_time: float = 1.0):
        """Initialize V1 scorer.
        
        Args:
            max_normalized_time: Maximum normalized time value for scaling.
                                Episodes taking longer than this get 0 time score.
        """
        self.max_normalized_time = max_normalized_time
    
    def compute_score(self, metrics: AggregateMetrics, max_episode_steps: int) -> float:
        """Compute final score from aggregate metrics.
        
        Args:
            metrics: Aggregate metrics from evaluation.
            max_episode_steps: Maximum steps per episode (for normalization).
            
        Returns:
            Final score in [0, 1] range.
        """
        success_component = metrics.success_rate
        
        if metrics.mean_completion_time is not None and metrics.successful_episodes > 0:
            normalized_time = metrics.mean_completion_time / max_episode_steps
            if normalized_time > self.max_normalized_time:
                time_component = 0.0
            else:
                time_component = 1.0 - (normalized_time / self.max_normalized_time)
        else:
            time_component = 0.0
        
        final_score = 0.7 * success_component + 0.3 * time_component
        
        return float(final_score)
    
    def compute_score_from_steps(self, metrics: AggregateMetrics, max_episode_steps: int, episodes: list[EpisodeMetrics] | None = None) -> float:
        """Compute score using steps instead of time (fallback).
        
        Args:
            metrics: Aggregate metrics from evaluation.
            max_episode_steps: Maximum steps per episode.
            episodes: Optional list of EpisodeMetrics for computing mean successful steps.
            
        Returns:
            Final score in [0, 1] range.
        """
        success_component = metrics.success_rate
        
        if metrics.successful_episodes > 0 and episodes:
            successful_steps = [
                e.steps for e in episodes if e.success
            ]
            if successful_steps:
                mean_successful_steps = float(np.mean(successful_steps))
                normalized_steps = mean_successful_steps / max_episode_steps
                if normalized_steps > self.max_normalized_time:
                    time_component = 0.0
                else:
                    time_component = 1.0 - (normalized_steps / self.max_normalized_time)
            else:
                time_component = 0.0
        elif metrics.mean_completion_time is not None:
            normalized_time = metrics.mean_completion_time / max_episode_steps
            if normalized_time > self.max_normalized_time:
                time_component = 0.0
            else:
                time_component = 1.0 - (normalized_time / self.max_normalized_time)
        else:
            time_component = 0.0
        
        final_score = 0.7 * success_component + 0.3 * time_component
        return float(final_score)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert scorer configuration to dictionary."""
        return {
            "version": "v1",
            "max_normalized_time": self.max_normalized_time,
            "weights": {
                "success_rate": 0.7,
                "time_to_completion": 0.3,
            },
        }


class V2Scorer:
    """V2 Scoring System — adds locomotion quality on top of V1.

    Designed for quadruped (Spot) waypoint navigation where we want to verify
    the robot walks stably and does not run.

    Score = w_s * success + w_t * time_efficiency + w_l * locomotion_quality

    ``locomotion_quality`` is derived from per-episode extra metrics collected
    by the episode runner.  The primary walking signal is foot contact data:
    walking means at least one of the four feet is always on the ground (no
    aerial phase).  Speed and stability heuristics remain as secondary signals.

    When locomotion data is absent (e.g. leatherback) the scorer silently
    falls back to V1 behaviour.
    """

    W_SUCCESS: float = 0.50
    W_TIME: float = 0.20
    W_LOCOMOTION: float = 0.30

    # Spot walking speed envelope (m/s).  Spot's physical top speed (1.6 m/s)
    # *is* a walking gait — there is no separate running mode.  Speeds above
    # the limit therefore indicate sim-physics anomalies, not gait changes.
    MAX_WALKING_SPEED: float = 1.6
    MIN_USEFUL_SPEED: float = 0.05
    SIM_SPEED_TOLERANCE: float = 0.1     # (m/s) — headroom for brief sim overshoots

    # Thresholds above which the sub-component score drops to zero.
    MAX_SPEED_STD: float = 0.8          # (m/s) — higher ⇒ jerky pace
    MAX_VERTICAL_SPEED: float = 0.3     # (m/s) — higher ⇒ bouncing / galloping
    MAX_ROLL_PITCH_RATE: float = 1.0    # (rad/s) — higher ⇒ body wobble

    # Aerial phase tolerance.  Any fraction above this is penalised harshly.
    MAX_AERIAL_PHASE: float = 0.05      # 5 % — allow a tiny margin for sensor noise

    def __init__(self, max_normalized_time: float = 1.0):
        self.max_normalized_time = max_normalized_time

    # ------------------------------------------------------------------

    def compute_score_from_steps(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
    ) -> float:
        success_component = metrics.success_rate
        time_component = self._time_component(metrics, max_episode_steps, episodes)
        loco_component = self._locomotion_component(metrics, episodes)

        if loco_component is None:
            return 0.0

        score = (
            self.W_SUCCESS * success_component
            + self.W_TIME * time_component
            + self.W_LOCOMOTION * loco_component
        )
        return float(score)

    # Alias used by evaluator
    compute_score = compute_score_from_steps

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
                if norm > self.max_normalized_time:
                    return 0.0
                return 1.0 - norm / self.max_normalized_time
        if metrics.mean_completion_time is not None:
            norm = metrics.mean_completion_time / max_episode_steps
            if norm > self.max_normalized_time:
                return 0.0
            return 1.0 - norm / self.max_normalized_time
        return 0.0

    def _locomotion_component(
        self,
        metrics: AggregateMetrics,
        episodes: list[EpisodeMetrics] | None,
    ) -> float | None:
        """Compute locomotion quality in [0, 1].  Returns None when data is absent."""
        ex = metrics.extra
        if not ex or "mean_speed" not in ex:
            return None

        mean_slope_deg = ex.get("mean_slope_deg", 0.0)
        slope_factor = self._slope_speed_factor(mean_slope_deg)

        speed_score = self._speed_compliance(
            ex.get("mean_speed", 0.0), ex.get("max_speed", 0.0), slope_factor,
        )
        smoothness_score = self._smoothness(ex.get("speed_std", 0.0))
        stability_score = self._body_stability(
            ex.get("mean_vertical_speed", 0.0),
            ex.get("mean_roll_pitch_rate", 0.0),
        )

        gait_score = self._gait_quality(ex.get("aerial_phase_fraction"))

        if gait_score is not None:
            return (
                0.35 * gait_score
                + 0.25 * speed_score
                + 0.20 * smoothness_score
                + 0.20 * stability_score
            )

        return 0.40 * speed_score + 0.30 * smoothness_score + 0.30 * stability_score

    # ------------------------------------------------------------------
    # Individual locomotion sub-scores (all return [0, 1])
    # ------------------------------------------------------------------

    def _gait_quality(self, aerial_phase_fraction: float | None) -> float | None:
        """1.0 when always walking (≥1 foot on ground), 0.0 when frequently airborne.

        Returns None when foot contact data is unavailable.
        """
        if aerial_phase_fraction is None:
            return None
        if aerial_phase_fraction <= self.MAX_AERIAL_PHASE:
            return 1.0
        overshoot = (aerial_phase_fraction - self.MAX_AERIAL_PHASE) / (1.0 - self.MAX_AERIAL_PHASE)
        return max(0.0, 1.0 - overshoot)

    @staticmethod
    def _slope_speed_factor(mean_slope_deg: float) -> float:
        """cos(slope) scaling — walking is slower uphill.

        Returns a factor in (0, 1] that scales the maximum acceptable walking
        speed.  On flat ground the factor is 1.0; at 45° it is ~0.71.
        Clamped to [0.3, 1.0] so extreme slopes don't collapse the threshold
        to near-zero.
        """
        slope_rad = np.radians(np.clip(mean_slope_deg, 0.0, 60.0))
        return float(np.clip(np.cos(slope_rad), 0.3, 1.0))

    def _speed_compliance(
        self, mean_speed: float, max_speed: float, slope_factor: float = 1.0,
    ) -> float:
        """1.0 when speed stays within the physical walking envelope.

        Spot's top speed (1.6 m/s) *is* a walking gait — there is no separate
        running mode.  This rewards speeds inside the hardware envelope and
        penalises anomalies (stuck, or sim-physics overshoots beyond the
        physical limit).

        *slope_factor* scales the envelope for inclines.
        """
        effective_max = self.MAX_WALKING_SPEED * slope_factor

        if mean_speed < self.MIN_USEFUL_SPEED:
            return 0.0

        if mean_speed <= effective_max:
            mean_ok = 1.0
        else:
            overshoot = (mean_speed - effective_max) / effective_max
            mean_ok = max(0.0, 1.0 - overshoot * 2.0)

        peak_limit = effective_max + self.SIM_SPEED_TOLERANCE
        if max_speed <= peak_limit:
            peak_ok = 1.0
        else:
            overshoot = (max_speed - peak_limit) / effective_max
            peak_ok = max(0.0, 1.0 - overshoot * 5.0)

        return 0.7 * mean_ok + 0.3 * peak_ok

    def _smoothness(self, speed_std: float) -> float:
        """Low speed variance → smooth gait."""
        if speed_std >= self.MAX_SPEED_STD:
            return 0.0
        return 1.0 - speed_std / self.MAX_SPEED_STD

    def _body_stability(self, mean_vert_speed: float, mean_rp_rate: float) -> float:
        """Low vertical bounce + low roll/pitch → stable body."""
        vert = max(0.0, 1.0 - mean_vert_speed / self.MAX_VERTICAL_SPEED) if self.MAX_VERTICAL_SPEED > 0 else 1.0
        rp = max(0.0, 1.0 - mean_rp_rate / self.MAX_ROLL_PITCH_RATE) if self.MAX_ROLL_PITCH_RATE > 0 else 1.0
        return 0.5 * vert + 0.5 * rp

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "v2",
            "max_normalized_time": self.max_normalized_time,
            "weights": {
                "success_rate": self.W_SUCCESS,
                "time_to_completion": self.W_TIME,
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


def get_scorer(version: str) -> V1Scorer | V2Scorer:
    """Get scorer by version.
    
    Args:
        version: Scoring version ('v1' or 'v2').
        
    Returns:
        Scorer instance.
        
    Raises:
        ValueError: If version is not supported.
    """
    if version == "v1":
        return V1Scorer()
    elif version == "v2":
        return V2Scorer()
    else:
        raise ValueError(f"Unsupported scoring version: {version}. Supported: 'v1', 'v2'.")

