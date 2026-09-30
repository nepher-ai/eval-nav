# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Go2 LiDAR maze scorer (v1).

Scores an online planner on the ``go2-maze-v0`` benchmark. Each maze is
judged against its own optimal route length and against the Unitree Go2 EDU
speed envelope (rated 3.7 m/s, peak 5 m/s). There is no yaw-rate term:
Unitree does not publish one, and 2 rad/s is only this task's command clip.

Formula
-------
    score = success_rate × (BASE + (1 − BASE) × mean_quality)

    BASE = 0.25

    quality (per successful episode):
        = 0.45 × E_time + 0.25 × E_path + 0.20 × S_stab + 0.10 × S_env

    E_time = min(1, L* / (T × 3.7 m/s))
        L* is ``extra["course_length_m"]`` (optimal route, metres) and T is
        the completion time. Faster than the rated speed saturates at 1.
        Without L*, falls back to ``max(0, 1 − T / max_episode_time_s)``.

    E_path = min(1, L* / (mean_speed × T))
        Walked distance over the optimal route. Dead-end exploration lowers
        it. Without L* or a mean speed, the term is 1.

    S_stab = max(0, 1 − mean_roll_pitch_rate / RP_REF)

    S_env = 1 up to 3.7 m/s, then linear to 0 at 5.0 m/s.
        Catches a simulated speed the hardware cannot hold.

    A missing telemetry field scores 1.0 for that term.

Used by
-------
    ``task_type: "navigation.go2"``, ``scoring_version: "v1"``
    → configs/task-go2-lidar-maze.yaml
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer

# Unitree Go2 EDU: sustained 0–3.7 m/s, peak about 5 m/s.
V_RATED_M_S: float = 3.7
V_PEAK_M_S: float = 5.0
# Mean |roll/pitch rate| [rad/s] that scores 0. Twice the 1.886 rad/s mean of
# best_policy.pt on go2-maze-v0 (eval_run_20260929_115911, 60 episodes).
# A run like that baseline scores about 0.5; twice its rate scores 0.
RP_REF_RAD_S: float = 3.772


class Go2MazeScorer(BaseScorer):
    """Go2 maze navigation scorer (v1).

    Success-rate-amplified score. Quality is time against the rated speed,
    path efficiency against the optimal route, body stability, and a
    hardware-envelope guard.
    """

    VERSION: str = "v1"
    BASE: float = 0.25
    W_TIME: float = 0.45
    W_PATH: float = 0.25
    W_STAB: float = 0.20
    W_ENV: float = 0.10

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute the maze score in [0, 1]. Zero when nothing succeeded."""
        del metrics
        if not episodes:
            return 0.0
        qualities = [
            self._episode_quality(ep, max_episode_steps, max_episode_time_s=max_episode_time_s)
            for ep in episodes
            if ep.success
        ]
        if not qualities:
            return 0.0
        success_rate = len(qualities) / len(episodes)
        mean_quality = float(np.mean(qualities))
        return float(success_rate * (self.BASE + (1.0 - self.BASE) * mean_quality))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "navigation.go2",
            "scoring_version": self.VERSION,
            "weights": {
                "time_efficiency": self.W_TIME,
                "path_efficiency": self.W_PATH,
                "stability": self.W_STAB,
                "speed_envelope": self.W_ENV,
            },
            "limits": {
                "rated_speed_m_s": V_RATED_M_S,
                "peak_speed_m_s": V_PEAK_M_S,
                "roll_pitch_ref_rad_s": RP_REF_RAD_S,
            },
        }

    def _episode_quality(
        self,
        ep: EpisodeMetrics,
        max_episode_steps: int,
        *,
        max_episode_time_s: float | None,
    ) -> float:
        """Quality in [0, 1] for one successful episode."""
        extra = ep.extra or {}
        length = _finite(extra.get("course_length_m"))
        return float(
            self.W_TIME * self._time_efficiency(ep, length, max_episode_steps, max_episode_time_s)
            + self.W_PATH * self._path_efficiency(ep, length)
            + self.W_STAB * self._stability(extra)
            + self.W_ENV * self._envelope(extra)
        )

    def _time_efficiency(
        self,
        ep: EpisodeMetrics,
        route_m: float | None,
        max_episode_steps: int,
        max_episode_time_s: float | None,
    ) -> float:
        """Fraction of the rated-speed time budget that was left unused, capped at 1."""
        if route_m is not None and ep.completion_time and ep.completion_time > 0:
            return float(min(1.0, route_m / (ep.completion_time * V_RATED_M_S)))
        if max_episode_time_s and max_episode_time_s > 0 and ep.completion_time is not None:
            return float(max(0.0, 1.0 - ep.completion_time / max_episode_time_s))
        if max_episode_steps > 0:
            return float(max(0.0, 1.0 - ep.steps / max_episode_steps))
        return 0.0

    @staticmethod
    def _path_efficiency(ep: EpisodeMetrics, route_m: float | None) -> float:
        """Optimal route over distance actually walked. 1 when the route is unknown."""
        if route_m is None or not ep.completion_time or ep.completion_time <= 0:
            return 1.0
        mean_speed = _finite((ep.extra or {}).get("mean_speed"))
        if mean_speed is None or mean_speed <= 0.0:
            return 1.0
        walked = mean_speed * ep.completion_time
        if walked <= 0.0:
            return 1.0
        return float(min(1.0, route_m / walked))

    @staticmethod
    def _stability(extra: dict[str, Any]) -> float:
        """1 at rest, 0 at ``RP_REF`` mean roll/pitch rate. 1 if unmeasured."""
        rate = _finite(extra.get("mean_roll_pitch_rate"))
        if rate is None or RP_REF_RAD_S <= 0.0:
            return 1.0
        return float(max(0.0, 1.0 - rate / RP_REF_RAD_S))

    @staticmethod
    def _envelope(extra: dict[str, Any]) -> float:
        """1 up to the rated speed, 0 at the published peak. 1 if unmeasured."""
        speed = _finite(extra.get("max_speed"))
        if speed is None:
            return 1.0
        if speed <= V_RATED_M_S:
            return 1.0
        span = V_PEAK_M_S - V_RATED_M_S
        if span <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, (V_PEAK_M_S - speed) / span)))


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
