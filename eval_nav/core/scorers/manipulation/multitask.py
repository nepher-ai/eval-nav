# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Multitask manipulation scorer.

Each task is scored on its own, then the tasks are averaged. A task with more
episodes does not outweigh a task with fewer. Inside a task the success rate
multiplies speed and hand-path smoothness, so a miss scores 0.

    speed      = clip(1 - T / T_budget, 0, 1)
    smoothness = clip((S_jerky - S) / (S_jerky - S_smooth), 0, 1)
    quality    = 0.7 * speed + 0.3 * smoothness
    task_score = success_rate * (0.30 + 0.70 * quality)

``T`` is the simulated time [s] at which the episode first succeeds. ``T_budget``
is ``max_episode_time_s`` [s]. ``S`` is the SPARC of the hand speed up to that
moment. ``S_smooth`` and ``S_jerky`` are fixed bounds, not fitted to the submission.
Quality is the mean over successful episodes only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer

# SPARC bounds for a 25 Hz hand path. -1.40 is a minimum-jerk reach.
# -4.00 is that reach plus an 8 Hz, 3 cm oscillation. Closer to zero is smoother.
S_SMOOTH = -1.40
S_JERKY = -4.00
_W_SPEED = 0.7
_W_SMOOTH = 0.3
_W_SUCCESS = 0.30
_W_QUALITY = 0.70


class MultitaskScorer(BaseScorer):
    """``manipulation.multitask`` version ``v1``."""

    VERSION = "v1"

    def __init__(self) -> None:
        self.task_rates: dict[str, float] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.episodes: list[dict[str, Any]] = []
        self.time_budget_s: float | None = None

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        del metrics
        rows = list(episodes or [])
        use_steps = max_episode_time_s is None or max_episode_time_s <= 0
        if use_steps:
            budget = float(max_episode_steps) if max_episode_steps > 0 else None
            self.time_budget_s = None
        else:
            budget = float(max_episode_time_s)
            self.time_budget_s = budget

        buckets: dict[str, list[dict[str, Any]]] = {}
        self.episodes = []
        for episode in rows:
            task_id = str(episode.extra.get("task_id", "default"))
            speed, smoothness = _episode_terms(episode, budget, use_steps=use_steps)
            row = {
                "task_id": task_id,
                "success": bool(episode.success),
                "steps": int(episode.steps),
                "completion_time_s": episode.completion_time,
                "sparc": episode.extra.get("sparc"),
                "speed": speed,
                "smoothness": smoothness,
            }
            self.episodes.append(row)
            buckets.setdefault(task_id, []).append(row)

        self.tasks = {}
        self.task_rates = {}
        for task_id, group in buckets.items():
            report = _task_report(group)
            self.tasks[task_id] = report
            self.task_rates[task_id] = report["success_rate"]
        if not self.tasks:
            return 0.0
        return float(np.mean([report["task_score"] for report in self.tasks.values()]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.multitask",
            "scoring_version": self.VERSION,
            "formula": (
                "task_score = success_rate * (0.30 + 0.70 * quality); "
                "quality = 0.70 * speed + 0.30 * smoothness"
            ),
            "time_budget_s": self.time_budget_s,
            "sparc_smooth": S_SMOOTH,
            "sparc_jerky": S_JERKY,
            "weights": {"speed": _W_SPEED, "smoothness": _W_SMOOTH, "success_floor": _W_SUCCESS, "quality": _W_QUALITY},
            "task_rates": self.task_rates,
            "tasks": self.tasks,
            "episodes": self.episodes,
        }


def _episode_terms(episode: EpisodeMetrics, budget: float | None, *, use_steps: bool) -> tuple[float | None, float | None]:
    """Speed and smoothness for one episode. Failures stay out of the quality mean."""
    if not episode.success or budget is None or budget <= 0:
        return None, None
    elapsed = float(episode.steps) if use_steps else episode.completion_time
    if elapsed is None:
        return None, None
    speed = float(np.clip(1.0 - float(elapsed) / budget, 0.0, 1.0))
    sparc = episode.extra.get("sparc")
    if sparc is None:
        smoothness = 0.0
    else:
        span = S_JERKY - S_SMOOTH
        smoothness = float(np.clip((S_JERKY - float(sparc)) / span, 0.0, 1.0))
    return speed, smoothness


def _task_report(group: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in group if row["success"]]
    success_rate = len(successes) / len(group)
    speeds = [row["speed"] for row in successes if row["speed"] is not None]
    smooth = [row["smoothness"] for row in successes if row["smoothness"] is not None]
    speed = float(np.mean(speeds)) if speeds else None
    smoothness = float(np.mean(smooth)) if smooth else None
    if speed is None or smoothness is None:
        quality = None
        task_score = 0.0
    else:
        quality = _W_SPEED * speed + _W_SMOOTH * smoothness
        task_score = success_rate * (_W_SUCCESS + _W_QUALITY * quality)
    return {
        "episodes": len(group),
        "successes": len(successes),
        "success_rate": success_rate,
        "speed": speed,
        "smoothness": smoothness,
        "quality": quality,
        "task_score": task_score,
    }
