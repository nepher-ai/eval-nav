# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Multitask manipulation scorer, version v1.

Each episode keeps the latched subtask progress in ``[0, 1]``. A finished episode
adds a bonus for speed and smoothness. An unfinished episode has no speed term, so
every finish scores at least 0.70 and every incomplete episode scores strictly less.

    progress   = latched subtask progress
    speed      = clip(1 - T / T_budget, 0, 1) when the episode finishes
    smoothness = clip((S_jerky - S) / (S_jerky - S_smooth), 0, 1)
    quality    = 0.70 * speed + 0.30 * smoothness
    episode    = 0.70 * progress + 0.30 * terminal * quality

``T`` is the simulated time [s] at the first finished step. ``T_budget`` is
``max_episode_time_s`` from the eval config. A missing SPARC does not drop the
episode: quality falls back to speed. The task score is the mean episode score.
The suite score is the unweighted mean of the task scores.

The report's progress term is ``success_rate + (1 - success_rate) * Score(fail)``
before the finish bonus. The report also includes the pooled episode mean, path
length, mean hand speed, and SPARC. Those three do not multiply the score.
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
_W_SPEED = 0.70
_W_SMOOTH = 0.30
_W_PROGRESS = 0.70
_W_FINISH = 0.30


class MultitaskScorer(BaseScorer):
    """``manipulation.multitask`` version ``v1``."""

    VERSION = "v1"

    def __init__(self) -> None:
        self.task_rates: dict[str, float] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.episodes: list[dict[str, Any]] = []
        self.time_budget_s: float | None = None
        self.pooled_score: float | None = None
        self.mean_path_length_m: float | None = None
        self.mean_hand_speed_mps: float | None = None
        self.mean_sparc: float | None = None

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
            row = _episode_row(episode, budget, use_steps=use_steps)
            row["task_id"] = task_id
            self.episodes.append(row)
            buckets.setdefault(task_id, []).append(row)

        self.tasks = {}
        self.task_rates = {}
        for task_id, group in buckets.items():
            report = _task_report(group)
            self.tasks[task_id] = report
            self.task_rates[task_id] = report["success_rate"]
        self.pooled_score = float(np.mean([row["episode_score"] for row in self.episodes])) if self.episodes else None
        self.mean_path_length_m = _mean_optional(self.episodes, "path_length_m")
        self.mean_hand_speed_mps = _mean_optional(self.episodes, "mean_hand_speed_mps")
        self.mean_sparc = _mean_optional(self.episodes, "sparc")
        if not self.tasks:
            return 0.0
        return float(np.mean([report["task_score"] for report in self.tasks.values()]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.multitask",
            "scoring_version": self.VERSION,
            "formula": (
                "episode_score = 0.70 * progress + 0.30 * terminal * quality; "
                "quality = 0.70 * speed + 0.30 * smoothness; "
                "a missing SPARC falls back to speed"
            ),
            "time_budget_s": self.time_budget_s,
            "sparc_smooth": S_SMOOTH,
            "sparc_jerky": S_JERKY,
            "weights": {
                "progress": _W_PROGRESS,
                "finish": _W_FINISH,
                "speed": _W_SPEED,
                "smoothness": _W_SMOOTH,
            },
            "pooled_score": self.pooled_score,
            "mean_path_length_m": self.mean_path_length_m,
            "mean_hand_speed_mps": self.mean_hand_speed_mps,
            "mean_sparc": self.mean_sparc,
            "task_rates": self.task_rates,
            "tasks": self.tasks,
            "episodes": self.episodes,
        }


def _episode_row(episode: EpisodeMetrics, budget: float | None, *, use_steps: bool) -> dict[str, Any]:
    """Score one episode. Speed and smoothness apply only after a finish."""
    progress = float(episode.extra.get("progress", 1.0 if episode.success else 0.0))
    progress = float(np.clip(progress, 0.0, 1.0))
    terminal = 1.0 if episode.success else 0.0
    speed = None
    smoothness = None
    quality = None
    if terminal and budget is not None and budget > 0:
        elapsed = float(episode.steps) if use_steps else episode.completion_time
        if elapsed is not None:
            speed = float(np.clip(1.0 - float(elapsed) / budget, 0.0, 1.0))
            sparc = episode.extra.get("sparc")
            if sparc is None:
                quality = speed
            else:
                span = S_JERKY - S_SMOOTH
                smoothness = float(np.clip((S_JERKY - float(sparc)) / span, 0.0, 1.0))
                quality = _W_SPEED * speed + _W_SMOOTH * smoothness
    finish = 0.0 if quality is None else quality
    return {
        "success": bool(episode.success),
        "steps": int(episode.steps),
        "completion_time_s": episode.completion_time,
        "progress": progress,
        "sparc": episode.extra.get("sparc"),
        "speed": speed,
        "smoothness": smoothness,
        "quality": quality,
        "episode_score": _W_PROGRESS * progress + _W_FINISH * terminal * finish,
        "path_length_m": episode.extra.get("path_length_m"),
        "mean_hand_speed_mps": episode.extra.get("mean_hand_speed_mps"),
    }


def _task_report(group: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [row for row in group if row["success"]]
    fails = [row["progress"] for row in group if not row["success"]]
    success_rate = len(finished) / len(group)
    fail_score = float(np.mean(fails)) if fails else 0.0
    progress = success_rate + (1.0 - success_rate) * fail_score
    speeds = [row["speed"] for row in finished if row["speed"] is not None]
    smooth = [row["smoothness"] for row in finished if row["smoothness"] is not None]
    qualities = [row["quality"] for row in finished if row["quality"] is not None]
    speed = float(np.mean(speeds)) if speeds else None
    smoothness = float(np.mean(smooth)) if smooth else None
    quality = float(np.mean(qualities)) if qualities else None
    task_score = float(np.mean([row["episode_score"] for row in group]))
    return {
        "episodes": len(group),
        "successes": len(finished),
        "unmeasured": sum(1 for row in finished if row["sparc"] is None),
        "success_rate": success_rate,
        "progress": progress,
        "speed": speed,
        "smoothness": smoothness,
        "quality": quality,
        "task_score": task_score,
    }


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))
