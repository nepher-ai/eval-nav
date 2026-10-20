# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Multitask manipulation scorer for in-process learned policies, version v1.

The in-process runner reports success, step count, and completion time for each episode. It does
not report subtask progress or hand telemetry, so this scorer uses only those three values.

    speed   = clip(1 - T / T_budget, 0, 1)
    episode = success * (0.75 + 0.25 * speed)

``T`` is the completion time [s] and ``T_budget`` is ``max_episode_time_s`` from the eval config.
Without a time budget, ``T`` is the step count and ``T_budget`` is ``max_episode_steps``. A failed
episode scores 0. An episode's task is its ``scene`` name. The task score is the mean episode
score, and the suite score is the unweighted mean of the task scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer

_W_SUCCESS = 0.75
_W_SPEED = 0.25


class MultitaskRLScorer(BaseScorer):
    """``manipulation.multitask.rl`` version ``v1``."""

    VERSION = "v1"

    def __init__(self) -> None:
        self.time_budget_s: float | None = None
        self.task_rates: dict[str, float] = {}
        self.tasks: dict[str, dict[str, Any]] = {}

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        del metrics
        use_steps = max_episode_time_s is None or max_episode_time_s <= 0
        budget = float(max_episode_steps) if use_steps else float(max_episode_time_s)
        self.time_budget_s = None if use_steps else budget

        scores: dict[str, list[float]] = {}
        successes: dict[str, int] = {}
        for episode in episodes or []:
            task = str(episode.scene)
            scores.setdefault(task, []).append(_episode_score(episode, budget, use_steps=use_steps))
            successes[task] = successes.get(task, 0) + int(episode.success)

        self.tasks = {
            task: {
                "episodes": len(values),
                "successes": successes[task],
                "success_rate": successes[task] / len(values),
                "task_score": float(np.mean(values)),
            }
            for task, values in scores.items()
        }
        self.task_rates = {task: report["success_rate"] for task, report in self.tasks.items()}
        if not self.tasks:
            return 0.0
        return float(np.mean([report["task_score"] for report in self.tasks.values()]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.multitask.rl",
            "scoring_version": self.VERSION,
            "formula": "episode_score = success * (0.75 + 0.25 * speed); speed = clip(1 - T / T_budget, 0, 1)",
            "time_budget_s": self.time_budget_s,
            "weights": {"success": _W_SUCCESS, "speed": _W_SPEED},
            "task_rates": self.task_rates,
            "tasks": self.tasks,
        }


def _episode_score(episode: EpisodeMetrics, budget: float, *, use_steps: bool) -> float:
    """Score one episode: 0 on failure, 0.75 to 1.0 on success by speed."""
    if not episode.success or budget <= 0:
        return 0.0
    elapsed = float(episode.steps) if use_steps else episode.completion_time
    speed = 0.0 if elapsed is None else float(np.clip(1.0 - float(elapsed) / budget, 0.0, 1.0))
    return _W_SUCCESS + _W_SPEED * speed
