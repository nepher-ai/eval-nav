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

When one scene runs several tasks, the task module's ``tasks_of_scene`` lists them in env order. The
runner numbers episodes ``run * num_envs + i`` and ``num_envs`` is a multiple of that count, so episode
``n`` runs task ``n % T``. Keep that scene as the only env_scene, because episode numbers of several
env_scenes continue each other. A scene the module does not split is scored under its own name.
"""

from __future__ import annotations

import importlib
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
        self.task_module: str | None = None

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
            task = _task_of(episode, self.task_module)
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


def _scene_tasks(task_module: str | None, env_id: str | None, scene: str) -> list[str] | None:
    """Task ids ``task_module`` assigns to ``scene``. None when it does not split the scene."""
    if not task_module or env_id is None:
        return None
    declare = getattr(importlib.import_module(task_module), "tasks_of_scene", None)
    if declare is None:
        return None
    declared = declare(env_id, scene)
    return [str(task) for task in declared] if declared else None


def _task_of(episode: EpisodeMetrics, task_module: str | None) -> str:
    """Task id of an episode: its scene name, or task ``episode_id % T`` when the scene runs ``T`` tasks."""
    tasks = _scene_tasks(task_module, episode.env_id, str(episode.scene))
    if not tasks or tasks == [str(episode.scene)]:
        return str(episode.scene)
    return tasks[episode.episode_id % len(tasks)]


def _episode_score(episode: EpisodeMetrics, budget: float, *, use_steps: bool) -> float:
    """Score one episode: 0 on failure, 0.75 to 1.0 on success by speed."""
    if not episode.success or budget <= 0:
        return 0.0
    elapsed = float(episode.steps) if use_steps else episode.completion_time
    speed = 0.0 if elapsed is None else float(np.clip(1.0 - float(elapsed) / budget, 0.0, 1.0))
    return _W_SUCCESS + _W_SPEED * speed
