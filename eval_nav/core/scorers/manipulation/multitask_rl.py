# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Multitask manipulation scorer for in-process learned policies, version v1.

The runner reports success, step count, completion time, and, when the env exposes the tool
position, the end-effector speed of each episode.

    time_eff = clip(1 - T / T_budget, 0, 1)
    ee       = rise(v_mean) * fall(v_p95)
    episode  = success * (0.50 + 0.20 * time_eff + 0.30 * ee)

``T`` is the completion time [s] and ``T_budget`` is ``max_episode_time_s``. Without a time
budget, ``T`` is the step count and ``T_budget`` is ``max_episode_steps``.

``v_mean`` [m/s] is the mean tool speed on steps faster than 2 cm/s, and ``v_p95`` [m/s] is the
95th percentile of those steps. ``rise`` reaches 1 at 20 cm/s and stays there. ``fall`` stays 1
through 27 cm/s and reaches 0 at 35 cm/s. A missing speed sample scores ``ee`` as 0, so a success
stays at ``0.50 + 0.20 * time_eff``.

A failed episode scores 0. An episode's task is the task id the env recorded for it, or its
``scene`` name when the env did not. The task score is the mean episode score, and the suite
score is the unweighted mean of the task scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer

_W_SUCCESS = 0.50
_W_TIME = 0.20
_W_EE = 0.30
# Tool-speed band [m/s]. Steps at or below the moving floor are dwell, not travel.
_EE_MOVE_M_S = 0.02
_EE_SATURATE_M_S = 0.20
_EE_PENALTY_M_S = 0.27
_EE_ZERO_M_S = 0.35


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
        observed: dict[str, int] = {}
        means: dict[str, list[float]] = {}
        peaks: dict[str, list[float]] = {}
        for episode in episodes or []:
            task = _task_of(episode)
            scores.setdefault(task, []).append(_episode_score(episode, budget, use_steps=use_steps))
            successes[task] = successes.get(task, 0) + int(episode.success)
            mean = episode.extra.get("mean_ee_speed_m_s")
            peak = episode.extra.get("p95_ee_speed_m_s")
            if _finite(mean) and _finite(peak):
                observed[task] = observed.get(task, 0) + 1
                means.setdefault(task, []).append(float(mean))
                peaks.setdefault(task, []).append(float(peak))

        self.tasks = {}
        for task, values in scores.items():
            report: dict[str, Any] = {
                "episodes": len(values),
                "successes": successes[task],
                "success_rate": successes[task] / len(values),
                "task_score": float(np.mean(values)),
                "ee_speed_observed": observed.get(task, 0) / len(values),
            }
            if task in means:
                report["mean_ee_speed_m_s"] = float(np.mean(means[task]))
                report["mean_p95_ee_speed_m_s"] = float(np.mean(peaks[task]))
            self.tasks[task] = report
        self.task_rates = {task: report["success_rate"] for task, report in self.tasks.items()}
        if not self.tasks:
            return 0.0
        return float(np.mean([report["task_score"] for report in self.tasks.values()]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.multitask.rl",
            "scoring_version": self.VERSION,
            "formula": (
                "episode = success * (0.50 + 0.20 * time_eff + 0.30 * ee); "
                "ee = rise(mean speed) * fall(p95 speed)"
            ),
            "time_budget_s": self.time_budget_s,
            "weights": {"success": _W_SUCCESS, "time": _W_TIME, "ee_speed": _W_EE},
            "ee_speed_m_s": {
                "moving_above": _EE_MOVE_M_S,
                "saturate": _EE_SATURATE_M_S,
                "penalty": _EE_PENALTY_M_S,
                "zero": _EE_ZERO_M_S,
            },
            "task_rates": self.task_rates,
            "tasks": self.tasks,
        }


def ee_speed_stats(speeds_m_s: np.ndarray, dt_s: float | None = None) -> dict[str, float]:
    """Mean and 95th percentile [m/s] of tool speeds faster than the dwell floor.

    An empty moving set is a still hand: both values are 0. The keys mark the episode as observed.
    ``dt_s`` [s] is one control step. When it is set, ``ee_path_length_m`` is the summed step
    displacement [m], including steps at or below the dwell floor.
    """
    speeds = np.asarray(speeds_m_s, dtype=np.float64).reshape(-1)
    finite = speeds[np.isfinite(speeds)]
    moving = finite[finite > _EE_MOVE_M_S]
    if moving.size == 0:
        mean = 0.0
        peak = 0.0
    else:
        mean = float(moving.mean())
        peak = float(np.percentile(moving, 95))
    stats = {"mean_ee_speed_m_s": mean, "p95_ee_speed_m_s": peak}
    if dt_s is not None and float(dt_s) > 0:
        stats["ee_path_length_m"] = float(finite.sum() * float(dt_s))
    return stats


def _task_of(episode: EpisodeMetrics) -> str:
    """Task id of an episode, or its scene name when the env did not record one."""
    if episode.task_id:
        return str(episode.task_id)
    return str(episode.scene)


def _episode_score(episode: EpisodeMetrics, budget: float, *, use_steps: bool) -> float:
    """Score one episode. A success stays at or above 0.50 when the hand-speed term is 0."""
    if not episode.success or budget <= 0:
        return 0.0
    elapsed = float(episode.steps) if use_steps else episode.completion_time
    time_eff = 0.0 if elapsed is None else float(np.clip(1.0 - float(elapsed) / budget, 0.0, 1.0))
    return _W_SUCCESS + _W_TIME * time_eff + _W_EE * _ee_factor(episode)


def _ee_factor(episode: EpisodeMetrics) -> float:
    """Hand-speed factor in ``[0, 1]``. Missing samples score 0."""
    mean = episode.extra.get("mean_ee_speed_m_s")
    peak = episode.extra.get("p95_ee_speed_m_s")
    if not _finite(mean) or not _finite(peak):
        return 0.0
    rise = float(np.clip(float(mean) / _EE_SATURATE_M_S, 0.0, 1.0))
    span = _EE_ZERO_M_S - _EE_PENALTY_M_S
    fall = float(np.clip((_EE_ZERO_M_S - float(peak)) / span, 0.0, 1.0))
    return rise * fall


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False
