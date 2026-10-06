# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Multitask manipulation scorer.

The score is the unweighted mean of per-task success rates. A task that is
attempted more often does not count for more than a task that is attempted
less often. Per-task rates are kept on ``task_rates`` for the result metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....domain.metrics import AggregateMetrics, EpisodeMetrics
from ..base import BaseScorer


class MultitaskScorer(BaseScorer):
    """``manipulation.multitask`` version ``v1``."""

    VERSION = "v1"

    def __init__(self) -> None:
        self.task_rates: dict[str, float] = {}

    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        del metrics, max_episode_steps, max_episode_time_s
        buckets: dict[str, list[bool]] = {}
        for episode in episodes or []:
            task_id = str(episode.extra.get("task_id", "default"))
            buckets.setdefault(task_id, []).append(bool(episode.success))
        self.task_rates = {task_id: sum(flags) / len(flags) for task_id, flags in buckets.items()}
        if not self.task_rates:
            return 0.0
        return float(np.mean(list(self.task_rates.values())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": "manipulation.multitask",
            "scoring_version": self.VERSION,
            "formula": "mean over tasks of success_rate(task)",
            "task_rates": self.task_rates,
        }
