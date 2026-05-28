# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Abstract base class for all task-type scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...domain.metrics import AggregateMetrics, EpisodeMetrics


class BaseScorer(ABC):
    """Abstract base for all Nepher task scorers.

    Every concrete scorer must implement ``compute_score`` and ``to_dict``.
    The ``compute_score_from_steps`` name is kept as an alias for backward
    compatibility with callers that used the V1–V4 naming convention.
    """

    @abstractmethod
    def compute_score(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Compute a final score in [0, 1].

        Args:
            metrics: Aggregate metrics from all evaluated episodes.
            max_episode_steps: Episode step budget (used for time normalization).
            episodes: Individual episode records; required by per-episode scorers.
            max_episode_time_s: Physical time budget in seconds.  When provided,
                scorers that measure time efficiency normalize against seconds
                instead of steps, making the score decimation-invariant.

        Returns:
            Final score in the range [0, 1].
        """

    def compute_score_from_steps(
        self,
        metrics: AggregateMetrics,
        max_episode_steps: int,
        episodes: list[EpisodeMetrics] | None = None,
        *,
        max_episode_time_s: float | None = None,
    ) -> float:
        """Backward-compatible alias for ``compute_score``."""
        return self.compute_score(
            metrics,
            max_episode_steps,
            episodes,
            max_episode_time_s=max_episode_time_s,
        )

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize scorer configuration for logging / reporting."""
