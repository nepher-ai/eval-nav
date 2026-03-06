# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Metric collection for navigation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    
    episode_id: int
    scene: str | int
    seed: int
    success: bool
    steps: int
    timeout: bool
    env_id: str | None = None
    """Environment ID."""
    completion_time: float | None = None
    """Completion time in seconds (only for successful episodes)."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Task-specific metrics (e.g. locomotion quality for quadrupeds).
    Populated by the episode runner when the environment exposes the data.
    Ignored by scorers that don't need it (V1)."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "episode_id": self.episode_id,
            "scene": self.scene,
            "seed": self.seed,
            "success": self.success,
            "steps": self.steps,
            "timeout": self.timeout,
            "env_id": self.env_id,
            "completion_time": self.completion_time,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all episodes."""
    
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    timeout_episodes: int
    success_rate: float
    mean_completion_time: float | None
    """Mean completion time for successful episodes only."""
    std_completion_time: float | None
    """Standard deviation of completion time for successful episodes."""
    mean_steps: float
    """Mean steps across all episodes."""
    std_steps: float
    """Standard deviation of steps across all episodes."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Aggregated task-specific metrics (mean of each numeric extra field)."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
            "timeout_episodes": self.timeout_episodes,
            "success_rate": self.success_rate,
            "mean_completion_time": self.mean_completion_time,
            "std_completion_time": self.std_completion_time,
            "mean_steps": self.mean_steps,
            "std_steps": self.std_steps,
        }
        if self.extra:
            d["extra"] = self.extra
        return d
    
    @classmethod
    def from_episodes(cls, episodes: list[EpisodeMetrics]) -> AggregateMetrics:
        """Compute aggregate metrics from episode list.
        
        Args:
            episodes: List of episode metrics.
            
        Returns:
            AggregateMetrics instance.
        """
        if not episodes:
            return cls(
                total_episodes=0,
                successful_episodes=0,
                failed_episodes=0,
                timeout_episodes=0,
                success_rate=0.0,
                mean_completion_time=None,
                std_completion_time=None,
                mean_steps=0.0,
                std_steps=0.0,
            )
        
        total = len(episodes)
        successful = sum(1 for e in episodes if e.success)
        failed = total - successful
        timeouts = sum(1 for e in episodes if e.timeout)
        success_rate = successful / total if total > 0 else 0.0
        
        successful_times = [e.completion_time for e in episodes if e.success and e.completion_time is not None]
        if successful_times:
            mean_time = float(np.mean(successful_times))
            std_time = float(np.std(successful_times))
        else:
            mean_time = None
            std_time = None
        
        all_steps = [e.steps for e in episodes]
        mean_steps = float(np.mean(all_steps))
        std_steps = float(np.std(all_steps))
        
        extra = cls._aggregate_extra(episodes)
        
        return cls(
            total_episodes=total,
            successful_episodes=successful,
            failed_episodes=failed,
            timeout_episodes=timeouts,
            success_rate=success_rate,
            mean_completion_time=mean_time,
            std_completion_time=std_time,
            mean_steps=mean_steps,
            std_steps=std_steps,
            extra=extra,
        )
    
    @staticmethod
    def _aggregate_extra(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
        """Compute mean of each numeric extra field across episodes."""
        buckets: dict[str, list[float]] = {}
        for ep in episodes:
            for k, v in ep.extra.items():
                if isinstance(v, (int, float)):
                    buckets.setdefault(k, []).append(float(v))
        return {k: float(np.mean(vals)) for k, vals in buckets.items()}

