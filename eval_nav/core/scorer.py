# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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


def get_scorer(version: str) -> V1Scorer:
    """Get scorer by version.
    
    Args:
        version: Scoring version ('v1' for MVP).
        
    Returns:
        Scorer instance.
        
    Raises:
        ValueError: If version is not supported.
    """
    if version == "v1":
        return V1Scorer()
    else:
        raise ValueError(f"Unsupported scoring version: {version}. MVP supports 'v1' only.")

