# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Domain models for navigation evaluation.

This module contains the core domain models:
- Configuration (EvalConfig)
- Metrics (EpisodeMetrics, AggregateMetrics)
- Errors (EvaluationStatus, EvaluationError, etc.)
"""

from .config import EvalConfig
from .errors import (
    EnvironmentError,
    EvaluationError,
    EvaluationRuntimeError,
    EvaluationStatus,
    EvaluationTimeoutError,
)
from .metrics import AggregateMetrics, EpisodeMetrics

__all__ = [
    # Configuration
    "EvalConfig",
    # Metrics
    "EpisodeMetrics",
    "AggregateMetrics",
    # Errors
    "EvaluationStatus",
    "EvaluationError",
    "EnvironmentError",
    "EvaluationRuntimeError",
    "EvaluationTimeoutError",
]

