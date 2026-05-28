# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Backward-compatibility shim for the legacy V1–V4 scorer API.

New code should import from ``eval_nav.core.scorers`` directly and use
task-type keys (e.g. ``get_scorer("navigation.goal")``).

This module re-exports the new classes under the old names so that any
existing call-sites continue to work without modification:

    from eval_nav.core.scorer import get_scorer, V1Scorer, V4Scorer
"""

from .scorers import (
    BaseScorer,
    GoalNavScorer,
    PickPlaceScorer,
    REGISTRY,
    SimpleNavScorer,
    WaypointNavScorer,
    get_scorer,
)

# Legacy class aliases
V1Scorer = SimpleNavScorer
V2Scorer = WaypointNavScorer
V3Scorer = GoalNavScorer
V4Scorer = GoalNavScorer

__all__ = [
    # New names
    "BaseScorer",
    "SimpleNavScorer",
    "WaypointNavScorer",
    "GoalNavScorer",
    "PickPlaceScorer",
    "REGISTRY",
    "get_scorer",
    # Legacy names
    "V1Scorer",
    "V2Scorer",
    "V3Scorer",
    "V4Scorer",
]
