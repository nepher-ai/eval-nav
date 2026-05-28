# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Backward-compatibility shim for the legacy V1–V4 scorer API.

New code should use ``eval_nav.core.scorers.get_scorer(task_type, scoring_version)``.

This module re-exports the new classes under the old names so that any
existing call-sites continue to work without modification:

    from eval_nav.core.scorer import get_scorer, V1Scorer, V4Scorer

Legacy single-argument ``get_scorer("v4")`` is still accepted via the
``_LEGACY_MAP`` below, which maps old version strings to the closest
equivalent scorer class.
"""

from __future__ import annotations

from .scorers import (
    REGISTRY,
    SUPPORTED_TASK_TYPES,
    VALID_VERSIONS_PER_TASK_TYPE,
    BaseScorer,
    LeatherbackNavScorer,
    PickPlaceScorer,
    SpotGoalScorerV3,
    SpotGoalScorerV4,
    SpotWaypointScorer,
    get_scorer,
)

# ---------------------------------------------------------------------------
# Legacy class aliases  (V1Scorer → LeatherbackNavScorer, etc.)
# ---------------------------------------------------------------------------

V1Scorer = LeatherbackNavScorer
V2Scorer = SpotWaypointScorer
V3Scorer = SpotGoalScorerV3
V4Scorer = SpotGoalScorerV4

__all__ = [
    # New names
    "BaseScorer",
    "LeatherbackNavScorer",
    "SpotWaypointScorer",
    "SpotGoalScorerV3",
    "SpotGoalScorerV4",
    "PickPlaceScorer",
    "REGISTRY",
    "SUPPORTED_TASK_TYPES",
    "VALID_VERSIONS_PER_TASK_TYPE",
    "get_scorer",
    # Legacy aliases
    "V1Scorer",
    "V2Scorer",
    "V3Scorer",
    "V4Scorer",
]
