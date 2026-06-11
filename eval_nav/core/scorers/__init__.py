# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Scorer registry — resolve (task_type, scoring_version) to a scorer instance.

Task types and their supported scoring versions
-----------------------------------------------

Navigation
~~~~~~~~~~
``navigation.leatherback`` — leatherback waypoint nav and ANYmal B waypoint nav
    - v1 : success (70%) + time efficiency (30%)

``navigation.spot`` — Spot quadruped tasks
    - v2 : success (50%) + time (20%) + locomotion quality (30%)   [waypoint benchmark]
    - v3 : per-episode mean; fail=0, success = time(50%) + stability(50%)   [goal nav]
    - v4 : success_rate × (bonus + quality); quality = time(40%) + stability(40%) + directness(20%)   [goal nav + directness]

Manipulation
~~~~~~~~~~~~
``manipulation.pick_place`` — arm pick-and-place tasks (e.g. Franka HL)
    - v1 : task success (70%) + time efficiency (30%)  [deprecated: additive, SR not dominant]
    - v2 : success_rate × (0.75 + 0.25 × time_efficiency)  [success rate is first-class multiplier]

Usage
-----
    from eval_nav.core.scorers import get_scorer

    scorer = get_scorer("navigation.spot", "v4")
    scorer = get_scorer("navigation.leatherback", "v1")
    scorer = get_scorer("manipulation.pick_place", "v2")
"""

from __future__ import annotations

from .base import BaseScorer
from .manipulation.pick_place import PickPlaceScorer, PickPlaceScorerV2
from .navigation.leatherback import LeatherbackNavScorer
from .navigation.spot import SpotGoalScorerV3, SpotGoalScorerV4, SpotWaypointScorer

__all__ = [
    "BaseScorer",
    "LeatherbackNavScorer",
    "SpotWaypointScorer",
    "SpotGoalScorerV3",
    "SpotGoalScorerV4",
    "PickPlaceScorer",
    "PickPlaceScorerV2",
    "REGISTRY",
    "VALID_VERSIONS_PER_TASK_TYPE",
    "get_scorer",
]

# ---------------------------------------------------------------------------
# Registry — keyed by (task_type, scoring_version)
# ---------------------------------------------------------------------------

REGISTRY: dict[tuple[str, str], type[BaseScorer]] = {
    ("navigation.leatherback", "v1"): LeatherbackNavScorer,
    ("navigation.spot", "v2"): SpotWaypointScorer,
    ("navigation.spot", "v3"): SpotGoalScorerV3,
    ("navigation.spot", "v4"): SpotGoalScorerV4,
    ("manipulation.pick_place", "v1"): PickPlaceScorer,
    ("manipulation.pick_place", "v2"): PickPlaceScorerV2,
}

VALID_VERSIONS_PER_TASK_TYPE: dict[str, list[str]] = {
    "navigation.leatherback": ["v1"],
    "navigation.spot": ["v2", "v3", "v4"],
    "manipulation.pick_place": ["v1", "v2"],
}

SUPPORTED_TASK_TYPES: tuple[str, ...] = tuple(VALID_VERSIONS_PER_TASK_TYPE.keys())


def get_scorer(task_type: str, scoring_version: str) -> BaseScorer:
    """Instantiate a scorer for the given task type and scoring version.

    Args:
        task_type: The task domain, e.g. ``"navigation.spot"``.
        scoring_version: The version within that domain, e.g. ``"v4"``.

    Returns:
        A fresh scorer instance ready for use.

    Raises:
        ValueError: When the combination is not in the registry.

    Examples:
        >>> get_scorer("navigation.spot", "v4")
        SpotGoalScorerV4(...)
        >>> get_scorer("navigation.leatherback", "v1")
        LeatherbackNavScorer(...)
        >>> get_scorer("manipulation.pick_place", "v2")
        PickPlaceScorerV2(...)
    """
    key = (task_type, scoring_version)
    if key not in REGISTRY:
        if task_type not in VALID_VERSIONS_PER_TASK_TYPE:
            raise ValueError(
                f"Unknown task_type: {task_type!r}. "
                f"Supported task types: {SUPPORTED_TASK_TYPES}"
            )
        valid = VALID_VERSIONS_PER_TASK_TYPE[task_type]
        raise ValueError(
            f"Unsupported scoring_version {scoring_version!r} for task_type {task_type!r}. "
            f"Valid versions for this task type: {valid}"
        )
    return REGISTRY[key]()
