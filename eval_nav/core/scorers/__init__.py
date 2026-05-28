# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Scorer registry — resolve task types to scorer instances.

Task-type names follow the ``<domain>.<variant>`` convention:

Navigation
----------
- ``navigation.simple``   — success + time; no locomotion required
                            (animal waypoint nav, leatherback waypoint nav)
- ``navigation.waypoint`` — success + time + locomotion quality
                            (Spot waypoint benchmark)
- ``navigation.goal``     — success-rate-amplified quality + path directness
                            (Spot obstacle-terrain goal nav)

Manipulation
------------
- ``manipulation.pick_place`` — task success + completion speed
                                (Franka high-level pick-and-place)

Legacy aliases (``scoring_version`` field)
------------------------------------------
- ``v1`` → navigation.simple / manipulation.pick_place context
- ``v2`` → navigation.waypoint
- ``v3`` → navigation.goal  (V3 was an intermediate; GoalNavScorer covers it)
- ``v4`` → navigation.goal
"""

from __future__ import annotations

from .base import BaseScorer
from .manipulation.pick_place import PickPlaceScorer
from .navigation.goal import GoalNavScorer
from .navigation.simple import SimpleNavScorer
from .navigation.waypoint import WaypointNavScorer

__all__ = [
    "BaseScorer",
    "SimpleNavScorer",
    "WaypointNavScorer",
    "GoalNavScorer",
    "PickPlaceScorer",
    "REGISTRY",
    "get_scorer",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[BaseScorer]] = {
    # Semantic task-type names (preferred)
    "navigation.simple": SimpleNavScorer,
    "navigation.waypoint": WaypointNavScorer,
    "navigation.goal": GoalNavScorer,
    "manipulation.pick_place": PickPlaceScorer,
    # Legacy version aliases (backward compatibility)
    "v1": SimpleNavScorer,
    "v2": WaypointNavScorer,
    "v3": GoalNavScorer,
    "v4": GoalNavScorer,
}

SUPPORTED_TASK_TYPES: tuple[str, ...] = (
    "navigation.simple",
    "navigation.waypoint",
    "navigation.goal",
    "manipulation.pick_place",
)

SUPPORTED_LEGACY_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3", "v4")


def get_scorer(task_type: str) -> BaseScorer:
    """Instantiate a scorer by task type or legacy version string.

    Args:
        task_type: A semantic task-type key (e.g. ``"navigation.goal"``) or a
            legacy version string (``"v1"``–``"v4"``).

    Returns:
        A fresh scorer instance ready for use.

    Raises:
        ValueError: When ``task_type`` is not in the registry.

    Examples:
        >>> scorer = get_scorer("navigation.goal")
        >>> scorer = get_scorer("manipulation.pick_place")
        >>> scorer = get_scorer("v4")  # legacy — same as navigation.goal
    """
    if task_type not in REGISTRY:
        supported = sorted(REGISTRY)
        raise ValueError(
            f"Unknown task type or scoring version: {task_type!r}. "
            f"Supported: {supported}"
        )
    return REGISTRY[task_type]()
