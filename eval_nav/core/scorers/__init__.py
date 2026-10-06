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

``navigation.humanoid`` — G1 waypoint race
    - v1 : success_rate × (0.25 + 0.75 × time_efficiency)

``navigation.humanoid.runjump`` — G1 obstacle-course HL
    - v1 : success_rate × (0.25 + 0.75 × (0.35×time + 0.30×clear_land + 0.20×track + 0.15×safety))
    - v2 : success_rate × mean(N × (0.40×speed + 0.20×clearance + 0.15×land_stable + 0.15×land_impact + 0.10×track))

``navigation.go2`` — Unitree Go2 LiDAR maze
    - v1 : success_rate × (0.25 + 0.75 × (0.45×time + 0.25×path + 0.20×stability + 0.10×envelope))

Manipulation
~~~~~~~~~~~~
``manipulation.pick_place`` — arm pick-and-place tasks (e.g. Franka HL)
    - v1 : task success (70%) + time efficiency (30%)  [deprecated: additive, SR not dominant]
    - v2 : success_rate × (0.75 + 0.25 × time_efficiency)  [success rate is first-class multiplier]

``manipulation.multitask`` — several manipulation families in one benchmark
    - v1 : mean over tasks of success_rate × (0.30 + 0.70 × quality);
           quality = 0.70 × speed + 0.30 × hand-path smoothness

Usage
-----
    from eval_nav.core.scorers import get_scorer

    scorer = get_scorer("navigation.spot", "v4")
    scorer = get_scorer("navigation.leatherback", "v1")
    scorer = get_scorer("manipulation.pick_place", "v2")
"""

from __future__ import annotations

from .base import BaseScorer
from .manipulation.multitask import MultitaskScorer
from .manipulation.pick_place import PickPlaceScorer, PickPlaceScorerV2
from .navigation.go2_maze import Go2MazeScorer
from .navigation.humanoid import HumanoidRaceScorer
from .navigation.humanoid_runjump import HumanoidRunJumpScorer
from .navigation.humanoid_runjump_v2 import HumanoidRunJumpScorerV2
from .navigation.leatherback import LeatherbackNavScorer
from .navigation.leatherback_maze import LeatherbackMazeScorer
from .navigation.spot import SpotGoalScorerV3, SpotGoalScorerV4, SpotWaypointScorer

__all__ = [
    "BaseScorer",
    "Go2MazeScorer",
    "HumanoidRaceScorer",
    "HumanoidRunJumpScorer",
    "HumanoidRunJumpScorerV2",
    "LeatherbackNavScorer",
    "LeatherbackMazeScorer",
    "SpotWaypointScorer",
    "SpotGoalScorerV3",
    "SpotGoalScorerV4",
    "MultitaskScorer",
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
    ("navigation.humanoid", "v1"): HumanoidRaceScorer,
    ("navigation.humanoid.runjump", "v1"): HumanoidRunJumpScorer,
    ("navigation.humanoid.runjump", "v2"): HumanoidRunJumpScorerV2,
    ("navigation.go2", "v1"): Go2MazeScorer,
    ("navigation.leatherback", "v1"): LeatherbackNavScorer,
    ("navigation.leatherback", "v2"): LeatherbackMazeScorer,
    ("navigation.spot", "v2"): SpotWaypointScorer,
    ("navigation.spot", "v3"): SpotGoalScorerV3,
    ("navigation.spot", "v4"): SpotGoalScorerV4,
    ("manipulation.pick_place", "v1"): PickPlaceScorer,
    ("manipulation.pick_place", "v2"): PickPlaceScorerV2,
    ("manipulation.multitask", "v1"): MultitaskScorer,
}

VALID_VERSIONS_PER_TASK_TYPE: dict[str, list[str]] = {
    "navigation.humanoid": ["v1"],
    "navigation.humanoid.runjump": ["v1", "v2"],
    "navigation.go2": ["v1"],
    "navigation.leatherback": ["v1", "v2"],
    "navigation.spot": ["v2", "v3", "v4"],
    "manipulation.pick_place": ["v1", "v2"],
    "manipulation.multitask": ["v1"],
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
