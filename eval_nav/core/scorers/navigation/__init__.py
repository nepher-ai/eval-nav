# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Navigation scorers, grouped by robot platform and scoring version.

Navigation task types
---------------------
- ``navigation.leatherback`` — leatherback/ANYmal B tasks
    - v1: success (70%) + time efficiency (30%)
    - v2: success-rate-amplified + time + speed/yaw-rate compliance
- ``navigation.spot``        — Spot waypoint (v2) and goal-nav (v3, v4) tasks
"""

from .leatherback import LeatherbackNavScorer
from .leatherback_maze import LeatherbackMazeScorer
from .spot import SpotGoalScorerV3, SpotGoalScorerV4, SpotWaypointScorer

__all__ = [
    "LeatherbackNavScorer",
    "LeatherbackMazeScorer",
    "SpotWaypointScorer",
    "SpotGoalScorerV3",
    "SpotGoalScorerV4",
]
