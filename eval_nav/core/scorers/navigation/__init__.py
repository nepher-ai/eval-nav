# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Navigation scorers, grouped by robot platform and scoring version.

Navigation task types
---------------------
- ``navigation.leatherback`` — leatherback and ANYmal B waypoint tasks (v1)
- ``navigation.spot``        — Spot waypoint (v2) and goal-nav (v3, v4) tasks
"""

from .leatherback import LeatherbackNavScorer
from .spot import SpotGoalScorerV3, SpotGoalScorerV4, SpotWaypointScorer

__all__ = [
    "LeatherbackNavScorer",
    "SpotWaypointScorer",
    "SpotGoalScorerV3",
    "SpotGoalScorerV4",
]
