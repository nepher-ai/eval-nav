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
- ``navigation.go2``         — Go2 LiDAR maze
    - v1: success-rate-amplified; time vs 3.7 m/s, path efficiency, stability, speed envelope
"""

from .go2_maze import Go2MazeScorer
from .leatherback import LeatherbackNavScorer
from .leatherback_maze import LeatherbackMazeScorer
from .humanoid import HumanoidRaceScorer
from .humanoid_runjump import HumanoidRunJumpScorer
from .humanoid_runjump_v2 import HumanoidRunJumpScorerV2
from .spot import SpotGoalScorerV3, SpotGoalScorerV4, SpotWaypointScorer

__all__ = [
    "Go2MazeScorer",
    "HumanoidRaceScorer",
    "HumanoidRunJumpScorer",
    "HumanoidRunJumpScorerV2",
    "LeatherbackNavScorer",
    "LeatherbackMazeScorer",
    "SpotWaypointScorer",
    "SpotGoalScorerV3",
    "SpotGoalScorerV4",
]
