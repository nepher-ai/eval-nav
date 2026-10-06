# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Go2 maze scorer. Loaded without the eval_nav package init (that pulls Isaac Lab)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def _install() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    _namespace("eval_nav", _ROOT / "eval_nav")
    _namespace("eval_nav.core", _ROOT / "eval_nav" / "core")
    _namespace("eval_nav.domain", _ROOT / "eval_nav" / "domain")


_install()

from eval_nav.core.scorers import get_scorer  # noqa: E402
from eval_nav.core.scorers.navigation.go2_maze import (  # noqa: E402
    Go2MazeScorer,
    RP_REF_RAD_S,
    V_PEAK_M_S,
    V_RATED_M_S,
)
from eval_nav.domain.config import EvalConfig  # noqa: E402
from eval_nav.domain.metrics import AggregateMetrics, EpisodeMetrics  # noqa: E402


def _episode(**extra) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=0,
        scene=0,
        seed=42,
        success=True,
        steps=500,
        timeout=False,
        env_id="go2-maze-v0",
        completion_time=50.0,
        extra=extra,
    )


def test_registry_resolves_navigation_go2_v1():
    scorer = get_scorer("navigation.go2", "v1")
    assert isinstance(scorer, Go2MazeScorer)
    config = EvalConfig(
        task_name="Nepher-Go2-LidarMaze-Planner-Envhub-Play-v0",
        num_envs=60,
        task_type="navigation.go2",
        scoring_version="v1",
        env_scenes=[{"env_id": "go2-maze-v0", "scene": 0}],
    )
    config.validate()


def test_all_failures_score_zero():
    failed = _episode(course_length_m=122.5, mean_speed=2.0, max_speed=3.0, mean_roll_pitch_rate=0.2)
    failed.success = False
    metrics = AggregateMetrics.from_episodes([failed])
    assert Go2MazeScorer().compute_score(metrics, 3000, [failed], max_episode_time_s=300.0) == 0.0


def test_one_success_matches_the_hand_calculation():
    """122.5 m in 50 s at 2 m/s mean, peak 3 m/s, roll/pitch rate 0.25 rad/s."""
    episode = _episode(
        course_length_m=122.5,
        mean_speed=2.0,
        max_speed=3.0,
        mean_roll_pitch_rate=0.25,
    )
    time_eff = 122.5 / (50.0 * V_RATED_M_S)
    path_eff = 1.0  # 122.5 / (2 * 50) > 1
    stability = max(0.0, 1.0 - 0.25 / RP_REF_RAD_S)
    envelope = 1.0
    quality = 0.45 * time_eff + 0.25 * path_eff + 0.20 * stability + 0.10 * envelope
    expected = 0.25 + 0.75 * quality
    metrics = AggregateMetrics.from_episodes([episode])
    score = Go2MazeScorer().compute_score(metrics, 3000, [episode], max_episode_time_s=300.0)
    assert np.isclose(score, expected)


def test_speed_envelope_ramps_from_rated_to_peak():
    scorer = Go2MazeScorer()
    assert scorer._envelope({"max_speed": V_RATED_M_S}) == 1.0
    assert np.isclose(scorer._envelope({"max_speed": 0.5 * (V_RATED_M_S + V_PEAK_M_S)}), 0.5)
    assert scorer._envelope({"max_speed": V_PEAK_M_S}) == 0.0
    assert scorer._envelope({}) == 1.0


def test_missing_route_length_uses_the_time_budget():
    """No L*: time is 1 - 50/300, path efficiency is 1, other terms default to 1."""
    episode = _episode()
    time_eff = 1.0 - 50.0 / 300.0
    quality = 0.45 * time_eff + 0.25 * 1.0 + 0.20 * 1.0 + 0.10 * 1.0
    expected = 0.25 + 0.75 * quality
    metrics = AggregateMetrics.from_episodes([episode])
    score = Go2MazeScorer().compute_score(metrics, 3000, [episode], max_episode_time_s=300.0)
    assert np.isclose(score, expected)
