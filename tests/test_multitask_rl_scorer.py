# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Tests for the in-process multitask scorer and the franka-tabletop-rl config."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_nav.core.scorers import get_scorer
from eval_nav.domain.config import EvalConfig
from eval_nav.domain.metrics import AggregateMetrics, EpisodeMetrics

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "task-franka-tabletop-rl.yaml"


def _episode(scene: str, success: bool, steps: int = 300, completion_time: float | None = None) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=0,
        scene=scene,
        seed=1,
        success=success,
        steps=steps,
        timeout=not success,
        completion_time=completion_time if success else None,
    )


def _score(episodes: list[EpisodeMetrics], **kwargs) -> tuple[float, object]:
    scorer = get_scorer("manipulation.multitask.rl", "v1")
    metrics = AggregateMetrics.from_episodes(episodes)
    return scorer.compute_score(metrics, 600, episodes, **kwargs), scorer


def test_success_scores_between_floor_and_one_by_speed():
    fast, _ = _score([_episode("a", True, completion_time=0.0)], max_episode_time_s=40.0)
    slow, _ = _score([_episode("a", True, completion_time=40.0)], max_episode_time_s=40.0)
    half, _ = _score([_episode("a", True, completion_time=20.0)], max_episode_time_s=40.0)
    assert fast == pytest.approx(1.0)
    assert slow == pytest.approx(0.75)
    assert half == pytest.approx(0.875)


def test_failure_scores_zero_and_tasks_weigh_equally():
    episodes = [
        _episode("many", True, completion_time=20.0),
        _episode("many", True, completion_time=20.0),
        _episode("many", True, completion_time=20.0),
        _episode("one", False),
    ]
    score, scorer = _score(episodes, max_episode_time_s=40.0)
    assert score == pytest.approx((0.875 + 0.0) / 2)
    assert scorer.task_rates == {"many": 1.0, "one": 0.0}


def test_step_budget_is_used_without_a_time_budget():
    score, scorer = _score([_episode("a", True, steps=300)])
    assert score == pytest.approx(0.75 + 0.25 * 0.5)
    assert scorer.to_dict()["time_budget_s"] is None


def test_config_loads_and_validates():
    config = EvalConfig.from_yaml(CONFIG)
    config.validate()
    assert (config.task_type, config.scoring_version) == ("manipulation.multitask.rl", "v1")
    assert len(config.env_scenes) == 16
    assert len({scene["scene"] for scene in config.env_scenes}) == 16
    assert {scene["env_id"] for scene in config.env_scenes} == {"franka-tabletop-rl-v0"}
