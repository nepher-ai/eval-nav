# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Tests for the in-process multitask scorer and the franka-tabletop-rl config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval_nav.core.episode_runner import _ee_speed_extra
from eval_nav.core.scorers import get_scorer
from eval_nav.core.scorers.manipulation.multitask_rl import ee_speed_stats
from eval_nav.domain.config import EvalConfig
from eval_nav.domain.metrics import AggregateMetrics, EpisodeMetrics

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "task-franka-tabletop-rl.yaml"


def _episode(
    scene: str,
    success: bool,
    steps: int = 300,
    completion_time: float | None = None,
    episode_id: int = 0,
    env_id: str | None = None,
    task_id: str | None = None,
    extra: dict | None = None,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=episode_id,
        scene=scene,
        seed=1,
        success=success,
        steps=steps,
        timeout=not success,
        env_id=env_id,
        completion_time=completion_time if success else None,
        task_id=task_id,
        extra=extra or {},
    )


def _score(episodes: list[EpisodeMetrics], **kwargs) -> tuple[float, object]:
    scorer = get_scorer("manipulation.multitask.rl", "v1")
    metrics = AggregateMetrics.from_episodes(episodes)
    return scorer.compute_score(metrics, 600, episodes, **kwargs), scorer


def test_success_scores_between_floor_and_one_by_speed():
    # No hand-speed sample: the ee term is 0, and a success stays on the 0.50 floor plus time.
    fast, _ = _score([_episode("a", True, completion_time=0.0)], max_episode_time_s=40.0)
    slow, _ = _score([_episode("a", True, completion_time=40.0)], max_episode_time_s=40.0)
    half, _ = _score([_episode("a", True, completion_time=20.0)], max_episode_time_s=40.0)
    assert fast == pytest.approx(0.70)
    assert slow == pytest.approx(0.50)
    assert half == pytest.approx(0.60)


def test_hand_speed_in_band_lifts_the_score_and_over_speed_does_not_zero_it():
    in_band = {"mean_ee_speed_m_s": 0.24, "p95_ee_speed_m_s": 0.26}
    over = {"mean_ee_speed_m_s": 0.30, "p95_ee_speed_m_s": 0.40}
    partial = {"mean_ee_speed_m_s": 0.10, "p95_ee_speed_m_s": 0.31}
    perfect, scorer = _score(
        [_episode("a", True, completion_time=0.0, extra=in_band)], max_episode_time_s=40.0
    )
    slow_in_band, _ = _score(
        [_episode("a", True, completion_time=40.0, extra=in_band)], max_episode_time_s=40.0
    )
    fast_over, _ = _score(
        [_episode("a", True, completion_time=0.0, extra=over)], max_episode_time_s=40.0
    )
    mixed, _ = _score(
        [_episode("a", True, completion_time=40.0, extra=partial)], max_episode_time_s=40.0
    )
    assert perfect == pytest.approx(1.0)
    assert slow_in_band == pytest.approx(0.80)
    assert fast_over == pytest.approx(0.70)
    assert mixed == pytest.approx(0.50 + 0.30 * 0.5 * 0.5)
    assert scorer.tasks["a"]["ee_speed_observed"] == pytest.approx(1.0)
    assert scorer.tasks["a"]["mean_ee_speed_m_s"] == pytest.approx(0.24)


def test_failure_scores_zero_and_tasks_weigh_equally():
    episodes = [
        _episode("many", True, completion_time=20.0),
        _episode("many", True, completion_time=20.0),
        _episode("many", True, completion_time=20.0),
        _episode("one", False),
    ]
    score, scorer = _score(episodes, max_episode_time_s=40.0)
    assert score == pytest.approx((0.60 + 0.0) / 2)
    assert scorer.task_rates == {"many": 1.0, "one": 0.0}


def test_step_budget_is_used_without_a_time_budget():
    score, scorer = _score([_episode("a", True, steps=300)])
    assert score == pytest.approx(0.50 + 0.20 * 0.5)
    assert scorer.to_dict()["time_budget_s"] is None
    assert scorer.to_dict()["weights"] == {"success": 0.50, "time": 0.20, "ee_speed": 0.30}


def test_a_scene_that_runs_several_tasks_uses_the_episode_task_id():
    # The env records which task each episode ran. The scorer does not infer it from the episode number.
    names = ["a", "b", "c"]
    outcomes = [True, False, True, True, False, False] + [True] * 6
    episodes = [
        _episode(
            "all",
            success,
            completion_time=20.0,
            episode_id=index,
            env_id="bundle",
            task_id=names[index % 3],
        )
        for index, success in enumerate(outcomes)
    ]
    score, scorer = _score(episodes, max_episode_time_s=40.0)
    assert scorer.task_rates == {"a": 1.0, "b": 0.5, "c": 0.75}
    assert {task: report["episodes"] for task, report in scorer.tasks.items()} == {"a": 4, "b": 4, "c": 4}
    assert score == pytest.approx(0.60 * (1.0 + 0.5 + 0.75) / 3)


def test_ee_speed_stats_ignore_dwell_and_report_a_still_hand_as_zero():
    still = ee_speed_stats(np.array([0.0, 0.01, 0.02]))
    assert still == {"mean_ee_speed_m_s": 0.0, "p95_ee_speed_m_s": 0.0}
    moving = ee_speed_stats(np.array([0.0, 0.02, 0.25, 0.30]), dt_s=0.1)
    assert moving["mean_ee_speed_m_s"] == pytest.approx(0.275)
    assert moving["p95_ee_speed_m_s"] == pytest.approx(0.2975)
    assert moving["ee_path_length_m"] == pytest.approx(0.057)


def test_runner_copies_ee_speed_when_the_env_reports_it():
    class _Env:
        def episode_ee_speeds(self):
            return [np.array([0.0, 0.25])]

    assert _ee_speed_extra(_Env(), 0)["mean_ee_speed_m_s"] == pytest.approx(0.25)
    assert _ee_speed_extra(_Env(), 0, step_dt=0.1)["ee_path_length_m"] == pytest.approx(0.025)
    assert _ee_speed_extra(object(), 0) == {}


def test_config_loads_and_validates():
    config = EvalConfig.from_yaml(CONFIG)
    config.validate()
    assert (config.task_type, config.scoring_version) == ("manipulation.multitask.rl", "v1")
    assert config.env_scenes == [{"env_id": "franka-tabletop-rl-v0", "scene": "all"}]
    assert config.num_envs % 16 == 0
