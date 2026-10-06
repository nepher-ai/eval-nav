# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

import sys
import threading
import time
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / "nepher-brain-comm"))

from eval_nav.benchmark.expand import expand, make_shards
from eval_nav.benchmark.lockstep import run_shard
from eval_nav.benchmark.manifest import manifest_from_dict
from eval_nav.benchmark.scheduler import assign_groups, group_shards
from eval_nav.core.scorers import get_scorer
from eval_nav.domain.config import EvalConfig
from eval_nav.domain.metrics import AggregateMetrics, EpisodeMetrics
from eval_nav.runtime.brain import BrainRuntime
from nepher_brain_comm.client import BrainClient
from nepher_brain_comm.serve import run_replica


MANIFEST = {
    "version": "tabletop-phase1-v1",
    "shard_size": 4,
    "seed_salt": "phase1",
    "tasks": [
        {
            "task_id": "place_in_container",
            "scenes": [{"composer": "kitchen_counter", "count": 2}],
            "variants": [{"name": "nominal"}, {"name": "jitter10", "pose_jitter_m": 0.10}],
            "instruction_pool": "phase1_hidden",
            "episodes": 8,
        },
        {
            "task_id": "stack_on",
            "scenes": [{"composer": "workbench", "count": 1}],
            "variants": [{"name": "nominal"}],
            "instruction_pool": "phase1_hidden",
            "episodes": 3,
        },
    ],
}


def test_expansion_ignores_gpu_count():
    manifest = manifest_from_dict(MANIFEST)
    jobs = expand(manifest)
    shards = make_shards(jobs, manifest.shard_size)
    groups = group_shards(shards)
    flattened = [shard.jobs for shard in shards]
    expected = sorted(job.job_id for jobs in flattened for job in jobs)
    for gpu_count in (1, 2, 4, 8):
        assigned = assign_groups(groups, gpu_count)
        again = [shard.jobs for bucket in assigned for group in bucket for shard in group]
        assert sorted(job.job_id for jobs in again for job in jobs) == expected
    assert shards[0].jobs[0].seed == expand(manifest)[0].seed


def test_shards_stay_inside_one_group():
    shards = make_shards(expand(manifest_from_dict(MANIFEST)), 4)
    for shard in shards:
        keys = {(job.task_id, job.scene_id, job.variant) for job in shard.jobs}
        assert len(keys) == 1
        assert len(shard.jobs) <= 4
    # 2 scenes x 2 variants x 8 episodes = 32, plus 1 scene x 3 episodes.
    assert sum(len(shard.jobs) for shard in shards) == 32 + 3


def test_sparc_is_closer_to_zero_for_a_minimum_jerk_reach():
    from eval_nav.core.smoothness import sparc_from_positions

    dt = 0.04
    t = np.arange(0.0, 4.0, dt)
    u = t / t[-1]
    reach = 10 * u**3 - 15 * u**4 + 6 * u**5
    smooth = np.stack((reach, np.zeros_like(reach), np.zeros_like(reach)), axis=1)
    wobble = reach + 0.03 * np.sin(2 * np.pi * 8 * t)
    jerky = np.stack((wobble, 0.02 * np.sin(2 * np.pi * 7 * t), np.zeros_like(reach)), axis=1)
    smooth_sparc = sparc_from_positions(smooth, dt)
    jerky_sparc = sparc_from_positions(jerky, dt)
    assert smooth_sparc is not None and jerky_sparc is not None
    assert smooth_sparc > jerky_sparc


def test_multitask_score_mixes_speed_and_smoothness():
    # Budget 20 s. place_relative finishes at 8 s on a minimum-jerk path.
    # speed = 0.6, smoothness = 1, quality = 0.72, task_score = 0.804.
    # place_in_container never succeeds, so its task score is 0.
    episodes = [
        EpisodeMetrics(
            0,
            "a",
            1,
            True,
            200,
            False,
            completion_time=8.0,
            extra={"task_id": "place_relative", "sparc": -1.40},
        ),
        EpisodeMetrics(
            1,
            "b",
            1,
            False,
            500,
            True,
            extra={"task_id": "place_in_container", "sparc": -4.0},
        ),
    ]
    scorer = get_scorer("manipulation.multitask", "v1")
    score = scorer.compute_score(AggregateMetrics.from_episodes(episodes), 500, episodes, max_episode_time_s=20.0)
    relative = scorer.tasks["place_relative"]
    assert abs(relative["speed"] - 0.6) < 1e-9
    assert abs(relative["smoothness"] - 1.0) < 1e-9
    assert abs(relative["quality"] - 0.72) < 1e-9
    assert abs(relative["task_score"] - 0.804) < 1e-9
    assert scorer.tasks["place_in_container"]["task_score"] == 0.0
    assert scorer.task_rates["place_in_container"] == 0.0
    assert abs(score - 0.402) < 1e-9


def test_multitask_clips_a_late_jerky_success():
    # Finishing at the time budget with the jerky SPARC bound: speed 0, smoothness 0.
    # task_score = 1 * 0.30.
    episodes = [
        EpisodeMetrics(
            0,
            "a",
            1,
            True,
            500,
            False,
            completion_time=20.0,
            extra={"task_id": "place_relative", "sparc": -4.0},
        )
    ]
    scorer = get_scorer("manipulation.multitask", "v1")
    score = scorer.compute_score(AggregateMetrics.from_episodes(episodes), 500, episodes, max_episode_time_s=20.0)
    assert scorer.tasks["place_relative"]["speed"] == 0.0
    assert scorer.tasks["place_relative"]["smoothness"] == 0.0
    assert abs(score - 0.30) < 1e-9


def test_evaluation_summary_lists_task_terms(tmp_path: Path):
    from eval_nav.benchmark.aggregate import score_records, write_outputs

    records = [
        {
            "job_id": "a",
            "task_id": "place_relative",
            "scene_id": "kitchen_counter-0",
            "variant": "nominal",
            "episode_index": 0,
            "seed": 1,
            "success": True,
            "failed": False,
            "steps": 200,
            "timeout": False,
            "completion_time_s": 8.0,
            "sparc": -1.40,
            "trajectory_hash": "abc",
        }
    ]
    score, metrics, report = score_records(records, "manipulation.multitask", "v1", 500, max_episode_time_s=20.0)
    write_outputs(tmp_path, score, metrics, records, report=report, metadata={"runtime": "brain"})
    text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "task place_relative:" in text
    assert "task_score:" in text
    assert "time_s=8.000000" in text
    assert "smoothness=1.000000" in text
    payload = (tmp_path / "evaluation_result.json").read_text(encoding="utf-8")
    assert '"tasks"' in payload
    assert '"episodes"' in payload


def test_brain_config_requires_brain_fields(tmp_path: Path):
    path = tmp_path / "eval.yaml"
    path.write_text(
        "task_name: Nepher-FrankaTabletop-Envhub-Play-v0\n"
        "task_type: manipulation.multitask\n"
        "scoring_version: v1\n"
        "category: manipulation\n"
        "runtime: brain\n"
        "num_envs: 4\n"
        "env_scenes:\n"
        "  - {env_id: tabletop-phase1-v1, scene: 0}\n",
        encoding="utf-8",
    )
    config = EvalConfig.from_yaml(path)
    try:
        config.validate()
    except ValueError as exc:
        assert "benchmark_env_id" in str(exc) or "brain." in str(exc)
    else:
        raise AssertionError("brain config without fields should fail validation")

    path.write_text(
        "task_name: Nepher-FrankaTabletop-Envhub-Play-v0\n"
        "task_type: manipulation.multitask\n"
        "scoring_version: v1\n"
        "category: manipulation\n"
        "runtime: brain\n"
        "num_envs: 4\n"
        "seeds: [1]\n"
        "benchmark_env_id: tabletop-phase1-v0\n"
        "brain:\n"
        "  socket_dir: /run/brain\n"
        "  step_timeout_s: 60\n"
        "  open_loop_horizon: 8\n",
        encoding="utf-8",
    )
    EvalConfig.from_yaml(path).validate()


def test_lockstep_hashes_match_across_runs():
    shard = make_shards(expand(manifest_from_dict(MANIFEST)), 2)[0]
    first = run_shard(_FakeEnv(len(shard.jobs)), _ZeroRuntime(), shard, open_loop_horizon=2, max_steps=4)
    second = run_shard(_FakeEnv(len(shard.jobs)), _ZeroRuntime(), shard, open_loop_horizon=2, max_steps=4)
    assert [row["trajectory_hash"] for row in first] == [row["trajectory_hash"] for row in second]
    assert all(row["success"] for row in first)


def test_brain_runtime_against_zero_brain(tmp_path: Path):
    import argparse

    submission = SOURCE / "nepher-brain-comm" / "examples" / "zero_brain"
    args = argparse.Namespace(
        entry=None,
        replicas=1,
        socket_dir=str(tmp_path),
        submission=str(submission),
        max_gb=None,
    )
    threading.Thread(target=run_replica, args=(0, args), daemon=True).start()
    deadline = time.time() + 5
    socket_path = tmp_path / "brain-0.sock"
    while time.time() < deadline and not socket_path.exists():
        time.sleep(0.02)
    client = BrainClient(socket_path, timeout_s=5)
    client.connect()
    runtime = BrainRuntime(client)
    obs = {"probe": np.zeros((1, 4), dtype=np.float32)}
    runtime.begin_episode(["job"], [1], ["put the cup in the bin"])
    first = runtime.act(obs, 0, 1)
    second = runtime.act(obs, 0, 1)
    runtime.end_episode(["job"])
    client.close()
    assert first.shape == (1, 1, 8)
    assert np.array_equal(first, second)


class _ZeroRuntime:
    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step, seed):
        count = next(iter(obs.values())).shape[0]
        return np.zeros((count, 2, 8), dtype=np.float32)

    def end_episode(self, episode_ids):
        return None


class _FakeEnv:
    def __init__(self, count: int):
        self.count = count
        self._steps = 0
        self._done = np.zeros(count, dtype=bool)

    def reset(self, *, seed: int):
        self._steps = 0
        self._done[:] = False
        return self.get_obs(), {}

    def get_obs(self):
        return {"joint_pos": np.zeros((self.count, 7), dtype=np.float32)}

    def step(self, action):
        self._steps += 1
        if self._steps >= 2:
            self._done[:] = True
        return self.get_obs(), None, self._done, self._done, {}

    def task_completed(self):
        return self._done.copy()

    def task_failed(self):
        return np.zeros(self.count, dtype=bool)

    def instructions(self):
        return ["put the cup in the bin"] * self.count

    def get_raw_state(self):
        return np.zeros((self.count, 3), dtype=np.float64)
