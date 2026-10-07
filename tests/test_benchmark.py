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
from eval_nav.benchmark.manifest import load_manifest, manifest_from_dict
from eval_nav.benchmark.scheduler import assign_groups, group_shards, plan_slots
from eval_nav.core.scorers import get_scorer
from eval_nav.domain.config import EvalConfig
from eval_nav.domain.metrics import AggregateMetrics, EpisodeMetrics
from eval_nav.runtime.brain import BrainRuntime
from nepher_brain_comm.client import BrainClient
from nepher_brain_comm.serve import run_replica

MANIFEST = {
    "version": "franka-tabletop-v1",
    "shard_size": 4,
    "seed_salt": "phase1",
    "tasks": [
        {
            "task_id": "banana_in_bowl",
            "scene": "banana_bowl",
            "instruction": "Pick up the banana and place it in the bowl",
            "episode_length_s": 50,
            "episodes": 8,
            "success": {"kind": "item_in_container", "roles": {"item": "banana", "container": "bowl"}},
        },
        {
            "task_id": "rubiks_cube_left_of_bowl",
            "scene": "rubiks_cube_banana_bowl",
            "instruction": "Put the rubiks cube to the left of the bowl",
            "episode_length_s": 30,
            "episodes": 3,
            "success": {
                "kind": "item_on_side",
                "side": "left",
                "roles": {"item": "rubiks_cube", "reference": "bowl"},
            },
        },
    ],
}


def test_authored_task_is_one_nominal_episode():
    manifest = manifest_from_dict(
        {
            "version": "franka-tabletop-v0",
            "shard_size": 4,
            "seed_salt": "phase1",
            "tasks": [
                {
                    "task_id": "banana_in_bowl",
                    "scene": "banana_bowl",
                    "instruction": "Pick up the banana and place it in the bowl",
                    "episode_length_s": 50,
                    "success": {"kind": "item_in_container", "roles": {"item": "banana", "container": "bowl"}},
                }
            ],
        }
    )
    jobs = expand(manifest)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.scene_id == "banana_bowl"
    assert job.variant == "nominal"
    assert job.pose_jitter_m == 0.0
    assert job.instruction == "Pick up the banana and place it in the bowl"
    assert manifest.tasks[0].success is not None
    assert manifest.tasks[0].success.kind == "item_in_container"


def test_phase_bundles_are_sixteen_distinct_tasks():
    root = SOURCE / "envhub" / "environments"
    seen: list[set[str]] = []
    for name in ("franka-tabletop-v0", "franka-tabletop-v1"):
        manifest = load_manifest(root / name / "benchmark.yaml")
        jobs = expand(manifest)
        task_ids = {job.task_id for job in jobs}
        assert len(jobs) == 16
        assert len(task_ids) == 16
        assert {job.variant for job in jobs} == {"nominal"}
        assert {job.pose_jitter_m for job in jobs} == {0.0}
        seen.append(task_ids)
    assert seen[0].isdisjoint(seen[1])


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
    shards = make_shards(expand(manifest_from_dict(MANIFEST), episodes=4), 4)
    for shard in shards:
        keys = {(job.task_id, job.scene_id, job.variant) for job in shard.jobs}
        assert len(keys) == 1
        assert len(shard.jobs) <= 4
        assert shard.jobs[0].variant == "nominal"
        assert shard.jobs[0].pose_jitter_m == 0.0
    # The manifest's episodes field is ignored. Four repeats of two tasks.
    assert len(expand(manifest_from_dict(MANIFEST))) == 2
    assert sum(len(shard.jobs) for shard in shards) == 8


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
    # speed = 0.6, smoothness = 1, quality = 0.72.
    # episode = 0.70 + 0.30 * 0.72 = 0.916. The other task never finishes.
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
    assert abs(relative["task_score"] - 0.916) < 1e-9
    assert scorer.tasks["place_in_container"]["task_score"] == 0.0
    assert scorer.task_rates["place_in_container"] == 0.0
    assert abs(score - 0.458) < 1e-9


def test_missing_sparc_scores_the_finish_from_speed():
    episodes = [
        EpisodeMetrics(
            0,
            "a",
            1,
            True,
            1,
            False,
            completion_time=0.04,
            extra={"task_id": "place_relative", "sparc": None},
        )
    ]
    scorer = get_scorer("manipulation.multitask", "v1")
    score = scorer.compute_score(AggregateMetrics.from_episodes(episodes), 500, episodes, max_episode_time_s=20.0)
    task = scorer.tasks["place_relative"]
    # T = 0.04 s, budget 20 s, speed = 0.998. Quality falls back to speed.
    assert abs(task["speed"] - 0.998) < 1e-9
    assert task["smoothness"] is None
    assert task["successes"] == 1
    assert task["unmeasured"] == 1
    assert abs(score - (0.70 + 0.30 * 0.998)) < 1e-9


def test_config_time_budget_is_the_speed_budget():
    # A recorded task length does not replace the eval config. 8 s of 20 s is speed 0.6.
    episodes = [
        EpisodeMetrics(
            0,
            "banana_bowl",
            1,
            True,
            120,
            False,
            completion_time=8.0,
            extra={"task_id": "banana_in_bowl", "sparc": -1.40, "time_budget_s": 50.0},
        )
    ]
    scorer = get_scorer("manipulation.multitask", "v1")
    scorer.compute_score(AggregateMetrics.from_episodes(episodes), 750, episodes, max_episode_time_s=20.0)
    assert abs(scorer.tasks["banana_in_bowl"]["speed"] - 0.6) < 1e-9


def test_partial_progress_scores_below_every_finish():
    # One of two objects placed is progress 0.5 and is not a finish: 0.70 * 0.5 = 0.35.
    half = EpisodeMetrics(
        0,
        "a",
        1,
        False,
        100,
        True,
        extra={"task_id": "both", "progress": 0.5, "sparc": -1.40},
    )
    instant = EpisodeMetrics(
        1,
        "a",
        1,
        True,
        1,
        False,
        completion_time=0.0,
        extra={"task_id": "instant", "progress": 1.0, "sparc": -1.40},
    )
    late = EpisodeMetrics(
        2,
        "a",
        1,
        True,
        300,
        False,
        completion_time=20.0,
        extra={"task_id": "late", "progress": 1.0, "sparc": None},
    )
    scorer = get_scorer("manipulation.multitask", "v1")
    scorer.compute_score(
        AggregateMetrics.from_episodes([half, instant, late]),
        300,
        [half, instant, late],
        max_episode_time_s=20.0,
    )
    assert abs(scorer.tasks["both"]["task_score"] - 0.35) < 1e-9
    assert abs(scorer.tasks["instant"]["task_score"] - 1.0) < 1e-9
    assert abs(scorer.tasks["late"]["task_score"] - 0.70) < 1e-9


def test_multitask_clips_a_late_jerky_success():
    # Finishing at the time budget with the jerky SPARC bound: speed 0, smoothness 0.
    # episode_score = 0.70.
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
    assert abs(score - 0.70) < 1e-9


def test_kind_free_success_block_loads():
    manifest = manifest_from_dict(
        {
            "version": "franka-tabletop-v0",
            "shard_size": 4,
            "seed_salt": "phase1",
            "tasks": [
                {
                    "task_id": "three_in_bin",
                    "scene": "bin",
                    "instruction": "Put the three blocks in the bin",
                    "success": {"op": "in_container", "object": ["a", "b", "c"], "container": "bin"},
                }
            ],
        }
    )
    assert manifest.tasks[0].success is not None
    assert manifest.tasks[0].success.kind == "in_container"
    assert "kind" not in manifest.tasks[0].success.body


def test_group_step_cap_is_the_eval_config():
    from eval_nav.benchmark.orchestrate import _step_cap

    class Config:
        max_episode_steps = 300

    assert _step_cap(Config()) == 300


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
    assert '"log_version": 2' in payload
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
        "env_scenes:\n"
        "  - {env_id: franka-tabletop-v1, scene: 0}\n",
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
        "seeds: [1]\n"
        "benchmark_env_id: franka-tabletop-v0\n"
        "brain:\n"
        "  socket_dir: /run/brain\n"
        "  step_timeout_s: 60\n"
        "  open_loop_horizon: 8\n",
        encoding="utf-8",
    )
    loaded = EvalConfig.from_yaml(path)
    loaded.validate()
    assert loaded.num_envs is None
    tabletop = EvalConfig.from_yaml(SOURCE / "eval-nav" / "configs" / "task-franka-tabletop.yaml")
    tabletop.validate()
    assert tabletop.num_envs is None
    assert tabletop.num_episodes == 4
    assert tabletop.brain["placement"] == "paired"


def test_plan_slots_pairs_one_brain_per_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    slots = plan_slots("paired")
    assert [(slot.device_id, slot.brain_index) for slot in slots] == [("0", 0), ("1", 1)]


def test_plan_slots_splits_extra_gpus_into_brains(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert [(slot.device_id, slot.brain_index) for slot in plan_slots("split")] == [("0", 0)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    slots = plan_slots("split")
    assert [(slot.device_id, slot.brain_index) for slot in slots] == [("2", 0), ("3", 1)]


def test_lockstep_hashes_match_across_runs():
    shard = make_shards(expand(manifest_from_dict(MANIFEST)), 2)[0]
    first = run_shard(_FakeEnv(len(shard.jobs)), _ZeroRuntime(), shard, open_loop_horizon=2, max_steps=4)
    second = run_shard(_FakeEnv(len(shard.jobs)), _ZeroRuntime(), shard, open_loop_horizon=2, max_steps=4)
    assert [row["trajectory_hash"] for row in first] == [row["trajectory_hash"] for row in second]
    assert all(row["success"] for row in first)


def test_next_brain_call_overlaps_open_loop_steps():
    shard = make_shards(expand(manifest_from_dict(MANIFEST)), 1)[0]
    env = _ClockEnv(len(shard.jobs))
    runtime = _ClockRuntime()
    run_shard(env, runtime, shard, open_loop_horizon=4, max_steps=8)
    overlapped = [
        stamp
        for start, end in runtime.windows[1:]
        for stamp in env.step_times
        if start < stamp < end
    ]
    assert overlapped
    assert not all(env.renders)
    assert any(env.renders)


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


class _ClockRuntime:
    def __init__(self):
        self.windows: list[tuple[float, float]] = []

    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step, seed):
        start = time.perf_counter()
        time.sleep(0.15)
        self.windows.append((start, time.perf_counter()))
        count = next(iter(obs.values())).shape[0]
        return np.zeros((count, 4, 8), dtype=np.float32)

    def end_episode(self, episode_ids):
        return None


class _ClockEnv:
    def __init__(self, count: int):
        self.count = count
        self.render_enabled = True
        self.renders: list[bool] = []
        self.step_times: list[float] = []

    def reset(self, *, seed: int):
        return self.get_obs(), {}

    def get_obs(self):
        return {"joint_pos": np.zeros((self.count, 7), dtype=np.float32)}

    def step(self, action):
        self.renders.append(bool(self.render_enabled))
        self.step_times.append(time.perf_counter())
        done = np.zeros(self.count, dtype=bool)
        return self.get_obs(), None, done, done, {}

    def task_completed(self):
        return np.zeros(self.count, dtype=bool)

    def task_failed(self):
        return np.zeros(self.count, dtype=bool)

    def instructions(self):
        return [""] * self.count

    def get_raw_state(self):
        return np.zeros((self.count, 3), dtype=np.float64)


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
