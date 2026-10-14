# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

import json
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
from eval_nav.benchmark.scheduler import (
    GROUPS_PER_PROCESS,
    assign_groups,
    batch_groups,
    group_shards,
    plan_slots,
    plan_units,
)
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


def test_scene_rebuilds_stay_under_the_camera_graph_limit():
    groups = list(range(10))
    batches = batch_groups(groups)
    assert GROUPS_PER_PROCESS < 10
    assert [len(batch) for batch in batches] == [GROUPS_PER_PROCESS, 10 - GROUPS_PER_PROCESS]
    assert [item for batch in batches for item in batch] == groups


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


def test_brain_run_directory_matches_checkpoint_layout(tmp_path: Path):
    from datetime import datetime

    from eval_nav.benchmark.orchestrate import run_directory

    created = run_directory(tmp_path, datetime(2026, 10, 7, 19, 30, 0))
    assert created == tmp_path / "eval_run_20261007_193000"
    assert created.is_dir()


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
    write_outputs(
        tmp_path,
        score,
        metrics,
        report=report,
        metadata={"runtime": "brain", "elapsed_seconds": 12.5},
    )
    text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert text.startswith("=" * 60)
    assert "Evaluation Summary" in text
    assert "Final Score:" in text
    assert "Evaluation Metadata:" in text
    assert "Elapsed Time: 12.50 seconds" in text
    assert "Runtime: brain" in text
    assert "\nTasks:\n" not in text
    assert "\nEpisodes:\n" not in text
    assert "place_relative" not in text
    analysis = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert abs(analysis["report"]["tasks"]["place_relative"]["task_score"] - 0.9160) < 1e-4
    assert analysis["report"]["episodes"][0]["completion_time_s"] == 8.0
    payload = json.loads((tmp_path / "evaluation_result.json").read_text(encoding="utf-8"))
    assert payload["log_version"] == 2
    assert "tasks" not in payload
    assert "episodes" not in payload
    assert "episodes" not in payload["metadata"]


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
    assert tabletop.num_episodes == 1
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


# --- scene packing, pull queue, crash resume ---------------------------------------------------


def _scene_file(bundle: Path, task: str, scene: str, **overrides) -> None:
    payload = {
        "scene_id": scene,
        "task_id": task,
        "background": "home_office",
        "surface": "oak",
        "instruction": f"do {task}",
        "placements": [
            {"slot": "banana", "kind": "item", "usd": "banana.usd", "position_m": [0.3, 0.0, 0.0]},
            {"slot": "bowl", "kind": "container", "usd": "bowl.usd", "position_m": [0.4, 0.1, 0.0]},
            {"slot": "mark", "kind": "anchor", "position_m": [0.0, 0.0, 0.0]},
        ],
    }
    payload.update(overrides)
    (bundle / "scenes").mkdir(parents=True, exist_ok=True)
    (bundle / "scenes" / f"{task}-{scene}.json").write_text(json.dumps(payload), encoding="utf-8")


def _packed_manifest(tmp_path: Path, shard_size: int = 4):
    success = {"op": "in_container", "objects": ["banana"], "container": "bowl"}
    tasks = [
        {"task_id": name, "scene": scene, "instruction": f"do {name}", "success": success}
        for name, scene in (("a", "s1"), ("b", "s1"), ("c", "s2"), ("d", "s3"))
    ]
    data = {"version": "v", "shard_size": shard_size, "seed_salt": "x", "pack_by": "assets", "tasks": tasks}
    _scene_file(tmp_path, "a", "s1")
    _scene_file(tmp_path, "b", "s1", instruction="other words")
    _scene_file(tmp_path, "c", "s2", background="billiard_hall")
    _scene_file(tmp_path, "d", "s3")
    return manifest_from_dict(data, bundle_dir=tmp_path)


def test_assets_pack_tasks_that_build_the_same_stage(tmp_path: Path):
    from eval_nav.benchmark.packing import pack_key_for

    manifest = _packed_manifest(tmp_path)
    shards = make_shards(expand(manifest), manifest.shard_size, pack_key_for(manifest))
    groups = group_shards(shards)
    assert sorted(sum(len(shard.jobs) for shard in group) for group in groups) == [1, 3]
    big = max(groups, key=lambda group: len(group[0].jobs))
    assert {job.task_id for job in big[0].jobs} == {"a", "b", "d"}


def test_packing_splits_by_shard_size_and_ignores_gpu_count(tmp_path: Path):
    from eval_nav.benchmark.packing import pack_key_for

    manifest = _packed_manifest(tmp_path, shard_size=2)
    shards = make_shards(expand(manifest), manifest.shard_size, pack_key_for(manifest))
    assert all(len(shard.jobs) <= 2 for shard in shards)
    groups = group_shards(shards)
    assert len(groups) == 2
    expected = sorted(job.job_id for shard in shards for job in shard.jobs)
    for workers in (1, 2, 5):
        units = plan_units(groups, workers)
        assert sorted(_job_ids_of(units)) == expected


def test_pack_by_none_keeps_one_build_per_task(tmp_path: Path):
    from eval_nav.benchmark.packing import pack_key_for

    manifest = _packed_manifest(tmp_path)
    manifest.pack_by = "none"
    assert pack_key_for(manifest) is None
    assert len(group_shards(make_shards(expand(manifest), 4))) == 4


def test_signature_ignores_poses_and_tracks_assets(tmp_path: Path):
    from eval_nav.benchmark.packing import scene_signature

    _scene_file(tmp_path, "a", "s1")
    _scene_file(tmp_path, "b", "s1", instruction="x")
    _scene_file(tmp_path, "c", "s1", surface="maple")
    other = [{"slot": "banana", "kind": "item", "usd": "other.usd", "position_m": [0, 0, 0]}]
    _scene_file(tmp_path, "d", "s1", placements=other)
    moved = [
        {"slot": "banana", "kind": "item", "usd": "banana.usd", "position_m": [0.9, 0.9, 0.0]},
        {"slot": "bowl", "kind": "container", "usd": "bowl.usd", "position_m": [0.1, 0.1, 0.0]},
    ]
    _scene_file(tmp_path, "e", "s1", placements=moved)
    sig = {name: scene_signature(tmp_path, name, "s1") for name in "abcde"}
    assert sig["a"] == sig["b"] == sig["e"]
    assert len({sig["a"], sig["c"], sig["d"]}) == 3


def test_unknown_pack_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        manifest_from_dict({**MANIFEST, "pack_by": "everything"})


def test_v10_bundle_packs_into_fewer_scene_builds():
    import pytest

    from eval_nav.benchmark.packing import pack_key_for

    path = SOURCE / "envhub" / "environments" / "franka-tabletop-v10" / "benchmark.yaml"
    if not path.is_file():
        pytest.skip("franka-tabletop-v10 is not checked out")
    manifest = load_manifest(path)
    jobs = expand(manifest)
    plain = group_shards(make_shards(jobs, 1))
    manifest.pack_by = "assets"
    manifest.shard_size = 8
    packed = group_shards(make_shards(jobs, manifest.shard_size, pack_key_for(manifest)))
    assert len(plain) == len(jobs) == 120
    assert len(packed) < len(plain) / 2
    assert sum(len(shard.jobs) for group in packed for shard in group) == 120


def test_v11_bundle_is_packed_and_self_contained():
    import json

    import pytest

    from eval_nav.benchmark.packing import pack_key_for

    root = SOURCE / "envhub" / "environments" / "franka-tabletop-v11"
    if not (root / "benchmark.yaml").is_file():
        pytest.skip("franka-tabletop-v11 is not checked out")
    manifest = load_manifest(root / "benchmark.yaml")
    assert manifest.version == "franka-tabletop-v11"
    assert manifest.pack_by == "assets"
    jobs = expand(manifest)
    groups = group_shards(make_shards(jobs, manifest.shard_size, pack_key_for(manifest)))
    shards = [shard for group in groups for shard in group]
    assert len(jobs) == 120
    assert len(groups) == 47
    assert max(len(shard.jobs) for shard in shards) <= manifest.shard_size
    assert sum(len(shard.jobs) for shard in shards) == 120
    # Every asset a scene names is inside the bundle, not borrowed from another version.
    missing = set()
    for scene in (root / "scenes").glob("*.json"):
        for placement in json.loads(scene.read_text(encoding="utf-8"))["placements"]:
            if placement.get("usd") and not (root / placement["usd"]).is_file():
                missing.add(placement["usd"])
    assert not missing
    assert not (root / "assets").is_symlink()


def _groups(count: int):
    """One group with one job per task."""
    success = {"op": "in_container", "objects": ["banana"], "container": "bowl"}
    tasks = [
        {"task_id": f"t{index}", "scene": f"s{index}", "instruction": "go", "success": success}
        for index in range(count)
    ]
    manifest = manifest_from_dict({"version": "v", "shard_size": 1, "seed_salt": "x", "tasks": tasks})
    return group_shards(make_shards(expand(manifest), 1))


def _job_ids_of(units) -> list[str]:
    return [job.job_id for unit in units for group in unit for shard in group for job in shard.jobs]


def _record(job_id: str) -> dict:
    return {"job_id": job_id, "task_id": "t", "scene_id": "s", "variant": "nominal"}


def test_plan_units_orders_by_cost_and_bounds_process_size():
    groups = _groups(3)
    mixed = [groups[0], groups[1] + groups[1], groups[2] + groups[2] + groups[2]]
    flat = [group for unit in plan_units(mixed, workers=1) for group in unit]
    assert [sum(len(shard.jobs) for shard in group) for group in flat] == [3, 2, 1]
    many = _groups(100)
    assert max(len(unit) for unit in plan_units(many, workers=1)) == GROUPS_PER_PROCESS
    assert len(plan_units(many, workers=8)) >= 8
    assert plan_units([], workers=4) == []


def test_queue_gives_work_to_whichever_worker_is_idle():
    from eval_nav.benchmark.scheduler import WorkerSlot, run_queue

    units = [[group] for group in _groups(8)]
    seen = {"0": 0, "1": 0}

    def run_unit(slot, unit, attempt):
        seen[slot.device_id] += 1
        time.sleep(0.2 if slot.device_id == "0" else 0.01)
        return [_record(job_id) for job_id in _job_ids_of([unit])]

    records = run_queue([WorkerSlot("0", 0), WorkerSlot("1", 1)], units, run_unit)
    assert sorted(row["job_id"] for row in records) == sorted(_job_ids_of(units))
    assert seen["1"] > seen["0"]


def test_queue_runs_only_unfinished_groups_after_a_crash():
    from eval_nav.benchmark.scheduler import UnitFailed, WorkerSlot, run_queue

    unit = _groups(4)
    calls: list[tuple[int, list[str]]] = []

    def run_unit(slot, groups_in, attempt):
        ids = _job_ids_of([groups_in])
        calls.append((attempt, ids))
        if attempt == 0:
            raise UnitFailed("exit 139", [_record(job_id) for job_id in ids[:2]])
        return [_record(job_id) for job_id in ids]

    records = run_queue([WorkerSlot("0", 0)], [unit], run_unit, retries=1)
    first, second = calls
    assert sorted(row["job_id"] for row in records) == sorted(first[1])
    assert second == (1, first[1][2:])


def test_queue_raises_when_a_unit_keeps_crashing():
    import pytest

    from eval_nav.benchmark.scheduler import UnitFailed, WorkerSlot, run_queue

    def run_unit(slot, unit, attempt):
        raise UnitFailed("exit 139", [])

    with pytest.raises(UnitFailed):
        run_queue([WorkerSlot("0", 0)], [_groups(2)], run_unit, retries=1)


def test_partial_worker_output_is_kept_and_a_torn_line_is_dropped(tmp_path: Path):
    from eval_nav.benchmark.orchestrate import _read_records

    path = tmp_path / "gpu0-000.jsonl"
    path.write_text(json.dumps(_record("a")) + "\n" + json.dumps(_record("b")) + '\n{"job_id": "c', encoding="utf-8")
    assert [row["job_id"] for row in _read_records(path)] == ["a", "b"]
    assert _read_records(tmp_path / "missing.jsonl") == []


def test_lockstep_records_where_wall_time_went():
    from eval_nav.benchmark.orchestrate import wall_summary

    shard = make_shards(expand(manifest_from_dict(MANIFEST)), 2)[0]
    rows = run_shard(_FakeEnv(len(shard.jobs)), _ZeroRuntime(), shard, open_loop_horizon=2, max_steps=4)
    assert len({json.dumps(row["wall"], sort_keys=True) for row in rows}) == 1
    assert set(rows[0]["wall"]) >= {"reset_s", "obs_s", "brain_wait_s", "step_s", "envs", "shard"}
    assert "over 1 shard runs" in wall_summary(rows)
