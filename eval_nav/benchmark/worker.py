# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Isaac process for one group of episodes. Started by ``evaluate.py --worker``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from eval_nav.benchmark.expand import EpisodeJob, Shard


def main() -> None:
    """Start Isaac Sim and run the episodes in ``--group``."""
    from isaaclab.app import AppLauncher

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--group", required=True)
    pre_args, _ = pre.parse_known_args()
    group = json.loads(Path(pre_args.group).read_text(encoding="utf-8"))
    if group.get("device") and "--device" not in sys.argv:
        sys.argv.extend(["--device", group["device"]])

    parser = argparse.ArgumentParser(description="Run one evaluation group")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    # Lab 3 dropped the --enable_cameras flag. The launcher still reads the attribute.
    if group.get("enable_cameras"):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    from eval_nav.benchmark.lockstep import run_shard
    from eval_nav.runtime.brain import BrainRuntime
    from eval_nav.utils.determinism import apply_determinism

    module_name, class_name = group["cfg_entry"].split(":")
    cfg_cls = getattr(importlib.import_module(module_name), class_name)
    if group.get("task_module"):
        importlib.import_module(group["task_module"])
    first = group["shards"][0]
    num_envs = max(len(shard["jobs"]) for shard in group["shards"])
    cfg = cfg_cls(
        task_id=first["task_id"],
        scene=first["scene_id"],
        variant=first["variant"],
        composer=first["composer"],
        pose_jitter_m=first["pose_jitter_m"],
        env_id=group.get("benchmark_env_id"),
        num_envs=num_envs,
    )
    env = gym.make(group["task_name"], cfg=cfg)
    unwrapped = env.unwrapped
    client = None
    if group.get("runtime") == "brain":
        from nepher_brain_comm.client import BrainClient

        client = BrainClient(group["socket"], timeout_s=float(group["step_timeout_s"]))
        client.connect()
        runtime = BrainRuntime(client)
    else:
        from eval_nav.runtime.in_process import InProcessRuntime
        from eval_nav.utils.policy_loader import load_policy_from_checkpoint

        policy = load_policy_from_checkpoint(group["policy_path"], group["task_name"], env)
        runtime = InProcessRuntime(policy)
    lines = []
    try:
        for shard_payload in group["shards"]:
            apply_determinism(int(shard_payload["jobs"][0]["seed"]))
            shard = _shard(shard_payload)
            _prepare_jobs(unwrapped, shard)
            records = run_shard(
                unwrapped,
                runtime,
                shard,
                open_loop_horizon=int(group["open_loop_horizon"]),
                max_steps=int(group["max_steps"]),
            )
            lines.extend(json.dumps(record) for record in records)
    finally:
        if client is not None:
            client.close()
        env.close()
    Path(args_cli.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    simulation_app.close()


def _prepare_jobs(env, shard: Shard) -> None:
    if hasattr(env, "set_episode_jobs"):
        env.set_episode_jobs(shard.jobs)


def _shard(payload: dict) -> Shard:
    jobs = tuple(EpisodeJob(**job) for job in payload["jobs"])
    return Shard(
        task_id=payload["task_id"],
        scene_id=payload["scene_id"],
        composer=payload["composer"],
        variant=payload["variant"],
        pose_jitter_m=payload["pose_jitter_m"],
        jobs=jobs,
    )
