# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Open-loop lockstep between one vectorized environment and a brain."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np

from ..core.smoothness import sparc_from_positions
from .expand import Shard


class StepEnv(Protocol):
    """The slice of an evaluation environment the lockstep loop uses."""

    def reset(self, *, seed: int) -> Any: ...

    def step(self, action: np.ndarray) -> Any: ...

    def task_completed(self) -> np.ndarray: ...

    def task_failed(self) -> np.ndarray: ...

    def instructions(self) -> list[str]: ...

    def get_raw_state(self) -> np.ndarray: ...

    def get_obs(self) -> dict[str, np.ndarray]: ...


class ActionRuntime(Protocol):
    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None: ...

    def act(self, obs: dict[str, np.ndarray], step: int, seed: int) -> np.ndarray: ...

    def end_episode(self, episode_ids: list[str]) -> None: ...


def run_shard(
    env: StepEnv,
    runtime: ActionRuntime,
    shard: Shard,
    *,
    open_loop_horizon: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Run one shard and return one record per real job.

    Extra padded environments are not used: ``num_envs`` is the job count.
    """
    seed = shard.jobs[0].seed
    env.reset(seed=seed)
    ids = [job.job_id for job in shard.jobs]
    seeds = [job.seed for job in shard.jobs]
    runtime.begin_episode(ids, seeds, env.instructions())
    count = len(shard.jobs)
    frozen = np.zeros(count, dtype=bool)
    success = np.zeros(count, dtype=bool)
    failed = np.zeros(count, dtype=bool)
    steps = np.zeros(count, dtype=int)
    logged: list[np.ndarray] = []
    hold: np.ndarray | None = None
    chunk: np.ndarray | None = None
    chunk_index = 0
    for step in range(max_steps):
        if chunk is None or chunk_index >= open_loop_horizon:
            chunk = np.asarray(runtime.act(env.get_obs(), step, seed), dtype=np.float32)
            chunk_index = 0
        action = np.array(chunk[:, chunk_index], copy=True)
        chunk_index += 1
        if hold is None:
            hold = np.zeros_like(action)
        action[frozen] = hold[frozen]
        hold = action
        env.step(action)
        logged.append(action.copy())
        completed = np.asarray(env.task_completed(), dtype=bool)[:count]
        newly_failed = np.asarray(env.task_failed(), dtype=bool)[:count]
        active = ~frozen
        success[active] = completed[active]
        failed[active] = newly_failed[active]
        steps[active] += 1
        frozen |= completed | newly_failed
        if bool(frozen.all()):
            break
    runtime.end_episode(ids)
    poses = np.asarray(env.get_raw_state())
    control_dt_s = float(getattr(env, "step_dt", 0.04) or 0.04)
    paths = env.hand_positions() if hasattr(env, "hand_positions") else None
    texts = list(env.instructions()) if hasattr(env, "instructions") else []
    records = []
    for index, job in enumerate(shard.jobs):
        done = bool(success[index])
        taken = int(steps[index])
        path = None if paths is None or index >= len(paths) else paths[index]
        records.append(
            {
                "job_id": job.job_id,
                "task_id": job.task_id,
                "scene_id": job.scene_id,
                "variant": job.variant,
                "episode_index": job.episode_index,
                "seed": job.seed,
                "instruction": texts[index] if index < len(texts) else "",
                "success": done,
                "failed": bool(failed[index]),
                "steps": taken,
                "timeout": not bool(frozen[index]),
                "control_dt_s": control_dt_s,
                "elapsed_s": taken * control_dt_s,
                "completion_time_s": taken * control_dt_s if done else None,
                "sparc": None if path is None else sparc_from_positions(path, control_dt_s),
                "final_positions_m": np.round(np.asarray(poses[index], dtype=np.float64), 4).tolist(),
                "trajectory_hash": trajectory_hash([row[index] for row in logged], poses[index]),
            }
        )
    return records


def trajectory_hash(actions: list[np.ndarray], final_pose: np.ndarray) -> str:
    """Hash quantized actions and the final object pose for one environment."""
    digest = hashlib.sha256()
    for action in actions:
        quantized = np.round(np.asarray(action, dtype=np.float64), 6)
        digest.update(quantized.tobytes())
    digest.update(np.round(np.asarray(final_pose, dtype=np.float64), 6).tobytes())
    return digest.hexdigest()

