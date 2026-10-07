# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Open-loop lockstep between one vectorized environment and a brain."""

from __future__ import annotations

import hashlib
import threading
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
    Cameras render on the step whose image the next brain call will read.
    Once a chunk is half consumed, the next call runs while the rest of the
    chunk is stepped. That call sees the mid-chunk observation.
    """
    seed = shard.jobs[0].seed
    _set_render(env, True)
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
    pending: _PendingAct | None = None
    prefetch = open_loop_horizon >= 2
    stagger = max(1, open_loop_horizon // 2)
    try:
        for step in range(max_steps):
            if chunk is None or chunk_index >= open_loop_horizon:
                if pending is not None:
                    chunk = pending.join()
                    pending = None
                else:
                    _set_render(env, True)
                    chunk = np.asarray(runtime.act(env.get_obs(), step, seed), dtype=np.float32)
                chunk_index = 0
            action = np.array(chunk[:, chunk_index], copy=True)
            chunk_index += 1
            if hold is None:
                hold = np.zeros_like(action)
            action[frozen] = hold[frozen]
            hold = action
            actions_left_after = open_loop_horizon - chunk_index
            steps_left_after = max_steps - step - 1
            capture = (
                prefetch
                and pending is None
                and chunk_index == stagger
                and steps_left_after > actions_left_after
                and not bool(frozen.all())
            )
            _set_render(env, capture or (chunk_index >= open_loop_horizon and pending is None))
            env.step(action)
            logged.append(action.copy())
            completed = np.asarray(env.task_completed(), dtype=bool)[:count]
            newly_failed = np.asarray(env.task_failed(), dtype=bool)[:count]
            active = ~frozen
            success[active] = completed[active]
            failed[active] = newly_failed[active]
            steps[active] += 1
            frozen |= completed | newly_failed
            if capture and not bool(frozen.all()):
                pending = _PendingAct(runtime, _snapshot_obs(env.get_obs()), step + 1, seed)
                pending.start()
            if bool(frozen.all()):
                break
    finally:
        # A prefetch that the episode never consumes must not outlive the socket.
        if pending is not None and not pending.joined:
            try:
                pending.join()
            except Exception:
                pass
    runtime.end_episode(ids)
    poses = np.asarray(env.get_raw_state())
    control_dt_s = float(getattr(env, "step_dt", 0.04) or 0.04)
    paths = env.hand_positions() if hasattr(env, "hand_positions") else None
    texts = list(env.instructions()) if hasattr(env, "instructions") else []
    progress = np.asarray(env.task_progress(), dtype=np.float64) if hasattr(env, "task_progress") else None
    records = []
    for index, job in enumerate(shard.jobs):
        done = bool(success[index])
        taken = int(steps[index])
        path = None if paths is None or index >= len(paths) else paths[index]
        elapsed = taken * control_dt_s
        length = _path_length(path)
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
                "elapsed_s": elapsed,
                "completion_time_s": elapsed if done else None,
                "progress": 1.0 if done and progress is None else (0.0 if progress is None else float(progress[index])),
                "path_length_m": length,
                "mean_hand_speed_mps": length / elapsed if elapsed > 0.0 else 0.0,
                "sparc": None if path is None else sparc_from_positions(path, control_dt_s),
                "final_positions_m": np.round(np.asarray(poses[index], dtype=np.float64), 4).tolist(),
                "trajectory_hash": trajectory_hash([row[index] for row in logged], poses[index]),
            }
        )
    return records


class _PendingAct:
    """One brain call running while the current chunk is still being stepped."""

    def __init__(self, runtime: ActionRuntime, obs: dict[str, np.ndarray], step: int, seed: int):
        self._runtime = runtime
        self._obs = obs
        self._step = step
        self._seed = seed
        self._result: np.ndarray | None = None
        self._error: BaseException | None = None
        self.joined = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> np.ndarray:
        if not self.joined:
            self._thread.join()
            self.joined = True
        if self._error is not None:
            raise self._error
        return self._result

    def _run(self) -> None:
        try:
            self._result = np.asarray(self._runtime.act(self._obs, self._step, self._seed), dtype=np.float32)
        except BaseException as exc:
            self._error = exc


def _snapshot_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def _set_render(env: StepEnv, enabled: bool) -> None:
    """Turn RTX cameras on only for the step whose image will be sent to the brain."""
    if hasattr(env, "render_enabled"):
        env.render_enabled = enabled


def _path_length(path: np.ndarray | None) -> float:
    """Length [m] of a hand path. An empty path has length 0."""
    if path is None or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(path, dtype=np.float64), axis=0), axis=1).sum())


def trajectory_hash(actions: list[np.ndarray], final_pose: np.ndarray) -> str:
    """Hash quantized actions and the final object pose for one environment."""
    digest = hashlib.sha256()
    for action in actions:
        quantized = np.round(np.asarray(action, dtype=np.float64), 6)
        digest.update(quantized.tobytes())
    digest.update(np.round(np.asarray(final_pose, dtype=np.float64), 6).tobytes())
    return digest.hexdigest()

