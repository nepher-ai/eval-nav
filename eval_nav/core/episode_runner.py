# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Episode execution utilities for navigation evaluation.

This module handles running individual episodes, supporting both
single and vectorized environments.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from ..domain.errors import EvaluationRuntimeError
from ..domain.metrics import EpisodeMetrics
from ..utils.state_logger import StateLogger
from ..utils.task_checker import check_success


class EpisodeRunner:
    """Runs episodes for navigation evaluation."""
    
    def __init__(self, config: Any):  # type: ignore[type-arg]
        """Initialize episode runner.
        
        Args:
            config: Evaluation configuration (EvalConfig).
        """
        self.config = config
        
        # Initialize state logger if enabled
        self.state_logger = None
        if config.enable_logging and config.log_dir:
            self.state_logger = StateLogger(log_dir=config.log_dir, enabled=True)
    
    # ------------------------------------------------------------------
    # Locomotion data collection (populated only when env exposes it)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_locomotion(env: gym.Env) -> bool:
        return callable(getattr(env, "get_locomotion_data", None))

    @staticmethod
    def _collect_locomotion_step(
        env: gym.Env,
        buffers: dict[int, dict[str, list[float]]],
        env_idx: int,
        done: bool,
    ) -> None:
        """Append one locomotion sample for *env_idx* unless it is already done."""
        if done:
            return
        data = env.get_locomotion_data(env_idx=env_idx)  # type: ignore[attr-defined]
        if data is None:
            return
        buf = buffers[env_idx]
        for k, v in data.items():
            buf[k].append(v)

    @staticmethod
    def _summarise_locomotion(buf: dict[str, list[float]]) -> dict[str, Any]:
        """Reduce a per-step locomotion buffer into episode-level stats."""
        if not buf or not buf.get("speed_2d"):
            return {}
        speeds = np.asarray(buf["speed_2d"])
        summary: dict[str, Any] = {
            "mean_speed": float(speeds.mean()),
            "max_speed": float(speeds.max()),
            "speed_std": float(speeds.std()),
            "mean_vertical_speed": float(np.mean(buf["vertical_speed"])),
            "mean_angular_speed": float(np.mean(buf["yaw_rate"])),
            "angular_speed_std": float(np.std(buf["yaw_rate"])),
            "mean_roll_pitch_rate": float(np.mean(buf["roll_pitch_rate"])),
        }
        if "lateral_speed" in buf:
            lat = np.asarray(buf["lateral_speed"])
            summary["mean_lateral_speed"] = float(lat.mean())
            summary["max_lateral_speed"] = float(lat.max())
        if "feet_in_contact" in buf:
            feet = np.asarray(buf["feet_in_contact"])
            summary["mean_feet_in_contact"] = float(feet.mean())
            summary["aerial_phase_fraction"] = float((feet == 0).mean())
        if "terrain_slope" in buf:
            slopes = np.asarray(buf["terrain_slope"])
            summary["mean_slope_deg"] = float(np.degrees(slopes.mean()))
            summary["max_slope_deg"] = float(np.degrees(slopes.max()))
        return summary
    
    def run_episode(
        self,
        env: gym.Env,
        policy: Any | None,
        scene: str | int,
        env_id: str,
        seed: int,
        episode_id: int,
    ) -> EpisodeMetrics | list[EpisodeMetrics]:
        """Run a single episode (or all episodes in a vectorized environment).
        
        Args:
            env: Gymnasium environment.
            policy: Policy to evaluate (None for random).
            scene: Scene ID.
            seed: Random seed.
            episode_id: Episode identifier (base ID, will be incremented for each env in vectorized case).
            
        Returns:
            EpisodeMetrics instance for single environment, or list of EpisodeMetrics for vectorized environments.
            
        Raises:
            EvaluationRuntimeError: If episode execution fails.
        """
        try:
            obs, info = env.reset(seed=seed)
            max_steps = self.config.max_episode_steps or getattr(env.unwrapped, "max_episode_length", 900)
            step_dt: float | None = getattr(env.unwrapped, "step_dt", None)

            num_envs = self._detect_num_envs(env, obs)
            is_vectorized = num_envs > 1
            
            if self.state_logger is not None:
                if is_vectorized:
                    for env_idx in range(num_envs):
                        self.state_logger.reset(episode_id=episode_id + env_idx, env_idx=env_idx, env=env)
                else:
                    self.state_logger.reset(episode_id=episode_id, env_idx=None, env=env)
            
            if is_vectorized:
                steps_per_env = [0] * num_envs
                done_per_env = [False] * num_envs
                success_per_env = [False] * num_envs
                timeout_per_env = [False] * num_envs
                completion_time_per_env: list[float | None] = [None] * num_envs
            else:
                steps = 0
                done = False
                success = False
                timeout = False
                done_per_env = []
            
            collect_loco = self._has_locomotion(env)
            loco_buffers: dict[int, dict[str, list[float]]] = (
                defaultdict(lambda: defaultdict(list)) if collect_loco else {}
            )
            
            all_done = False
            steps = 0
            
            while not all_done and steps < max_steps:
                action = self._get_action(env, obs, policy, is_vectorized, num_envs, done_per_env)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if self.state_logger is not None:
                    if is_vectorized:
                        for env_idx in range(num_envs):
                            if not done_per_env[env_idx]:
                                self.state_logger.log_step(
                                    env=env,
                                    episode_id=episode_id + env_idx,
                                    step=steps,
                                    env_idx=env_idx,
                                    info=info,
                                )
                    else:
                        self.state_logger.log_step(
                            env=env,
                            episode_id=episode_id,
                            step=steps,
                            env_idx=None,
                            info=info,
                        )
                
                if collect_loco:
                    if is_vectorized:
                        for env_idx in range(num_envs):
                            self._collect_locomotion_step(env, loco_buffers, env_idx, done_per_env[env_idx])
                    else:
                        self._collect_locomotion_step(env, loco_buffers, 0, False)
                
                if is_vectorized:
                    all_done = self._update_vectorized_state(
                        env, info, terminated, truncated, steps_per_env, done_per_env,
                        success_per_env, timeout_per_env, num_envs, steps, max_steps
                    )
                else:
                    done, success, timeout, all_done = self._update_single_state(
                        env, info, terminated, truncated, steps, max_steps, success, timeout
                    )
            
            if self.state_logger is not None:
                if is_vectorized:
                    for env_idx in range(num_envs):
                        self.state_logger.save(
                            episode_id=episode_id + env_idx,
                            scene=scene,
                            seed=seed,
                            env_idx=env_idx,
                            env_id=env_id,
                        )
                else:
                    self.state_logger.save(
                        episode_id=episode_id,
                        scene=scene,
                        seed=seed,
                        env_idx=None,
                        env_id=env_id,
                    )
            
            if is_vectorized:
                return self._finalize_vectorized_metrics(
                    env, info, scene, env_id, seed, episode_id, steps_per_env,
                    success_per_env, timeout_per_env, completion_time_per_env, num_envs,
                    loco_buffers, step_dt=step_dt,
                )
            else:
                return self._finalize_single_metrics(
                    env, info, scene, env_id, seed, episode_id, steps, success, timeout,
                    loco_buffers.get(0, {}), step_dt=step_dt,
                )
            
        except Exception as e:
            raise EvaluationRuntimeError(
                f"Episode {episode_id} failed: {str(e)}",
                details={
                    "episode_id": episode_id,
                    "scene": scene,
                    "seed": seed,
                    "error_type": type(e).__name__,
                },
            ) from e
    
    def _detect_num_envs(self, env: gym.Env, obs: Any) -> int:
        """Detect number of environments (for vectorized envs).
        
        Args:
            env: Gymnasium environment.
            obs: Initial observation.
            
        Returns:
            Number of environments (1 for single env).
        """
        unwrapped = getattr(env, "unwrapped", None)
        num_envs = None
        if unwrapped:
            num_envs = getattr(unwrapped, "num_envs", None)
            if num_envs is None:
                scene = getattr(unwrapped, "scene", None)
                if scene is not None:
                    num_envs = getattr(scene, "num_envs", None)
        
        if num_envs is None:
            if isinstance(obs, dict):
                obs_tensor = next(iter(obs.values()), obs)
            else:
                obs_tensor = obs
            if torch.is_tensor(obs_tensor) and obs_tensor.ndim > 0:
                num_envs = obs_tensor.shape[0]
            else:
                num_envs = 1
        
        return num_envs
    
    def _get_action(
        self,
        env: gym.Env,
        obs: Any,
        policy: Any | None,
        is_vectorized: bool,
        num_envs: int,
        done_per_env: list[bool],
    ) -> Any:
        """Get action from policy or random.
        
        Args:
            env: Gymnasium environment.
            obs: Current observation.
            policy: Policy to evaluate (None for random).
            is_vectorized: Whether environment is vectorized.
            num_envs: Number of environments.
            done_per_env: List of done flags for each environment.
            
        Returns:
            Action tensor or dict.
        """
        if policy is not None:
            if isinstance(obs, dict):
                action = policy(obs)
            else:
                action = policy(obs)
        else:
            action_np = env.action_space.sample()
            device = getattr(env.unwrapped, "device", "cpu")
            if isinstance(action_np, np.ndarray):
                action = torch.from_numpy(action_np).to(device=device, dtype=torch.float32)
            elif isinstance(action_np, dict):
                action = {k: torch.from_numpy(v).to(device=device, dtype=torch.float32) for k, v in action_np.items()}
            else:
                action = action_np
        
        if is_vectorized:
            action = self._mask_done_actions(action, num_envs, done_per_env)
        
        return action
    
    def _mask_done_actions(self, action: Any, num_envs: int, done_per_env: list[bool]) -> Any:
        """Mask actions for done environments.
        
        Args:
            action: Action tensor or dict.
            num_envs: Number of environments.
            done_per_env: List of done flags for each environment.
            
        Returns:
            Masked action.
        """
        if isinstance(action, dict):
            masked_action = {}
            for k, v in action.items():
                if torch.is_tensor(v) and len(v.shape) > 0 and v.shape[0] == num_envs:
                    masked_v = v.clone()
                    for env_idx in range(num_envs):
                        if done_per_env[env_idx]:
                            masked_v[env_idx] = 0.0
                    masked_action[k] = masked_v
                else:
                    masked_action[k] = v
            return masked_action
        elif torch.is_tensor(action) and len(action.shape) > 0 and action.shape[0] == num_envs:
            masked_action = action.clone()
            for env_idx in range(num_envs):
                if done_per_env[env_idx]:
                    masked_action[env_idx] = 0.0
            return masked_action
        else:
            return action
    
    def _update_vectorized_state(
        self,
        env: gym.Env,
        info: dict[str, Any],
        terminated: Any,
        truncated: Any,
        steps_per_env: list[int],
        done_per_env: list[bool],
        success_per_env: list[bool],
        timeout_per_env: list[bool],
        num_envs: int,
        steps: int,
        max_steps: int,
    ) -> bool:
        """Update state for vectorized environment.
        
        Args:
            env: Gymnasium environment.
            info: Info dictionary from step.
            terminated: Termination flags.
            truncated: Truncation flags.
            steps_per_env: List of step counts per environment.
            done_per_env: List of done flags per environment.
            success_per_env: List of success flags per environment.
            timeout_per_env: List of timeout flags per environment.
            num_envs: Number of environments.
            steps: Current global step count.
            max_steps: Maximum steps per episode.
            
        Returns:
            Whether all environments are done.
        """
        for env_idx in range(num_envs):
            if not done_per_env[env_idx]:
                steps_per_env[env_idx] = steps
                if torch.is_tensor(terminated) and torch.is_tensor(truncated):
                    done_per_env[env_idx] = bool(terminated[env_idx].item() or truncated[env_idx].item())
                else:
                    done_per_env[env_idx] = bool(terminated or truncated)
                
                success_per_env[env_idx] = check_success(
                    env=env,
                    info=info,
                    task_name=self.config.task_name,
                    env_idx=env_idx,
                    current_success=success_per_env[env_idx],
                )
                
                if steps >= max_steps:
                    timeout_per_env[env_idx] = True
        
        return all(done_per_env)
    
    def _update_single_state(
        self,
        env: gym.Env,
        info: dict[str, Any],
        terminated: Any,
        truncated: Any,
        steps: int,
        max_steps: int,
        success: bool,
        timeout: bool,
    ) -> tuple[bool, bool, bool, bool]:
        """Update state for single environment.
        
        Args:
            env: Gymnasium environment.
            info: Info dictionary from step.
            terminated: Termination flag.
            truncated: Truncation flag.
            steps: Current step count.
            max_steps: Maximum steps per episode.
            success: Current success status.
            timeout: Current timeout status.
            
        Returns:
            Tuple of (done, success, timeout, all_done).
        """
        if torch.is_tensor(terminated) and torch.is_tensor(truncated):
            done = bool(terminated.item() or truncated.item())
        else:
            done = bool(terminated or truncated)
        
        success = check_success(
            env=env,
            info=info,
            task_name=self.config.task_name,
            env_idx=None,
            current_success=success,
        )
        
        if steps >= max_steps:
            timeout = True
        
        all_done = done
        return done, success, timeout, all_done
    
    def _finalize_vectorized_metrics(
        self,
        env: gym.Env,
        info: dict[str, Any],
        scene: str | int,
        env_id: str,
        seed: int,
        episode_id: int,
        steps_per_env: list[int],
        success_per_env: list[bool],
        timeout_per_env: list[bool],
        completion_time_per_env: list[float | None],
        num_envs: int,
        loco_buffers: dict[int, dict[str, list[float]]],
        step_dt: float | None = None,
    ) -> list[EpisodeMetrics]:
        """Finalize metrics for vectorized environment."""
        for env_idx in range(num_envs):
            success_per_env[env_idx] = check_success(
                env=env,
                info=info,
                task_name=self.config.task_name,
                env_idx=env_idx,
                current_success=success_per_env[env_idx],
            )
            
            if success_per_env[env_idx] and not timeout_per_env[env_idx]:
                raw_steps = float(steps_per_env[env_idx])
                completion_time_per_env[env_idx] = raw_steps * step_dt if step_dt else raw_steps
        
        results: list[EpisodeMetrics] = []
        for env_idx in range(num_envs):
            extra = self._summarise_locomotion(loco_buffers.get(env_idx, {}))
            if step_dt is not None:
                extra["step_dt"] = step_dt
            results.append(EpisodeMetrics(
                episode_id=episode_id + env_idx,
                scene=scene,
                seed=seed,
                success=success_per_env[env_idx],
                steps=steps_per_env[env_idx],
                timeout=timeout_per_env[env_idx],
                env_id=env_id,
                completion_time=completion_time_per_env[env_idx],
                extra=extra,
            ))
        return results
    
    def _finalize_single_metrics(
        self,
        env: gym.Env,
        info: dict[str, Any],
        scene: str | int,
        env_id: str,
        seed: int,
        episode_id: int,
        steps: int,
        success: bool,
        timeout: bool,
        loco_buf: dict[str, list[float]] | None = None,
        step_dt: float | None = None,
    ) -> EpisodeMetrics:
        """Finalize metrics for single environment."""
        success = check_success(
            env=env,
            info=info,
            task_name=self.config.task_name,
            env_idx=None,
            current_success=success,
        )
        
        completion_time = None
        if success and not timeout:
            raw_steps = float(steps)
            completion_time = raw_steps * step_dt if step_dt else raw_steps
        
        extra = self._summarise_locomotion(loco_buf) if loco_buf else {}
        if step_dt is not None:
            extra["step_dt"] = step_dt
        
        return EpisodeMetrics(
            episode_id=episode_id,
            scene=scene,
            seed=seed,
            success=success,
            steps=steps,
            timeout=timeout,
            env_id=env_id,
            completion_time=completion_time,
            extra=extra,
        )

