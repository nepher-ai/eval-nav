# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode execution utilities for navigation evaluation.

This module handles running individual episodes, supporting both
single and vectorized environments.
"""

from __future__ import annotations

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
    
    def __init__(self, config: Any):
        """Initialize episode runner.
        
        Args:
            config: Evaluation configuration (EvalConfig).
        """
        self.config = config
        
        # Initialize state logger if enabled
        self.state_logger = None
        if config.enable_logging and config.log_dir:
            self.state_logger = StateLogger(log_dir=config.log_dir, enabled=True)
    
    def run_episode(
        self,
        env: gym.Env,
        policy: Any | None,
        scene: str | int,
        nav_env_id: str,
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
            # Reset environment with seed
            obs, info = env.reset(seed=seed)
            
            max_steps = self.config.max_episode_steps or getattr(env.unwrapped, "max_episode_length", 900)

            # Detect if environment is vectorized
            num_envs = self._detect_num_envs(env, obs)
            is_vectorized = num_envs > 1
            
            # Initialize state logging
            if self.state_logger is not None:
                if is_vectorized:
                    for env_idx in range(num_envs):
                        self.state_logger.reset(episode_id=episode_id + env_idx, env_idx=env_idx, env=env)
                else:
                    self.state_logger.reset(episode_id=episode_id, env_idx=None, env=env)
            
            # Track episode state for each environment
            if is_vectorized:
                # Track state per environment
                steps_per_env = [0] * num_envs
                done_per_env = [False] * num_envs
                success_per_env = [False] * num_envs
                timeout_per_env = [False] * num_envs
                completion_time_per_env: list[float | None] = [None] * num_envs
            else:
                # Single environment tracking
                steps = 0
                done = False
                success = False
                timeout = False
                # Initialize done_per_env for non-vectorized case (not used, but needed for _get_action signature)
                done_per_env = []
            
            # Run episode until all environments are done or max steps reached
            all_done = False
            steps = 0
            
            while not all_done and steps < max_steps:
                # Get action
                action = self._get_action(env, obs, policy, is_vectorized, num_envs, done_per_env)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                # Log state for this step
                if self.state_logger is not None:
                    if is_vectorized:
                        for env_idx in range(num_envs):
                            if not done_per_env[env_idx]:  # Only log for active environments
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
                
                # Handle vectorized environment
                if is_vectorized:
                    all_done = self._update_vectorized_state(
                        env, info, terminated, truncated, steps_per_env, done_per_env,
                        success_per_env, timeout_per_env, num_envs, steps, max_steps
                    )
                else:
                    done, success, timeout, all_done = self._update_single_state(
                        env, info, terminated, truncated, steps, max_steps, success, timeout
                    )
            
            # Save state logs before finalizing metrics
            if self.state_logger is not None:
                if is_vectorized:
                    for env_idx in range(num_envs):
                        self.state_logger.save(
                            episode_id=episode_id + env_idx,
                            scene=scene,
                            seed=seed,
                            env_idx=env_idx,
                            nav_env_id=nav_env_id,
                        )
                else:
                    self.state_logger.save(
                        episode_id=episode_id,
                        scene=scene,
                        seed=seed,
                        env_idx=None,
                        nav_env_id=nav_env_id,
                    )
            
            # Finalize metrics
            if is_vectorized:
                return self._finalize_vectorized_metrics(
                    env, info, scene, nav_env_id, seed, episode_id, steps_per_env,
                    success_per_env, timeout_per_env, completion_time_per_env, num_envs
                )
            else:
                return self._finalize_single_metrics(
                    env, info, scene, nav_env_id, seed, episode_id, steps, success, timeout
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
        # Detect if environment is vectorized by checking num_envs property
        # IsaacLab environments expose num_envs via unwrapped.num_envs or unwrapped.scene.num_envs
        unwrapped = getattr(env, "unwrapped", None)
        num_envs = (getattr(unwrapped, "num_envs", None) 
                   or getattr(getattr(unwrapped, "scene", None), "num_envs", None) if unwrapped else None)
        
        # Fallback: check observation shape
        if num_envs is None:
            obs_tensor = next(iter(obs.values()), obs) if isinstance(obs, dict) else obs
            num_envs = obs_tensor.shape[0] if torch.is_tensor(obs_tensor) and obs_tensor.ndim > 0 else 1
        
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
            # Policy inference
            if isinstance(obs, dict):
                # Handle dict observations
                action = policy(obs)
            else:
                action = policy(obs)
        else:
            # Random action - convert numpy to torch tensor
            action_np = env.action_space.sample()
            # Convert to torch tensor and move to device
            device = getattr(env.unwrapped, "device", "cpu")
            if isinstance(action_np, np.ndarray):
                action = torch.from_numpy(action_np).to(device=device, dtype=torch.float32)
            elif isinstance(action_np, dict):
                action = {k: torch.from_numpy(v).to(device=device, dtype=torch.float32) for k, v in action_np.items()}
            else:
                # Already a tensor or other type
                action = action_np
        
        # Mask actions for done environments (stop them from acting)
        # This prevents done environments from continuing to take actions
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
            # Mask each action component
            masked_action = {}
            for k, v in action.items():
                if torch.is_tensor(v) and len(v.shape) > 0 and v.shape[0] == num_envs:
                    # Clone and mask done environments by zeroing their actions
                    masked_v = v.clone()
                    for env_idx in range(num_envs):
                        if done_per_env[env_idx]:
                            masked_v[env_idx] = 0.0
                    masked_action[k] = masked_v
                else:
                    masked_action[k] = v
            return masked_action
        elif torch.is_tensor(action) and len(action.shape) > 0 and action.shape[0] == num_envs:
            # Mask tensor action by zeroing actions for done environments
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
        # Update state for each environment
        for env_idx in range(num_envs):
            if not done_per_env[env_idx]:
                steps_per_env[env_idx] = steps
                # Check if terminated/truncated are tensors before indexing
                if torch.is_tensor(terminated) and torch.is_tensor(truncated):
                    done_per_env[env_idx] = bool(terminated[env_idx].item() or truncated[env_idx].item())
                else:
                    # Fallback for non-tensor case
                    done_per_env[env_idx] = bool(terminated or truncated)
                
                # Check for success using task-aware function
                success_per_env[env_idx] = check_success(
                    env=env,
                    info=info,
                    task_name=self.config.task_name,
                    env_idx=env_idx,
                    current_success=success_per_env[env_idx],
                )
                
                # Check timeout
                if steps >= max_steps:
                    timeout_per_env[env_idx] = True
        
        # Check if all environments are done
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
        
        # Check for success using task-aware function
        success = check_success(
            env=env,
            info=info,
            task_name=self.config.task_name,
            env_idx=None,
            current_success=success,
        )
        
        # Check timeout
        if steps >= max_steps:
            timeout = True
        
        all_done = done
        
        return done, success, timeout, all_done
    
    def _finalize_vectorized_metrics(
        self,
        env: gym.Env,
        info: dict[str, Any],
        scene: str | int,
        nav_env_id: str,
        seed: int,
        episode_id: int,
        steps_per_env: list[int],
        success_per_env: list[bool],
        timeout_per_env: list[bool],
        completion_time_per_env: list[float | None],
        num_envs: int,
    ) -> list[EpisodeMetrics]:
        """Finalize metrics for vectorized environment.
        
        Args:
            env: Gymnasium environment.
            info: Info dictionary from step.
            scene: Scene ID.
            nav_env_id: Navigation environment ID.
            seed: Random seed.
            episode_id: Base episode ID.
            steps_per_env: List of step counts per environment.
            success_per_env: List of success flags per environment.
            timeout_per_env: List of timeout flags per environment.
            completion_time_per_env: List of completion times per environment.
            num_envs: Number of environments.
            
        Returns:
            List of EpisodeMetrics for all environments.
        """
        # Always check final success status for each environment
        # This ensures we capture success even if it wasn't detected during the loop
        for env_idx in range(num_envs):
            # Final success check using task-aware function
            success_per_env[env_idx] = check_success(
                env=env,
                info=info,
                task_name=self.config.task_name,
                env_idx=env_idx,
                current_success=success_per_env[env_idx],
            )
            
            # Compute completion time (for successful episodes)
            if success_per_env[env_idx] and not timeout_per_env[env_idx]:
                completion_time_per_env[env_idx] = float(steps_per_env[env_idx])
        
        # Return list of EpisodeMetrics for all environments
        return [
            EpisodeMetrics(
                episode_id=episode_id + env_idx,
                scene=scene,
                seed=seed,
                success=success_per_env[env_idx],
                steps=steps_per_env[env_idx],
                timeout=timeout_per_env[env_idx],
                nav_env_id=nav_env_id,
                completion_time=completion_time_per_env[env_idx],
            )
            for env_idx in range(num_envs)
        ]
    
    def _finalize_single_metrics(
        self,
        env: gym.Env,
        info: dict[str, Any],
        scene: str | int,
        nav_env_id: str,
        seed: int,
        episode_id: int,
        steps: int,
        success: bool,
        timeout: bool,
    ) -> EpisodeMetrics:
        """Finalize metrics for single environment.
        
        Args:
            env: Gymnasium environment.
            info: Info dictionary from step.
            scene: Scene ID.
            nav_env_id: Navigation environment ID.
            seed: Random seed.
            episode_id: Episode ID.
            steps: Step count.
            success: Success flag.
            timeout: Timeout flag.
            
        Returns:
            EpisodeMetrics instance.
        """
        # Always check final success status (episode might have succeeded)
        # This ensures we capture success even if it wasn't detected during the loop
        success = check_success(
            env=env,
            info=info,
            task_name=self.config.task_name,
            env_idx=None,
            current_success=success,
        )
        
        # Compute completion time (for successful episodes)
        completion_time = None
        if success and not timeout:
            # Use steps as proxy for time (assuming fixed dt)
            # In practice, you might want to track actual wall-clock time
            completion_time = float(steps)
        
        return EpisodeMetrics(
            episode_id=episode_id,
            scene=scene,
            seed=seed,
            success=success,
            steps=steps,
            timeout=timeout,
            nav_env_id=nav_env_id,
            completion_time=completion_time,
        )

