# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy loading utilities for navigation evaluation."""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import DistillationRunner, OnPolicyRunner


def load_policy_from_checkpoint(checkpoint_path: str, task_name: str, env: gym.Env, workflow: str = "rsl_rl") -> Any:
    """Load policy from checkpoint file using an existing environment.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        task_name: Gymnasium task name
        env: Existing gymnasium environment (must be from the same simulation context)
        workflow: RL framework to use ("rsl_rl" or "skrl"). Defaults to "rsl_rl".
        
    Returns:
        Policy function that takes observations and returns actions.
    """
    # Resolve checkpoint path
    checkpoint_path = retrieve_file_path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if workflow == "skrl":
        return _load_skrl_policy(checkpoint_path, task_name, env)
    else:
        return _load_rsl_rl_policy(checkpoint_path, task_name, env)


def _load_rsl_rl_policy(checkpoint_path: str, task_name: str, env: gym.Env) -> Any:
    """Load RSL-RL policy from checkpoint."""
    # Load agent config from registry
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if not isinstance(agent_cfg, RslRlBaseRunnerCfg):
        raise ValueError(f"Expected RslRlBaseRunnerCfg, got {type(agent_cfg)}")
    
    # Use the provided environment (don't create a new one)
    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Wrap for RSL-RL if not already wrapped
    # Check by looking at the class name or type
    is_wrapped = isinstance(env, RslRlVecEnvWrapper)
    if not is_wrapped:
        # Check if wrapped by traversing the wrapper chain
        current = env
        while hasattr(current, "env"):
            current = current.env
            if isinstance(current, RslRlVecEnvWrapper):
                is_wrapped = True
                break
    
    if not is_wrapped:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Create runner
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    # Load checkpoint
    runner.load(checkpoint_path)
    
    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Extract policy network for potential reset functionality
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    
    # Store policy_nn as attribute for potential reset calls
    # Note: The policy from get_inference_policy should handle observations correctly
    # We create a simple wrapper to ensure compatibility
    def policy_wrapper(obs):
        """Policy wrapper for evaluation."""
        # The policy from get_inference_policy expects TensorDict or dict observations
        # and returns actions as torch tensors
        return policy(obs)
    
    # Attach policy_nn for potential reset functionality
    policy_wrapper.policy_nn = policy_nn
    
    return policy_wrapper


def _load_skrl_policy(checkpoint_path: str, task_name: str, env: gym.Env) -> Any:
    """Load skrl policy from checkpoint."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner
    
    # Load agent config from registry
    experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(f"Expected dict for skrl config, got {type(experiment_cfg)}")
    
    # Use the provided environment (don't create a new one)
    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Wrap for skrl if not already wrapped
    is_wrapped = isinstance(env, SkrlVecEnvWrapper)
    if not is_wrapped:
        current = env
        while hasattr(current, "env"):
            current = current.env
            if isinstance(current, SkrlVecEnvWrapper):
                is_wrapped = True
                break
    
    if not is_wrapped:
        env = SkrlVecEnvWrapper(env, ml_framework="torch")
    
    # Configure runner
    experiment_cfg = experiment_cfg.copy()
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    
    # Load checkpoint
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")
    
    # Create policy wrapper
    def policy_wrapper(obs):
        """Policy wrapper for evaluation."""
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        # Extract actions from skrl output format
        if hasattr(env, "possible_agents"):
            return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
        else:
            return outputs[-1].get("mean_actions", outputs[0])
    
    # Attach agent for potential reset functionality
    policy_wrapper.policy_nn = runner.agent
    
    return policy_wrapper

