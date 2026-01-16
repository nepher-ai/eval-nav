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
    return _load_rsl_rl_policy(checkpoint_path, task_name, env)


def _load_rsl_rl_policy(checkpoint_path: str, task_name: str, env: gym.Env) -> Any:
    """Load RSL-RL policy from checkpoint."""
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if not isinstance(agent_cfg, RslRlBaseRunnerCfg):
        raise ValueError(f"Expected RslRlBaseRunnerCfg, got {type(agent_cfg)}")
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    is_wrapped = isinstance(env, RslRlVecEnvWrapper)
    if not is_wrapped:
        current = env
        while hasattr(current, "env"):
            current = current.env
            if isinstance(current, RslRlVecEnvWrapper):
                is_wrapped = True
                break
    
    if not is_wrapped:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    
    def policy_wrapper(obs):
        """Policy wrapper for evaluation."""
        return policy(obs)
    
    policy_wrapper.policy_nn = policy_nn
    return policy_wrapper


def _load_skrl_policy(checkpoint_path: str, task_name: str, env: gym.Env) -> Any:
    """Load skrl policy from checkpoint."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner
    
    experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(f"Expected dict for skrl config, got {type(experiment_cfg)}")
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
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
    
    experiment_cfg = experiment_cfg.copy()
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")
    
    def policy_wrapper(obs):
        """Policy wrapper for evaluation."""
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        if hasattr(env, "possible_agents"):
            return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
        return outputs[-1].get("mean_actions", outputs[0])
    
    policy_wrapper.policy_nn = runner.agent
    return policy_wrapper

