# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Policy loading utilities for navigation evaluation."""

from __future__ import annotations

import importlib.metadata as metadata
import os
from typing import Any

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl import rsl_rl as _rsl_rl
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Lab 3 exports this helper. Lab 2.3 does not; its rsl-rl still accepts the legacy policy config.
handle_deprecated_rsl_rl_cfg = getattr(_rsl_rl, "handle_deprecated_rsl_rl_cfg", None)


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
    # rsl-rl >= 5 rejects the legacy ``stochastic`` fields still present on the
    # Isaac Lab model configs. Play and train already strip them; without this
    # the runner fails to construct and evaluation falls back to random actions.
    # Lab 2.3 has no helper and still accepts those fields.
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    
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

    # rsl-rl < 2.3 stores one module as ``actor_critic``. 2.3–3 use ``policy``.
    # >= 4 splits the network into ``actor`` and ``critic`` and has neither name.
    policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "actor_critic", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "actor", None)

    import torch

    # Lab 2.3 returns ``act_inference`` (a bound method). Those tournaments were
    # scored from that grad-enabled mean, so leave the call unchanged.
    # Lab 3 returns the model module. Detach its output: a grad-enabled action
    # reaches the frozen low-level policy and Warp refuses it. ``no_grad`` rather
    # than ``inference_mode`` so an LSTM hidden state can be cleared next episode.
    recurrent = bool(getattr(policy, "is_recurrent", False) or getattr(policy_nn, "is_recurrent", False))
    if isinstance(policy, torch.nn.Module):

        def policy_wrapper(obs):
            """Policy wrapper for evaluation."""
            with torch.no_grad():
                action = policy(obs)
            if torch.is_tensor(action):
                return action.detach()
            if isinstance(action, dict):
                return {key: value.detach() if torch.is_tensor(value) else value for key, value in action.items()}
            return action

        if recurrent:

            def reset_hidden(dones=None):
                """Clear the recurrent state. ``dones`` selects environments; None clears all."""
                reset = getattr(policy, "reset", None)
                if reset is None and policy_nn is not None:
                    reset = getattr(policy_nn, "reset", None)
                if reset is not None:
                    reset(dones)

            policy_wrapper.reset = reset_hidden
    else:

        def policy_wrapper(obs):
            """Policy wrapper for evaluation."""
            return policy(obs)

    policy_wrapper.policy_nn = policy_nn if policy_nn is not None else policy
    return policy_wrapper


def _alias_skrl_preprocessor_modules(agent: Any) -> None:
    """Map skrl 1.x / 2.x preprocessor checkpoint keys onto each other.

    skrl 1 stored the observation scaler as ``state_preprocessor``. skrl 2's
    Runner rewrites that YAML field to ``observation_preprocessor``, so
    ``agent.load`` would skip the scaler and leave RunningStandardScaler empty.
    Point missing names at the live module so either checkpoint generation loads.
    """
    modules = getattr(agent, "checkpoint_modules", None)
    if not isinstance(modules, dict):
        return
    obs = modules.get("observation_preprocessor")
    state = modules.get("state_preprocessor")
    if obs is not None and state is None:
        modules["state_preprocessor"] = obs
    elif state is not None and obs is None:
        modules["observation_preprocessor"] = state


def _load_skrl_policy(checkpoint_path: str, task_name: str, env: gym.Env) -> Any:
    """Load skrl policy from checkpoint."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.envs.wrappers.torch import Wrapper as SkrlEnvWrapper
    from skrl.utils.runner.torch import Runner
    
    experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(f"Expected dict for skrl config, got {type(experiment_cfg)}")
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # SkrlVecEnvWrapper is a factory function, not a class — check the skrl Wrapper base.
    is_wrapped = isinstance(env, SkrlEnvWrapper)
    if not is_wrapped:
        current = env
        while True:
            next_env = getattr(current, "env", None)
            if next_env is None and type(current).__name__ == "EvalCompatEnv":
                next_env = getattr(current, "_env", None)
            if next_env is None:
                break
            current = next_env
            if isinstance(current, SkrlEnvWrapper):
                is_wrapped = True
                break
    
    if not is_wrapped:
        # Prefer the underlying gym env when an EvalCompatEnv sits on top so skrl
        # sees a normal gymnasium / Isaac Lab stack (same as play.py).
        wrap_target = env
        if type(wrap_target).__name__ == "EvalCompatEnv" and hasattr(wrap_target, "_env"):
            wrap_target = wrap_target._env
        env = SkrlVecEnvWrapper(wrap_target, ml_framework="torch")
    
    experiment_cfg = experiment_cfg.copy()
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    # skrl 2 remaps YAML ``state_preprocessor`` → ``observation_preprocessor``, so a
    # skrl 1 checkpoint's ``state_preprocessor`` key is skipped unless we alias it.
    _alias_skrl_preprocessor_modules(runner.agent)
    runner.agent.load(checkpoint_path)
    # skrl 1.x: set_running_mode / set_mode; skrl 2.x: enable_training_mode.
    agent = runner.agent
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode("eval")
    elif hasattr(agent, "enable_training_mode"):
        agent.enable_training_mode(False, apply_to_models=True)
    elif hasattr(agent, "set_mode"):
        agent.set_mode("eval")
    elif hasattr(agent, "enable_models_training_mode"):
        agent.enable_models_training_mode(False)

    # Episode runner steps the gym/EvalCompat stack (dict obs). skrl agents expect the
    # flat policy tensor produced by IsaacLabWrapper — mirror that conversion here.
    import inspect

    import torch
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

    # skrl 1.x: act(obs, *, timestep, timesteps)
    # skrl 2.x: act(observations, states, *, timestep, timesteps)  (see Isaac Lab #5311)
    _act_params = inspect.signature(agent.act).parameters
    _skrl_v2_act = "observations" in _act_params and "states" in _act_params

    def _to_skrl_obs(obs: Any) -> Any:
        if isinstance(obs, dict):
            if hasattr(env, "possible_agents"):
                return {
                    a: flatten_tensorized_space(tensorize_space(env.observation_spaces[a], obs[a]))
                    for a in env.possible_agents
                }
            policy_obs = obs["policy"] if "policy" in obs else next(iter(obs.values()))
            return flatten_tensorized_space(tensorize_space(env.observation_space, policy_obs))
        return obs

    def _extract_actions(outputs: Any) -> Any:
        # Both skrl generations return (actions, extras) / (... , extras); prefer mean.
        if hasattr(env, "possible_agents"):
            return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
        return outputs[-1].get("mean_actions", outputs[0])

    def policy_wrapper(obs):
        """Policy wrapper for evaluation (skrl 1.x and 2.x)."""
        skrl_obs = _to_skrl_obs(obs)
        with torch.inference_mode():
            if _skrl_v2_act:
                states = env.state() if hasattr(env, "state") else None
                outputs = agent.act(skrl_obs, states, timestep=0, timesteps=0)
            else:
                outputs = agent.act(skrl_obs, timestep=0, timesteps=0)
        return _extract_actions(outputs)

    policy_wrapper.policy_nn = agent
    return policy_wrapper

