# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Environment management utilities for navigation evaluation.

This module handles environment creation, verification, and configuration
building for different navigation tasks and robots.
"""

from __future__ import annotations

import importlib
from importlib import import_module
from typing import Any

import gymnasium as gym

from ..domain.config import EvalConfig
from ..domain.errors import EnvironmentError


class EnvironmentManager:
    """Manages environment lifecycle for navigation evaluation."""
    
    def __init__(self, config: EvalConfig):
        """Initialize environment manager.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
    
    def import_task_module(self) -> None:
        """Import the task module to ensure environment registration.
        
        Uses task_module from config if provided, otherwise attempts to infer
        from task_name. Imports the module to trigger gym.register() calls.
        
        Raises:
            EnvironmentError: If module import fails.
        """
        if self.config.task_module:
            try:
                module = importlib.import_module(self.config.task_module)
                if module is None:
                    raise ImportError(f"Failed to import module: {self.config.task_module}")
                return
            except ImportError as e:
                raise EnvironmentError(
                    f"Failed to import task module '{self.config.task_module}': {str(e)}. "
                    f"Make sure the module is installed and available in PYTHONPATH.",
                    details={
                        "task_name": self.config.task_name,
                        "task_module": self.config.task_module,
                        "error_type": type(e).__name__,
                    },
                ) from e
        else:
            raise EnvironmentError(
                f"Failed to import task module: task_module is not specified in the config.",
                details={
                    "task_name": self.config.task_name,
                    "task_module": self.config.task_module,
                    "error_type": "MissingConfig",
                },
            )
    
    def verify_environment_registered(self) -> None:
        """Verify that the environment is registered with gymnasium.
        
        Raises:
            EnvironmentError: If environment is not registered.
        """
        try:
            gym.spec(self.config.task_name)
        except gym.error.NameNotFound:
            registry = getattr(gym.envs, "registry", {})
            all_envs = list(registry.keys()) if registry else []
            available_envs = [env_id for env_id in all_envs if "Leatherback" in env_id or "Animal" in env_id]
            raise EnvironmentError(
                f"Environment '{self.config.task_name}' is not registered. "
                f"Task module '{self.config.task_module}' was imported but registration failed. "
                f"Available similar environments: {available_envs[:10] if available_envs else 'None found'}",
                details={
                    "task_name": self.config.task_name,
                    "task_module": self.config.task_module,
                    "error_type": "NameNotFound",
                },
            )
    
    def verify_scenes_available(self) -> None:
        """Verify that all required scenes are available in the navigation environments.
        
        Pre-loads the environments to ensure they're fully downloaded and checks that
        all scenes specified in the config are available.
        
        Raises:
            EnvironmentError: If any required scene is not available.
        """
        env_scenes = self.config.env_scenes
        env_scenes_map: dict[str, list[Any]] = {}
        for env_scene in env_scenes:
            env_id = env_scene["env_id"]
            scene = env_scene["scene"]
            if env_id not in env_scenes_map:
                env_scenes_map[env_id] = []
            env_scenes_map[env_id].append(scene)
        
        all_missing_scenes = []
        for env_id, scenes in env_scenes_map.items():
            try:
                from nepher import load_env
                env = load_env(env_id, category="navigation")
                
                if env.type == "preset":
                    available_scenes = list(range(len(env.preset_scenes)))
                    scene_list = env.preset_scenes
                else:
                    available_scenes = list(range(len(env.scenes)))
                    scene_list = env.scenes
                
                missing_scenes = []
                for scene in scenes:
                    if isinstance(scene, int):
                        if scene not in available_scenes:
                            missing_scenes.append(f"Scene index {scene} (available: 0-{len(available_scenes)-1})")
                    elif isinstance(scene, str):
                        scene_found = False
                        for s in scene_list:
                            if s.name.lower() == scene.lower():
                                scene_found = True
                                break
                        if not scene_found:
                            available_ids = [s.name for s in scene_list]
                            missing_scenes.append(f"Scene ID '{scene}' (available: {available_ids})")
                
                if missing_scenes:
                    all_missing_scenes.append({
                        "env_id": env_id,
                        "missing_scenes": missing_scenes,
                        "available_scenes": available_scenes,
                    })
                
            except EnvironmentError:
                raise
            except Exception as e:
                raise EnvironmentError(
                    f"Failed to verify scenes for environment '{env_id}': {str(e)}",
                    details={"env_id": env_id, "error_type": type(e).__name__},
                ) from e
        
        if all_missing_scenes:
            error_messages = []
            details_list = []
            for error_info in all_missing_scenes:
                error_messages.append(
                    f"Environment '{error_info['env_id']}' is missing required scenes: "
                    f"{', '.join(error_info['missing_scenes'])}. "
                    f"Total scenes available: {len(error_info['available_scenes'])}"
                )
                details_list.append(error_info)
            
            raise EnvironmentError(
                "Multiple environments have missing scenes: " + "; ".join(error_messages),
                details={
                    "errors": details_list,
                    "error_type": "ManifestError",
                },
            )
    
    def build_env_cfg(self, env_id: str, scene: str | int) -> Any:
        """Build environment configuration object.
        
        Args:
            env_id: Environment ID to use.
            scene: Scene ID to use.
            
        Returns:
            Environment configuration object.
        """
        cfg_entry_point = gym.spec(self.config.task_name).kwargs.get("env_cfg_entry_point")
        if cfg_entry_point is None:
            return None
        
        if ":" in cfg_entry_point:
            module_path, class_name = cfg_entry_point.rsplit(":", 1)
        else:
            raise ValueError(f"Expected class entry point, got: {cfg_entry_point}")
        
        module = import_module(module_path)
        cfg_class = getattr(module, class_name)
        
        env_config = self.config.env_config.copy() if self.config.env_config else {}
        env_config["env_id"] = env_id
        env_config["scene_id"] = scene
        
        cfg = cfg_class(**env_config)
        
        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg.scene.num_envs = self.config.num_envs
        
        return cfg
    
    def load_environment_for_scene(self, env_id: str, scene: str | int) -> gym.Env:
        """Load environment configured for a specific scene.
        
        Args:
            env_id: Environment ID.
            scene: Scene ID.
            
        Returns:
            Initialized Gymnasium environment for the scene (wrapped with EvalCompatEnv if available).
            
        Raises:
            EnvironmentError: If environment cannot be loaded.
        """
        try:
            print(f"[INFO] Building cfg for env_id={env_id}, scene={scene}")
            env_kwargs = {}
            cfg = self.build_env_cfg(env_id=env_id, scene=scene)
            if cfg is not None:
                env_kwargs["cfg"] = cfg
            
            print(f"[INFO] Creating gym env {self.config.task_name} with cfg (env_id={env_id}, scene={scene})")
            env = gym.make(self.config.task_name, **env_kwargs)
            
            print(f"[INFO] New Env Created: {env}")
            try:
                if self.config.task_module:
                    task_module = import_module(self.config.task_module)
                    if hasattr(task_module, "wrap_for_eval"):
                        print(f"\n[INFO] Wrapping environment with eval_compat from {self.config.task_module}")
                        env = task_module.wrap_for_eval(env)
            except (ImportError, AttributeError, TypeError):
                pass
            
            return env
            
        except Exception as e:
            raise EnvironmentError(
                f"Failed to load environment '{self.config.task_name}' for env_id={env_id}, scene={scene}: {str(e)}",
                details={
                    "task_name": self.config.task_name,
                    "env_id": env_id,
                    "scene": scene,
                    "error_type": type(e).__name__,
                },
            ) from e

