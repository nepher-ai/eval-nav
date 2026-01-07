# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment management utilities for navigation evaluation.

This module handles environment creation, verification, and configuration
building for different navigation tasks and robots.
"""

from __future__ import annotations

import importlib
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
        # Use configured module if provided
        if self.config.task_module:
            try:
                module = importlib.import_module(self.config.task_module)
                # Verify the import worked by checking if module was loaded
                if module is None:
                    raise ImportError(f"Failed to import module: {self.config.task_module}")
                return
            except ImportError as e:
                # Re-raise with more context - don't silently fail
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
        # Check if environment is registered using gym.spec()
        try:
            gym.spec(self.config.task_name)
        except gym.error.NameNotFound:
            # Environment not found - get list of available similar environments
            # registry is a runtime attribute, not in type stubs
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
        # Get environment-scene combinations
        env_scenes = self.config.env_scenes
        
        # Group by nav_env_id to verify each environment once
        env_scenes_map: dict[str, list[Any]] = {}
        for env_scene in env_scenes:
            nav_env_id = env_scene["nav_env_id"]
            nav_scene = env_scene["nav_scene"]
            if nav_env_id not in env_scenes_map:
                env_scenes_map[nav_env_id] = []
            env_scenes_map[nav_env_id].append(nav_scene)
        
        # Verify each environment
        all_missing_scenes = []
        for nav_env_id, scenes in env_scenes_map.items():
            try:
                # Load environment (this will download if needed, not just from cache)
                from envs_nav.lib.loader import load_env
                env = load_env(nav_env_id, cache_only=False)
                
                # Check available scenes
                if env.manifest is None:
                    raise EnvironmentError(
                        f"Environment '{nav_env_id}' has no manifest",
                        details={"nav_env_id": nav_env_id, "error_type": "ManifestError"},
                    )
                
                # Determine available scenes based on manifest type
                # Use resolved scenes (env.scenes/env.preset_scenes) to match what loader checks
                if env.manifest.type == "preset":
                    available_scenes = list(range(len(env.preset_scenes)))
                    scene_list = env.manifest.preset_scenes  # Use manifest for scene_id lookup
                else:
                    available_scenes = list(range(len(env.scenes)))
                    scene_list = env.manifest.scenes  # Use manifest for scene_id lookup
                
                # Check each required scene for this environment
                missing_scenes = []
                for scene in scenes:
                    if isinstance(scene, int):
                        if scene not in available_scenes:
                            missing_scenes.append(f"Scene index {scene} (available: 0-{len(available_scenes)-1})")
                    elif isinstance(scene, str):
                        # Try to find scene by ID
                        scene_found = False
                        for s in scene_list:
                            if s.scene_id.lower() == scene.lower():
                                scene_found = True
                                break
                        if not scene_found:
                            available_ids = [s.scene_id for s in scene_list]
                            missing_scenes.append(f"Scene ID '{scene}' (available: {available_ids})")
                
                if missing_scenes:
                    all_missing_scenes.append({
                        "nav_env_id": nav_env_id,
                        "missing_scenes": missing_scenes,
                        "available_scenes": available_scenes,
                    })
                
            except EnvironmentError:
                # Re-raise environment errors as-is
                raise
            except Exception as e:
                # Wrap other errors
                raise EnvironmentError(
                    f"Failed to verify scenes for environment '{nav_env_id}': {str(e)}",
                    details={"nav_env_id": nav_env_id, "error_type": type(e).__name__},
                ) from e
        
        if all_missing_scenes:
            error_messages = []
            details_list = []
            for error_info in all_missing_scenes:
                error_messages.append(
                    f"Environment '{error_info['nav_env_id']}' is missing required scenes: "
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
    
    def build_env_cfg(self, nav_env_id: str, nav_scene: str | int) -> Any:
        """Build environment configuration object.
        
        Args:
            nav_env_id: Navigation environment ID to use.
            nav_scene: Scene ID to use.
            
        Returns:
            Environment configuration object.
        """
        # Get the config class entry point from the registry
        cfg_entry_point = gym.spec(self.config.task_name).kwargs.get("env_cfg_entry_point")
        if cfg_entry_point is None:
            # If no entry point, return None and let gym.make use defaults
            return None
        
        # Parse the entry point (e.g., "module.path:ClassName")
        if ":" in cfg_entry_point:
            module_path, class_name = cfg_entry_point.rsplit(":", 1)
        else:
            # Handle YAML files if needed
            raise ValueError(f"Expected class entry point, got: {cfg_entry_point}")
        
        # Import and get the config class
        from importlib import import_module
        module = import_module(module_path)
        cfg_class = getattr(module, class_name)
        
        # Get env_config overrides from config (for additional parameters)
        env_config = self.config.env_config.copy() if self.config.env_config else {}
        
        # Set nav_env_id and nav_scene (required)
        env_config["nav_env_id"] = nav_env_id
        env_config["nav_scene"] = nav_scene
        
        # Instantiate config class with overrides
        # The config class will handle nav_env_id and nav_scene in its __post_init__
        cfg = cfg_class(**env_config)
        
        # Set num_envs for evaluation (mandatory, must be set after instantiation on scene.num_envs)
        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg.scene.num_envs = self.config.num_envs
        
        return cfg
    
    def load_environment_for_scene(self, nav_env_id: str, nav_scene: str | int) -> gym.Env:
        """Load environment configured for a specific scene.
        
        Args:
            nav_env_id: Navigation environment ID.
            nav_scene: Scene ID.
            
        Returns:
            Initialized Gymnasium environment for the scene (wrapped with EvalCompatEnv if available).
            
        Raises:
            EnvironmentError: If environment cannot be loaded.
        """
        try:
            print(f"[INFO] Building cfg for nav_env_id={nav_env_id}, nav_scene={nav_scene}")
            # Build environment config for this scene
            env_kwargs = {}
            cfg = self.build_env_cfg(nav_env_id=nav_env_id, nav_scene=nav_scene)
            if cfg is not None:
                env_kwargs["cfg"] = cfg
            
            print(f"[INFO] Creating gym env {self.config.task_name} with cfg (nav_env_id={nav_env_id}, nav_scene={nav_scene})")
            # Create environment
            env = gym.make(self.config.task_name, **env_kwargs)
            
            print(f"[INFO] New Env Created: {env}")
            # Try to wrap with eval_compat if available
            # This provides standardized state extraction methods for scalable logging
            try:
                # Try to import wrap_for_eval from the task module
                # Task modules export wrap_for_eval at the root level (e.g., leatherbacknav.wrap_for_eval)
                if self.config.task_module:
                    import importlib
                    task_module = importlib.import_module(self.config.task_module)
                    if hasattr(task_module, "wrap_for_eval"):
                        print(f"\n[INFO] Wrapping environment with eval_compat from {self.config.task_module}")
                        env = task_module.wrap_for_eval(env)
            except (ImportError, AttributeError, TypeError):
                # If wrap_for_eval is not available, continue without wrapping
                # The state logger will fall back to direct extraction
                pass
            
            return env
            
        except Exception as e:
            raise EnvironmentError(
                f"Failed to load environment '{self.config.task_name}' for nav_env_id={nav_env_id}, nav_scene={nav_scene}: {str(e)}",
                details={
                    "task_name": self.config.task_name,
                    "nav_env_id": nav_env_id,
                    "nav_scene": nav_scene,
                    "error_type": type(e).__name__,
                },
            ) from e

