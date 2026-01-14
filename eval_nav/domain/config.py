# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration system for navigation evaluation."""

from __future__ import annotations

import importlib
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalConfig:
    """Configuration for navigation evaluation campaign.
    
    Defines the evaluation parameters including:
    - Target environment (Gym/IsaacLab task name)
    - Scenes/environments to evaluate
    - Random seeds for reproducibility
    - Episode count and time horizon
    - Scoring version
    """
    
    # Environment selection
    task_name: str
    """Gymnasium task name (e.g., 'Nepher-Animal-WaypointNav-Envhub-Play-v0')."""
    
    num_envs: int
    """Number of parallel environments to use for evaluation (mandatory)."""
    
    task_module: str | None = None
    """Python module to import for environment registration (e.g., 'leatherbacknav', 'animalnav').
    If None, the evaluator will attempt to infer from task_name."""
    
    # Scene/environment selection
    env_scenes: list[dict[str, Any]] = field(default_factory=list)
    """List of environment-scene combinations to evaluate.
    Each dict must have 'env_id' and 'scene' keys.
    Example: [{"env_id": "waypoint-benchmark-v1", "scene": 0}, {"env_id": "waypoint-sample-v1", "scene": 0}]"""
    
    # Reproducibility
    seeds: list[int] = field(default_factory=lambda: [42])
    """List of random seeds for deterministic evaluation."""
    
    # Evaluation parameters
    num_episodes: int = 10
    """Number of episodes to run per scene-seed combination."""
    
    max_episode_steps: int | None = None
    """Maximum steps per episode. If None, uses environment default."""
    
    # Scoring
    scoring_version: str = "v1"
    """Scoring version to use. MVP supports 'v1' only."""
    
    # Environment-specific config (optional, for additional environment parameters)
    env_config: dict[str, Any] = field(default_factory=dict)
    """Additional environment configuration (optional, for non-scene-specific parameters)."""
    
    # Timeout
    timeout_seconds: float | None = None
    """Maximum wall-clock time for entire evaluation. None = no timeout."""
    
    # Logging
    log_dir: str | None = None
    """Directory to save state logs (.npy files per episode/env). None = no logging."""
    enable_logging: bool = False
    """Whether to enable state logging. Requires log_dir to be set."""
    
    # Policy
    policy_path: str | None = None
    """Path to policy checkpoint file. If None or "default", will attempt to find 
    "best_policy/best_policy.pt" in the task project folder."""
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> EvalConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            EvalConfig instance.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls(**data)
        # Resolve policy_path if it's "default" or None
        config._resolve_policy_path()
        return config
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "task_name": self.task_name,
            "task_module": self.task_module,
            "env_scenes": self.env_scenes,
            "seeds": self.seeds,
            "num_episodes": self.num_episodes,
            "max_episode_steps": self.max_episode_steps,
            "scoring_version": self.scoring_version,
            "env_config": self.env_config,
            "num_envs": self.num_envs,
            "timeout_seconds": self.timeout_seconds,
            "log_dir": self.log_dir,
            "enable_logging": self.enable_logging,
            "policy_path": self.policy_path,
        }
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.task_name:
            raise ValueError("task_name cannot be empty")
        
        # Validate env_scenes
        if not self.env_scenes:
            raise ValueError("env_scenes list cannot be empty")
        for i, env_scene in enumerate(self.env_scenes):
            if not isinstance(env_scene, dict):
                raise ValueError(f"env_scenes[{i}] must be a dictionary")
            if "env_id" not in env_scene:
                raise ValueError(f"env_scenes[{i}] must have 'env_id' key")
            if "scene" not in env_scene:
                raise ValueError(f"env_scenes[{i}] must have 'scene' key")
        
        if not self.seeds:
            raise ValueError("seeds list cannot be empty")
        
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be >= 1")
        
        if self.scoring_version != "v1":
            raise ValueError(f"Only scoring version 'v1' is supported in MVP. Got: {self.scoring_version}")
        
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 if specified")
        
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
    
    def _resolve_policy_path(self) -> None:
        """Resolve policy_path to actual file path.
        
        If policy_path is None or "default", attempts to find 
        "best_policy/best_policy.pt" in the task project folder.
        """
        if self.policy_path is None or self.policy_path == "default":
            # Try to find task project folder
            task_project_dir = self._find_task_project_folder()
            if task_project_dir:
                default_path = task_project_dir / "best_policy" / "best_policy.pt"
                if default_path.exists():
                    self.policy_path = str(default_path)
                else:
                    # Set to None if default path doesn't exist (will use random actions)
                    self.policy_path = None
            else:
                # Could not find task project folder, set to None
                self.policy_path = None
    
    def _find_task_project_folder(self) -> Path | None:
        """Find the task project folder based on task_module.
        
        Returns:
            Path to task project folder if found, None otherwise.
        """
        if not self.task_module:
            return None
        
        # Try to get the module's file location
        try:
            module = importlib.import_module(self.task_module)
            if hasattr(module, "__file__") and module.__file__:
                module_path = Path(module.__file__)
                # Navigate up from the module file to find the task project root
                # Module structure: source/task-*/source/module/...
                # We want: source/task-*/
                current = module_path.parent
                # Go up until we find a directory starting with "task-"
                for _ in range(10):  # Limit depth to avoid infinite loops
                    if current.name.startswith("task-"):
                        return current
                    parent = current.parent
                    if parent == current:  # Reached root
                        break
                    current = parent
        except ImportError:
            pass
        
        return None

