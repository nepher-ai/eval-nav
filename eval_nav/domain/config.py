# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Configuration system for navigation evaluation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml


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
    
    max_episode_time_s: float | None = None
    """Physical time budget in seconds for the episode.  When set, V3+ scorers
    normalize time efficiency against this instead of ``max_episode_steps``.
    If None, the evaluator auto-detects from the environment
    (``episode_length_s``) or derives it from ``max_episode_steps * step_dt``."""
    
    # Scoring
    scoring_version: str = "v1"
    """Scoring version. 'v1' = success+time aggregate, 'v2' = success+time+locomotion aggregate, 'v3' = mean per-episode (fail=0; success=time+stability), 'v4' = success-rate-amplified quality with directness (success_rate × (bonus + quality))."""
    
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
    
    enable_cameras: bool = False
    """When True, ``scripts/evaluate.py`` passes ``--enable_cameras`` to Isaac Sim
    before AppLauncher. Required for environments that spawn cameras (e.g. Spot
    student with depth). CLI ``--enable_cameras`` still overrides if present."""
    
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
        
        # Read YAML configs as UTF-8 to support non-ASCII characters
        with open(config_path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        
        config = cls(**data)
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
            "max_episode_time_s": self.max_episode_time_s,
            "scoring_version": self.scoring_version,
            "env_config": self.env_config,
            "num_envs": self.num_envs,
            "timeout_seconds": self.timeout_seconds,
            "log_dir": self.log_dir,
            "enable_logging": self.enable_logging,
            "policy_path": self.policy_path,
            "enable_cameras": self.enable_cameras,
        }
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.task_name:
            raise ValueError("task_name cannot be empty")
        
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
        
        if self.scoring_version not in ("v1", "v2", "v3", "v4"):
            raise ValueError(f"Unsupported scoring version: {self.scoring_version}. Supported: 'v1', 'v2', 'v3', 'v4'.")
        
        if self.max_episode_time_s is not None and self.max_episode_time_s <= 0:
            raise ValueError("max_episode_time_s must be > 0 if specified")
        
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
            task_project_dir = self._find_task_project_folder()
            if task_project_dir:
                default_path = task_project_dir / "best_policy" / "best_policy.pt"
                if default_path.exists():
                    self.policy_path = str(default_path)
                else:
                    self.policy_path = None
            else:
                self.policy_path = None
    
    def _find_task_project_folder(self) -> Path | None:
        """Find the task project folder based on task_module.
        
        Returns:
            Path to task project folder if found, None otherwise.
        """
        if not self.task_module:
            return None
        
        try:
            module = importlib.import_module(self.task_module)
            if hasattr(module, "__file__") and module.__file__:
                module_path = Path(module.__file__)
                current = module_path.parent
                for _ in range(10):  # Limit depth to avoid infinite loops
                    if current.name.startswith("task-"):
                        return current
                    parent = current.parent
                    if parent == current:
                        break
                    current = parent
        except ImportError:
            pass
        
        return None

