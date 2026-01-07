# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration system for navigation evaluation."""

from __future__ import annotations

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
    """Gymnasium task name (e.g., 'Nepher-Animal-WaypointNav-Envs-Play-v0')."""
    
    num_envs: int
    """Number of parallel environments to use for evaluation (mandatory)."""
    
    task_module: str | None = None
    """Python module to import for environment registration (e.g., 'leatherbacknav', 'animalnav').
    If None, the evaluator will attempt to infer from task_name."""
    
    # Scene/environment selection
    scenes: list[str | int] = field(default_factory=lambda: [0])
    """List of scene IDs to evaluate. Can be strings or integers."""
    
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
    
    # Environment-specific config
    env_config: dict[str, Any] = field(default_factory=dict)
    """Additional environment configuration (e.g., nav_env_id, nav_scene)."""
    
    # Timeout
    timeout_seconds: float | None = None
    """Maximum wall-clock time for entire evaluation. None = no timeout."""
    
    # Logging
    log_dir: str | None = None
    """Directory to save state logs (.npy files per episode/env). None = no logging."""
    enable_logging: bool = False
    """Whether to enable state logging. Requires log_dir to be set."""
    
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
        
        return cls(**data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "task_name": self.task_name,
            "task_module": self.task_module,
            "scenes": self.scenes,
            "seeds": self.seeds,
            "num_episodes": self.num_episodes,
            "max_episode_steps": self.max_episode_steps,
            "scoring_version": self.scoring_version,
            "env_config": self.env_config,
            "num_envs": self.num_envs,
            "timeout_seconds": self.timeout_seconds,
            "log_dir": self.log_dir,
            "enable_logging": self.enable_logging,
        }
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.task_name:
            raise ValueError("task_name cannot be empty")
        
        if not self.scenes:
            raise ValueError("scenes list cannot be empty")
        
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

