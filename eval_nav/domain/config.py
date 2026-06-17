# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Configuration system for evaluation campaigns."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Supported (task_type, scoring_version) combinations — kept in sync with
# eval_nav.core.scorers.VALID_VERSIONS_PER_TASK_TYPE.
_VALID_VERSIONS_PER_TASK_TYPE: dict[str, list[str]] = {
    "navigation.leatherback": ["v1"],
    "navigation.spot": ["v2", "v3", "v4"],
    "manipulation.pick_place": ["v1", "v2"],
}


@dataclass
class EvalConfig:
    """Configuration for an evaluation campaign.

    Defines the target environment, scene selection, reproducibility seeds,
    episode parameters, and — crucially — **both** the task type and the
    scoring version that together identify the exact scorer to use.

    Scorer selection
    ----------------
    ``task_type`` identifies the robot/task domain:

    - ``"navigation.leatherback"`` — leatherback and ANYmal B waypoint navigation
    - ``"navigation.spot"``        — Spot quadruped tasks (waypoint or goal nav)
    - ``"manipulation.pick_place"`` — Franka high-level pick-and-place

    ``scoring_version`` selects the algorithm *within* that task type:

    +---------------------------+----------+-----------------------------------------+
    | task_type                 | versions | description                             |
    +===========================+==========+=========================================+
    | navigation.leatherback    | v1       | success (70%) + time (30%)              |
    +---------------------------+----------+-----------------------------------------+
    | navigation.spot           | v2       | success + time + locomotion quality     |
    |                           | v3       | per-episode; fail=0; time+stability     |
    |                           | v4       | success-rate-amplified + directness     |
    +---------------------------+----------+-----------------------------------------+
    | manipulation.pick_place   | v1       | task success (70%) + time (30%)         |
    |                           | v2       | success_rate × (0.75 + 0.25 × time)     |
    +---------------------------+----------+-----------------------------------------+
    """

    # -----------------------------------------------------------------------
    # Required fields (no default)
    # -----------------------------------------------------------------------

    task_name: str
    """Gymnasium task name (e.g. 'Nepher-Spot-Nav-Envhub-Student-Play-v0')."""

    num_envs: int
    """Number of parallel environments to use for evaluation."""

    # -----------------------------------------------------------------------
    # Scoring (both fields are first-class, not legacy)
    # -----------------------------------------------------------------------

    task_type: str = "navigation.leatherback"
    """Task domain that selects the scorer family.

    Supported: ``"navigation.leatherback"``, ``"navigation.spot"``,
    ``"manipulation.pick_place"``.
    """

    scoring_version: str = "v1"
    """Scoring algorithm version within the task type.

    Each task type supports specific versions — see class docstring for the
    full matrix.  Setting an invalid combination raises ``ValueError`` in
    ``validate()``.
    """

    # -----------------------------------------------------------------------
    # Environment / scene
    # -----------------------------------------------------------------------

    task_module: str | None = None
    """Python module to import for environment registration."""

    env_scenes: list[dict[str, Any]] = field(default_factory=list)
    """Environment-scene pairs to evaluate.
    Each dict must have ``'env_id'`` and ``'scene'`` keys."""

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------

    seeds: list[int] = field(default_factory=lambda: [42])
    """Random seeds for deterministic evaluation."""

    # -----------------------------------------------------------------------
    # Episode parameters
    # -----------------------------------------------------------------------

    num_episodes: int = 10
    """Number of episodes to run per scene-seed combination."""

    max_episode_steps: int | None = None
    """Maximum steps per episode.  If None, uses the environment default."""

    max_episode_time_s: float | None = None
    """Physical time budget in seconds.  When set, v3/v4 scorers normalize time
    efficiency against seconds instead of steps (decimation-invariant).
    If None, the evaluator auto-detects from the environment."""

    # -----------------------------------------------------------------------
    # Environment extras
    # -----------------------------------------------------------------------

    env_config: dict[str, Any] = field(default_factory=dict)
    """Additional environment configuration (optional)."""

    category: str = "navigation"
    """Nepher envhub category: ``'navigation'`` or ``'manipulation'``."""

    enable_cameras: bool = False
    """When True, ``scripts/evaluate.py`` passes ``--enable_cameras`` to Isaac Sim.
    Required for environments that spawn depth cameras (e.g. Spot student)."""

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    timeout_seconds: float | None = None
    """Maximum wall-clock time for the entire evaluation.  None = no timeout."""

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    log_dir: str | None = None
    """Directory for state logs (.npy files per episode/env)."""

    enable_logging: bool = False
    """Whether to enable state logging.  Requires ``log_dir`` to be set."""

    # -----------------------------------------------------------------------
    # Policy
    # -----------------------------------------------------------------------

    policy_path: str | None = None
    """Path to the RSL-RL checkpoint.  ``None`` or ``"default"`` resolves to
    ``<task-project>/best_policy/best_policy.pt``."""

    # -----------------------------------------------------------------------
    # Class methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Validated ``EvalConfig`` instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If ``task_type`` is absent from the YAML (would
                silently default to ``"navigation.leatherback"`` and produce a
                confusing scorer-mismatch error later).
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)

        if "task_type" not in data:
            raise ValueError(
                f"Config file {config_path} is missing the required 'task_type' field. "
                f"Add 'task_type' explicitly (e.g. 'navigation.spot') — omitting it "
                f"would silently default to 'navigation.leatherback' and cause a "
                f"scorer-mismatch error at evaluation time."
            )

        config = cls(**data)
        config._resolve_policy_path()
        return config

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: On any invalid field or invalid (task_type, scoring_version) combo.
        """
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

        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        # Validate (task_type, scoring_version) combo against registry
        supported_types = tuple(_VALID_VERSIONS_PER_TASK_TYPE.keys())
        if self.task_type not in _VALID_VERSIONS_PER_TASK_TYPE:
            raise ValueError(
                f"Unknown task_type: {self.task_type!r}. "
                f"Supported: {supported_types}"
            )
        valid_versions = _VALID_VERSIONS_PER_TASK_TYPE[self.task_type]
        if self.scoring_version not in valid_versions:
            raise ValueError(
                f"Unsupported scoring_version {self.scoring_version!r} "
                f"for task_type {self.task_type!r}. "
                f"Valid versions for this task type: {valid_versions}"
            )

        if not self.category:
            raise ValueError("category cannot be empty")
        if self.category not in ("navigation", "manipulation"):
            raise ValueError(
                f"Unsupported category: {self.category!r}. "
                "Supported: ('navigation', 'manipulation')"
            )

        if self.max_episode_time_s is not None and self.max_episode_time_s <= 0:
            raise ValueError("max_episode_time_s must be > 0 if specified")

        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 if specified")

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "task_name": self.task_name,
            "task_module": self.task_module,
            "task_type": self.task_type,
            "scoring_version": self.scoring_version,
            "env_scenes": self.env_scenes,
            "seeds": self.seeds,
            "num_episodes": self.num_episodes,
            "max_episode_steps": self.max_episode_steps,
            "max_episode_time_s": self.max_episode_time_s,
            "env_config": self.env_config,
            "num_envs": self.num_envs,
            "category": self.category,
            "enable_cameras": self.enable_cameras,
            "timeout_seconds": self.timeout_seconds,
            "log_dir": self.log_dir,
            "enable_logging": self.enable_logging,
            "policy_path": self.policy_path,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_policy_path(self) -> None:
        """Resolve ``policy_path`` to an actual file path when set to default."""
        if self.policy_path is None or self.policy_path == "default":
            task_project_dir = self._find_task_project_folder()
            if task_project_dir:
                default_path = task_project_dir / "best_policy" / "best_policy.pt"
                self.policy_path = str(default_path) if default_path.exists() else None
            else:
                self.policy_path = None

    def _find_task_project_folder(self) -> Path | None:
        """Walk up from ``task_module``'s file to find the ``task-*`` project root."""
        if not self.task_module:
            return None
        try:
            module = importlib.import_module(self.task_module)
            if hasattr(module, "__file__") and module.__file__:
                current = Path(module.__file__).parent
                for _ in range(10):
                    if current.name.startswith("task-"):
                        return current
                    parent = current.parent
                    if parent == current:
                        break
                    current = parent
        except ImportError:
            pass
        return None
