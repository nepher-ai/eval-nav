# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core evaluation runner for navigation environments."""

from __future__ import annotations

import time
from typing import Any

from ..domain.config import EvalConfig
from ..domain.errors import (
    EnvironmentError,
    EvaluationRuntimeError,
    EvaluationStatus,
    TimeoutError,
)
from ..domain.metrics import AggregateMetrics, EpisodeMetrics
from ..managers.env_manager import EnvironmentManager
from .episode_runner import EpisodeRunner
from ..utils.policy_loader import load_policy_from_checkpoint
from .scorer import get_scorer


class NavigationEvaluator:
    """Evaluator for IsaacLab navigation environments.
    
    Runs fixed evaluation campaigns with:
    - Predefined scenes
    - Fixed random seeds
    - Fixed number of episodes
    - Deterministic execution
    """
    
    def __init__(self, config: EvalConfig, checkpoint_path: str | None = None):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
            checkpoint_path: Optional path to policy checkpoint (will be loaded lazily).
        """
        config.validate()
        self.config = config
        self.scorer = get_scorer(config.scoring_version)
        self.start_time: float | None = None
        self.checkpoint_path = checkpoint_path
        self._policy = None  # Will be loaded lazily
        
        # Initialize managers
        self.env_manager = EnvironmentManager(config)
        self.episode_runner = EpisodeRunner(config)
        
        # Import task module to ensure environment is registered
        try:
            self.env_manager.import_task_module()
            # Verify environment is registered
            self.env_manager.verify_environment_registered()
        except Exception as e:
            # Wrap import errors as EnvironmentError for consistent error handling
            if not isinstance(e, EnvironmentError):
                raise EnvironmentError(
                    f"Failed to import task module for '{config.task_name}': {str(e)}",
                    details={
                        "task_name": config.task_name,
                        "task_module": config.task_module,
                        "error_type": type(e).__name__,
                    },
                ) from e
            raise
    
    def evaluate(self, policy: Any | None = None) -> dict[str, Any]:
        """Run evaluation campaign.
        
        Args:
            policy: Policy to evaluate. If None, uses random actions.
            
        Returns:
            Dictionary containing evaluation results with keys:
            - status: EvaluationStatus
            - score: float (final score)
            - metrics: AggregateMetrics dict
            - episodes: list of EpisodeMetrics dicts
            - metadata: evaluation metadata
        """
        self.start_time = time.time()
        
        try:
            # Pre-verify that all required scenes are available
            self.env_manager.verify_scenes_available()
            
            # Run evaluation campaign (environments are created per scene)
            episodes = self._run_campaign(policy)
            
            # Compute aggregate metrics
            aggregate = AggregateMetrics.from_episodes(episodes)
            print(f"[INFO] Aggregate: {aggregate}")

            max_steps = self.config.max_episode_steps or 900
            # Compute final score
            score = self.scorer.compute_score_from_steps(aggregate, max_steps, episodes)
            # Prepare results
            results = {
                "status": EvaluationStatus.SUCCESS.value,
                "score": score,
                "metrics": aggregate.to_dict(),
                "episodes": [e.to_dict() for e in episodes],
                "metadata": self._get_metadata(),
            }
            
            return results
            
        except EnvironmentError as e:
            return {
                "status": e.status.value,
                "score": 0.0,
                "error": str(e),
                "details": e.details,
                "metadata": self._get_metadata(),
            }
        except EvaluationRuntimeError as e:
            return {
                "status": e.status.value,
                "score": 0.0,
                "error": str(e),
                "details": e.details,
                "metadata": self._get_metadata(),
            }
        except TimeoutError as e:
            return {
                "status": e.status.value,
                "score": 0.0,
                "error": str(e),
                "details": e.details,
                "metadata": self._get_metadata(),
            }
        except Exception as e:
            return {
                "status": EvaluationStatus.EVAL_ERROR.value,
                "score": 0.0,
                "error": f"Unexpected error: {str(e)}",
                "metadata": self._get_metadata(),
            }
    
    
    def _load_policy_lazy(self, env: Any) -> Any:
        """Load RSL-RL policy from checkpoint file using an existing environment.
        
        This is called lazily when we first have an environment available.
        
        Args:
            env: Existing gymnasium environment
            
        Returns:
            Policy function that takes observations and returns actions.
        """
        if self._policy is not None:
            return self._policy
        
        if self.checkpoint_path is None:
            return None
        
        # Load policy using the existing environment
        self._policy = load_policy_from_checkpoint(self.checkpoint_path, self.config.task_name, env)
        return self._policy
    
    def _run_campaign(
        self,
        policy: Any | None,
    ) -> list[EpisodeMetrics]:
        """Run evaluation campaign across all scene-seed combinations.
        
        Args:
            policy: Policy to evaluate (None for random, or will load from checkpoint if set).
            
        Returns:
            List of episode metrics.
            
        Raises:
            TimeoutError: If evaluation exceeds timeout.
        """
        episodes = []
        episode_id = 0
        
        # Iterate over all scene-seed combinations
        for scene in self.config.scenes:
            # Create environment for this scene (scene-specific config)
            scene_env = self.env_manager.load_environment_for_scene(scene)
            # Load policy lazily on first environment if checkpoint is provided
            if policy is None and self.checkpoint_path is not None:
                try:
                    policy = self._load_policy_lazy(scene_env)
                except Exception as e:
                    # If policy loading fails, continue with random actions
                    print(f"Warning: Failed to load policy: {e}. Using random actions.", file=__import__("sys").stderr)
                    policy = None
            
            try:
                for seed in self.config.seeds:
                    # Run episodes for this scene-seed combination
                    for _ in range(self.config.num_episodes):
                        # Check timeout
                        if self.config.timeout_seconds:
                            elapsed = time.time() - (self.start_time or 0)
                            if elapsed > self.config.timeout_seconds:
                                scene_env.close()
                                raise TimeoutError(
                                    f"Evaluation exceeded timeout of {self.config.timeout_seconds}s",
                                    details={"elapsed_seconds": elapsed},
                                )
                        
                        # Run single episode (may return multiple metrics for vectorized environments)
                        episode_metrics_list = self.episode_runner.run_episode(scene_env, policy, scene, seed, episode_id)
                        # Handle both single EpisodeMetrics and list of EpisodeMetrics
                        if isinstance(episode_metrics_list, list):
                            for episode_metrics in episode_metrics_list:
                                print(f"[INFO] Episode metrics: {episode_metrics}")
                                episodes.append(episode_metrics)
                                episode_id += 1
                        else:
                            print(f"[INFO] Episode metrics: {episode_metrics_list}")
                            episodes.append(episode_metrics_list)
                            episode_id += 1
            finally:
                scene_env.close()

        return episodes
    
    def _get_metadata(self) -> dict[str, Any]:
        """Get evaluation metadata.
        
        Returns:
            Metadata dictionary.
        """
        elapsed = time.time() - (self.start_time or time.time())
        
        return {
            "task_name": self.config.task_name,
            "scoring_version": self.config.scoring_version,
            "scenes": self.config.scenes,
            "seeds": self.config.seeds,
            "num_episodes": self.config.num_episodes,
            "max_episode_steps": self.config.max_episode_steps,
            "total_episodes_run": len(self.config.scenes) * len(self.config.seeds) * self.config.num_episodes,
            "elapsed_seconds": elapsed,
            "config": self.config.to_dict(),
        }

