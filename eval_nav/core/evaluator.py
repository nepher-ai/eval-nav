# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Core evaluation runner for navigation environments."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from multiprocessing import Process
from typing import Any

from ..domain.config import EvalConfig
from ..domain.errors import (
    EnvironmentError,
    EvaluationRuntimeError,
    EvaluationStatus,
    EvaluationTimeoutError,
)
from ..domain.metrics import AggregateMetrics, EpisodeMetrics
from ..managers.env_manager import EnvironmentManager
from .episode_runner import EpisodeRunner
from ..utils.policy_loader import load_policy_from_checkpoint
from .scorer import get_scorer


def _run_env_scene_worker(config_dict: dict, env_scene_combo: dict, checkpoint_path: str | None, output_path: str) -> None:
    """Subprocess worker to evaluate a single env-scene combination.
    
    Args:
        config_dict: Configuration dictionary.
        env_scene_combo: Environment-scene combination to evaluate.
        checkpoint_path: Optional path to policy checkpoint.
        output_path: Path to save episode results as JSON.
    """
    # Import here to avoid circular imports and ensure proper initialization in subprocess
    from ..domain.config import EvalConfig
    from .evaluator import NavigationEvaluator
    
    config_dict = dict(config_dict)
    config_dict["env_scenes"] = [env_scene_combo]
    cfg = EvalConfig(**config_dict)
    evaluator = NavigationEvaluator(cfg, checkpoint_path=checkpoint_path, subprocess_mode=True)
    episodes = evaluator.run_campaign(policy=None)
    with open(output_path, "w") as f:
        json.dump([e.to_dict() for e in episodes], f)


class NavigationEvaluator:
    """Evaluator for IsaacLab navigation environments.
    
    Runs fixed evaluation campaigns with:
    - Predefined scenes
    - Fixed random seeds
    - Fixed number of episodes
    - Deterministic execution
    """
    
    def __init__(self, config: EvalConfig, checkpoint_path: str | None = None, subprocess_mode: bool = False):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
            checkpoint_path: Optional path to policy checkpoint (will be loaded lazily).
            subprocess_mode: If True, do not spawn further subprocesses (used by worker).
        """
        config.validate()
        self.config = config
        self.scorer = get_scorer(config.scoring_version)
        self.start_time: float | None = None
        self.checkpoint_path = checkpoint_path
        self._policy = None  # Will be loaded lazily
        self.subprocess_mode = subprocess_mode
        self.env_manager = EnvironmentManager(config)
        self.episode_runner = EpisodeRunner(config)
        
        try:
            self.env_manager.import_task_module()
            self.env_manager.verify_environment_registered()
        except Exception as e:
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

    def run_campaign(self, policy: Any | None) -> list[EpisodeMetrics]:
        """Public wrapper for running the campaign (used by subprocess workers)."""
        return self._run_campaign(policy)
    
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
            self.env_manager.verify_scenes_available()
            episodes = self._run_campaign(policy)
            aggregate = AggregateMetrics.from_episodes(episodes)
            print(f"[INFO] Aggregate: {aggregate}")

            max_steps = self.config.max_episode_steps or 900
            score = self.scorer.compute_score_from_steps(aggregate, max_steps, episodes)
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
        except EvaluationTimeoutError as e:
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
        
        self._policy = load_policy_from_checkpoint(self.checkpoint_path, self.config.task_name, env)
        return self._policy
    
    def _run_campaign(
        self,
        policy: Any | None,
    ) -> list[EpisodeMetrics]:
        """Run evaluation campaign across all environment-scene-seed combinations.
        
        Args:
            policy: Policy to evaluate (None for random, or will load from checkpoint if set).
            
        Returns:
            List of episode metrics.
            
        Raises:
            TimeoutError: If evaluation exceeds timeout.
        """
        episodes = []
        episode_id = 0
        env_scene_combos = self.config.env_scenes

        # If multiple envs/scenes, run each in a fresh subprocess to avoid simulator reuse hangs
        if not self.subprocess_mode and len(env_scene_combos) > 1:
            temp_files = []
            processes = []
            for combo in env_scene_combos:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                tmp_path = tmp.name
                tmp.close()
                temp_files.append(tmp_path)
                p = Process(
                    target=_run_env_scene_worker,
                    args=(self.config.to_dict(), combo, self.checkpoint_path, tmp_path),
                )
                p.start()
                processes.append((p, tmp_path))

            for p, tmp_path in processes:
                p.join()
                if p.exitcode != 0:
                    raise EvaluationRuntimeError(f"Subprocess failed for {tmp_path}")
                with open(tmp_path, "r") as f:
                    episode_dicts = json.load(f)
                os.remove(tmp_path)
                for ep_dict in episode_dicts:
                    ep = EpisodeMetrics(**ep_dict)
                    ep.episode_id = episode_id
                    episode_id += 1
                    episodes.append(ep)

            return episodes
        
        for env_scene_combo in env_scene_combos:
            env_id = env_scene_combo["env_id"]
            scene = env_scene_combo["scene"]
            
            print(f"[INFO] Loading environment: env_id={env_id}, scene={scene}")
            scene_env = self.env_manager.load_environment_for_scene(env_id=env_id, scene=scene)  # type: ignore[attr-defined]
            print(f"[INFO] Environment ready: env_id={env_id}, scene={scene}")
            if policy is None and self.checkpoint_path is not None:
                try:
                    policy = self._load_policy_lazy(scene_env)
                except Exception as e:
                    print(f"Warning: Failed to load policy: {e}. Using random actions.", file=sys.stderr)
                    policy = None
            
            try:
                for seed in self.config.seeds:
                    for _ in range(self.config.num_episodes):
                        if self.config.timeout_seconds:
                            elapsed = time.time() - (self.start_time or 0)
                            if elapsed > self.config.timeout_seconds:
                                scene_env.close()
                                raise EvaluationTimeoutError(
                                    f"Evaluation exceeded timeout of {self.config.timeout_seconds}s",
                                    details={"elapsed_seconds": elapsed},
                                )
                        
                        episode_metrics_list = self.episode_runner.run_episode(
                            scene_env,
                            policy,
                            scene,
                            env_id,
                            seed,
                            episode_id,
                        )  # type: ignore[attr-defined]
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
                try:
                    import gc
                    del scene_env
                    gc.collect()
                except Exception:
                    pass

        return episodes
    
    def _get_metadata(self) -> dict[str, Any]:
        """Get evaluation metadata.
        
        Returns:
            Metadata dictionary.
        """
        elapsed = time.time() - (self.start_time or time.time())
        num_combos = len(self.config.env_scenes)
        scenes = [
            f"{combo['env_id']}:{combo['scene']}"
            for combo in self.config.env_scenes
        ]
        
        return {
            "task_name": self.config.task_name,
            "scoring_version": self.config.scoring_version,
            "scenes": scenes,
            "env_scenes": self.config.env_scenes,
            "seeds": self.config.seeds,
            "num_episodes": self.config.num_episodes,
            "max_episode_steps": self.config.max_episode_steps,
            "total_episodes_run": num_combos * len(self.config.seeds) * self.config.num_episodes,
            "elapsed_seconds": elapsed,
            "config": self.config.to_dict(),
        }

