# Copyright (c) 2026, Nepher Robotics
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


# Grace period (seconds) granted to a worker subprocess to exit on its own AFTER
# it has flushed episode results to disk. Isaac Sim teardown (USD stage detach /
# plugin unload) can hang indefinitely in native code; once results are on disk
# we must not let that hang consume the whole evaluation timeout. The worker
# normally hard-exits itself (see _run_campaign), so this is only a backstop.
_CLOSE_GRACE_SECONDS = float(os.environ.get("NEPHER_EVAL_CLOSE_GRACE_SECONDS", "120"))
_JOIN_POLL_SECONDS = 2.0


def _results_file_ready(path: str) -> bool:
    """Return True if `path` holds a valid, non-empty JSON results payload."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if not content:
            return False
        json.loads(content)
        return True
    except (OSError, ValueError):
        return False


def _join_worker_with_teardown_guard(p: "Process", tmp_path: str) -> None:
    """Join a worker subprocess without letting a hung teardown block forever.

    The worker flushes its episode results to ``tmp_path`` BEFORE calling
    ``env.close()`` and normally hard-exits immediately after the flush. If for
    any reason it is still alive after results are on disk (e.g. stuck in native
    Isaac Sim teardown), grant a short grace period and then terminate it so the
    parent can proceed with the already-collected data instead of timing out.
    """
    results_ready_since: float | None = None
    while True:
        p.join(timeout=_JOIN_POLL_SECONDS)
        if p.exitcode is not None:
            return  # exited on its own (clean exit or crash)

        if results_ready_since is None and _results_file_ready(tmp_path):
            results_ready_since = time.time()
            print(
                f"[INFO] Worker {p.pid} flushed results; granting "
                f"{_CLOSE_GRACE_SECONDS:.0f}s for clean shutdown before forcing termination."
            )

        if (
            results_ready_since is not None
            and (time.time() - results_ready_since) > _CLOSE_GRACE_SECONDS
        ):
            print(
                f"[WARNING] Worker {p.pid} still alive {_CLOSE_GRACE_SECONDS:.0f}s after "
                f"writing results (stuck in env.close()/teardown) — terminating."
            )
            p.terminate()
            p.join(timeout=10)
            if p.exitcode is None:
                print(f"[WARNING] Worker {p.pid} ignored SIGTERM — sending SIGKILL.")
                p.kill()
                p.join(timeout=10)
            return


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
    # Tell the evaluator to persist results before env.close() so that a native
    # crash inside close() (e.g. IsaacLab reward_manager destructor stack overflow)
    # does not discard the episode data.
    evaluator._subprocess_output_path = output_path
    episodes = evaluator.run_campaign(policy=None)
    # Fallback write: if close() completed cleanly the file was already written
    # inside _run_campaign; overwrite with the return value to stay consistent.
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump([e.to_dict() for e in episodes], f, ensure_ascii=False)


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
        self.scorer = get_scorer(config.task_type, config.scoring_version)
        self.start_time: float | None = None
        self.checkpoint_path = checkpoint_path
        self._policy = None  # Will be loaded lazily
        self.subprocess_mode = subprocess_mode
        # When running as a subprocess worker, set to the output file path so
        # that _run_campaign writes results before calling env.close().  This
        # lets the parent process recover episode data even when close() triggers
        # a native stack overflow (e.g. IsaacLab reward_manager destructor).
        self._subprocess_output_path: str | None = None
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
            max_episode_time_s = self._resolve_max_episode_time_s(episodes, max_steps)
            score = self.scorer.compute_score_from_steps(
                aggregate, max_steps, episodes,
                max_episode_time_s=max_episode_time_s,
            )
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

        # Run each env/scene in a dedicated subprocess so that a native crash
        # inside env.close() (e.g. IsaacLab destructor stack overflow) does not
        # kill the parent process and lose the collected episode data.
        if not self.subprocess_mode and len(env_scene_combos) >= 1:
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
                _join_worker_with_teardown_guard(p, tmp_path)
                if p.exitcode != 0:
                    # The subprocess may have crashed or been terminated inside
                    # env.close()/teardown *after* already writing results to
                    # tmp_path.  Validate the file before deciding whether to raise.
                    try:
                        with open(tmp_path, "r", encoding="utf-8", errors="replace") as _f:
                            _content = _f.read().strip()
                        if not _content:
                            raise EvaluationRuntimeError(
                                f"Subprocess crashed (exit {p.exitcode}) before writing results: {tmp_path}"
                            )
                        json.loads(_content)  # raises ValueError on malformed JSON
                    except (OSError, ValueError) as _exc:
                        raise EvaluationRuntimeError(
                            f"Subprocess crashed (exit {p.exitcode}) with no valid results: {tmp_path}"
                        ) from _exc
                    print(
                        f"[WARNING] Subprocess exited with code {p.exitcode} (likely during env.close()), "
                        f"but episode results were already saved — continuing."
                    )
                # Read subprocess output JSON as UTF-8
                with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
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
                # Persist results NOW, before env.close(), so that a native crash
                # OR hang inside close() (e.g. IsaacLab reward_manager destructor
                # stack overflow, or Isaac Sim USD stage detach / plugin unload
                # hanging indefinitely) does not discard the collected episode data.
                if self._subprocess_output_path:
                    with open(self._subprocess_output_path, "w", encoding="utf-8", errors="replace") as _f:
                        json.dump([e.to_dict() for e in episodes], _f, ensure_ascii=False)
                        _f.flush()
                        os.fsync(_f.fileno())

                    # Hard-exit the worker to bypass Isaac Sim's teardown, which
                    # can hang forever in native code (USD stage detach / plugin
                    # unload) AFTER a fully successful run. Results are already on
                    # disk, so there is nothing left to do in this ephemeral
                    # subprocess. os._exit() terminates immediately and does NOT
                    # run the `finally: scene_env.close()` below — that is
                    # intentional; the OS reclaims GPU/file/memory resources on
                    # process exit. Without this, a hung close() blocks the parent
                    # join() until the global eval timeout fires (exit 124) and the
                    # collected results are thrown away.
                    print("[INFO] Results flushed; hard-exiting worker to skip Isaac Sim teardown.")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)
            finally:
                scene_env.close()
                try:
                    import gc
                    del scene_env
                    gc.collect()
                except Exception:
                    pass

        return episodes
    
    def _resolve_max_episode_time_s(
        self,
        episodes: list[EpisodeMetrics],
        max_steps: int,
    ) -> float | None:
        """Resolve the physical time budget (seconds) for scoring.

        Priority:
          1. Explicit ``config.max_episode_time_s``
          2. ``step_dt`` recorded in episode extras × ``max_steps``
          3. None — scorer falls back to step-based normalization
        """
        if self.config.max_episode_time_s is not None:
            return self.config.max_episode_time_s

        step_dt: float | None = None
        for ep in episodes:
            step_dt = ep.extra.get("step_dt")
            if step_dt is not None:
                break

        if step_dt is not None:
            return max_steps * step_dt

        return None

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
            "task_type": self.config.task_type,
            "scoring_version": self.config.scoring_version,
            "scenes": scenes,
            "env_scenes": self.config.env_scenes,
            "seeds": self.config.seeds,
            "num_episodes": self.config.num_episodes,
            "max_episode_steps": self.config.max_episode_steps,
            "max_episode_time_s": self.config.max_episode_time_s,
            "total_episodes_run": num_combos * len(self.config.seeds) * self.config.num_episodes,
            "elapsed_seconds": elapsed,
            "config": self.config.to_dict(),
        }

