# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Turn per-job records into the evaluation_result.json contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.scorers import get_scorer
from ..domain.metrics import AggregateMetrics, EpisodeMetrics


def score_records(
    records: list[dict[str, Any]],
    task_type: str,
    scoring_version: str,
    max_episode_steps: int,
    max_episode_time_s: float | None = None,
) -> tuple[float, AggregateMetrics, dict[str, Any]]:
    """Score records sorted by job id.

    Returns the score, the aggregate counts, and the scorer report. The report
    carries per-task speed, smoothness, and the per-episode rows.
    """
    ordered = sorted(records, key=lambda record: record["job_id"])
    episodes = [_episode(index, record) for index, record in enumerate(ordered)]
    metrics = AggregateMetrics.from_episodes(episodes)
    scorer = get_scorer(task_type, scoring_version)
    score = scorer.compute_score(
        metrics,
        max_episode_steps,
        episodes,
        max_episode_time_s=max_episode_time_s,
    )
    report = scorer.to_dict()
    report["episodes"] = _merge_episodes(ordered, report.get("episodes", []))
    return score, metrics, report


def write_outputs(
    output_dir: Path,
    score: float,
    metrics: AggregateMetrics,
    *,
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write the short summary, one analysis file, and the score contract.

    Per-task and per-episode rows live only in ``results.json``. The summary is the
    aggregate, and ``evaluation_result.json`` repeats that aggregate for the validator.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(score, metrics, report, metadata)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps({"metrics": metrics.to_dict(), "report": report}, indent=2),
        encoding="utf-8",
    )
    result = {
        "score": score,
        "log_version": 2,
        "summary": summary,
        "metadata": {
            **metadata,
            "total_episodes": metrics.total_episodes,
            "success_rate": metrics.success_rate,
            "formula": report.get("formula"),
            "time_budget_s": report.get("time_budget_s"),
            "weights": report.get("weights"),
        },
    }
    (output_dir / "evaluation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def _episode(index: int, record: dict[str, Any]) -> EpisodeMetrics:
    """One scored episode. Completion time [s] is set only after a success."""
    success = bool(record["success"])
    steps = int(record["steps"])
    completion = record.get("completion_time_s")
    if success and completion is None:
        completion = steps * float(record.get("control_dt_s") or 0.04)
    return EpisodeMetrics(
        episode_id=index,
        scene=record.get("scene_id", ""),
        seed=int(record.get("seed", 0)),
        success=success,
        steps=steps,
        timeout=bool(record.get("timeout", False)),
        completion_time=None if completion is None else float(completion),
        extra={
            "task_id": record.get("task_id", "default"),
            "sparc": record.get("sparc"),
            "progress": record.get("progress", 1.0 if success else 0.0),
            "path_length_m": record.get("path_length_m"),
            "mean_hand_speed_mps": record.get("mean_hand_speed_mps"),
        },
    )


def _merge_episodes(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the job identity to each scored row. Both lists follow job-id order."""
    merged = []
    for record, row in zip(records, rows):
        merged.append(
            {
                "job_id": record.get("job_id"),
                "task_id": record.get("task_id", row.get("task_id")),
                "scene_id": record.get("scene_id"),
                "variant": record.get("variant"),
                "episode_index": record.get("episode_index"),
                "seed": record.get("seed"),
                "instruction": record.get("instruction"),
                "success": row.get("success", record.get("success")),
                "failed": record.get("failed"),
                "timeout": record.get("timeout"),
                "steps": row.get("steps", record.get("steps")),
                "elapsed_s": record.get("elapsed_s"),
                "completion_time_s": row.get("completion_time_s"),
                "progress": row.get("progress"),
                "sparc": row.get("sparc"),
                "speed": row.get("speed"),
                "smoothness": row.get("smoothness"),
                "path_length_m": row.get("path_length_m"),
                "mean_hand_speed_mps": row.get("mean_hand_speed_mps"),
                "final_positions_m": record.get("final_positions_m"),
                "trajectory_hash": record.get("trajectory_hash"),
            }
        )
    return merged


def _summary(
    score: float,
    metrics: AggregateMetrics,
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Sectioned report in the same shape as the checkpoint evaluator summary."""
    lines = [
        _RULE,
        "Evaluation Summary",
        _RULE,
        "",
        "Status: SUCCESS",
        f"Final Score: {float(score):.4f} (normalized [0, 1])",
        "",
        "Aggregate Metrics:",
        _DASH,
        f"  Total Episodes: {metrics.total_episodes}",
        f"  Successful: {metrics.successful_episodes}",
        f"  Failed: {metrics.failed_episodes}",
        f"  Timeouts: {metrics.timeout_episodes}",
        f"  Success Rate: {float(metrics.success_rate):.2%}",
        f"  Mean Steps: {metrics.mean_steps:.2f}",
        f"  Std Steps: {metrics.std_steps:.2f}",
    ]
    if metrics.mean_completion_time is not None:
        lines.append(f"  Mean Completion Time: {metrics.mean_completion_time:.2f} s")
        if metrics.std_completion_time is not None:
            lines.append(f"  Std Completion Time: {metrics.std_completion_time:.2f} s")
    lines.extend(
        [
            f"  Time Budget: {_quantity(report.get('time_budget_s'), 's')}",
            f"  Pooled Score: {_decimal(report.get('pooled_score'))}",
            f"  Mean Path Length: {_quantity(report.get('mean_path_length_m'), 'm')}",
            f"  Mean Hand Speed: {_quantity(report.get('mean_hand_speed_mps'), 'm/s')}",
            f"  Mean SPARC: {_decimal(report.get('mean_sparc'))}",
            f"  Weights: {_weights(report)}",
            f"  Formula: {report.get('formula') or 'n/a'}",
            "",
        ]
    )
    if metadata:
        lines.extend(["Evaluation Metadata:", _DASH])
        for key, label in _METADATA_FIELDS:
            if key not in metadata or metadata[key] is None:
                continue
            value = metadata[key]
            if key == "max_episode_time_s":
                value = f"{float(value):.2f} s"
            elif key == "elapsed_seconds":
                value = f"{float(value):.2f} seconds"
            lines.append(f"  {label}: {value}")
        lines.append("")
    lines.extend(
        [
            "Interpretation:",
            _DASH,
            f"  {_interpretation(float(score))}",
            "",
            _RULE,
        ]
    )
    return "\n".join(lines)


_RULE = "=" * 60
_DASH = "-" * 60
_METADATA_FIELDS = (
    ("runtime", "Runtime"),
    ("task_name", "Task"),
    ("benchmark_env_id", "Benchmark"),
    ("scoring_version", "Scoring Version"),
    ("version", "Bundle Version"),
    ("num_episodes", "Episodes per Task"),
    ("max_episode_steps", "Max Steps"),
    ("max_episode_time_s", "Max Episode Time"),
    ("elapsed_seconds", "Elapsed Time"),
)


def _interpretation(score: float) -> str:
    if score >= 0.8:
        return "Excellent performance. High task scores and clean finishes."
    if score >= 0.6:
        return "Good performance. Room for improvement in success rate or speed."
    if score >= 0.4:
        return "Moderate performance. Several tasks are unfinished."
    return "Poor performance. Most tasks did not finish."


def _weights(report: dict[str, Any]) -> str:
    weights = report.get("weights") or {}
    if not weights:
        return "n/a"
    return ", ".join(f"{name} {float(value):.2f}" for name, value in weights.items())


def _decimal(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _quantity(value: Any, unit: str, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f} {unit}"


