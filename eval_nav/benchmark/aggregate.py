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
    records: list[dict[str, Any]],
    *,
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write results.json, summary.txt, and evaluation_result.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record["job_id"])
    summary = _summary(score, metrics, report)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps({"records": ordered, "metrics": metrics.to_dict(), "report": report}, indent=2),
        encoding="utf-8",
    )
    result = {
        "score": score,
        "summary": summary,
        "metadata": {
            **metadata,
            "total_episodes": metrics.total_episodes,
            "success_rate": metrics.success_rate,
            "formula": report.get("formula"),
            "time_budget_s": report.get("time_budget_s"),
            "sparc_smooth": report.get("sparc_smooth"),
            "sparc_jerky": report.get("sparc_jerky"),
            "weights": report.get("weights"),
            "task_rates": report.get("task_rates", {}),
            "tasks": report.get("tasks", {}),
            "episodes": report.get("episodes", []),
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
        extra={"task_id": record.get("task_id", "default"), "sparc": record.get("sparc")},
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
                "success": row.get("success", record.get("success")),
                "failed": record.get("failed"),
                "timeout": record.get("timeout"),
                "steps": row.get("steps", record.get("steps")),
                "completion_time_s": row.get("completion_time_s"),
                "sparc": row.get("sparc"),
                "speed": row.get("speed"),
                "smoothness": row.get("smoothness"),
                "trajectory_hash": record.get("trajectory_hash"),
            }
        )
    return merged


def _summary(score: float, metrics: AggregateMetrics, report: dict[str, Any]) -> str:
    """Human-readable report printed at the end of a brain evaluation."""
    lines = [
        f"score: {_num(score)}",
        f"formula: {report.get('formula', '')}",
        f"episodes: {metrics.total_episodes}",
        f"successes: {metrics.successful_episodes}",
        f"failures: {metrics.failed_episodes}",
        f"timeouts: {metrics.timeout_episodes}",
        f"success_rate: {_num(metrics.success_rate)}",
        f"time_budget_s: {_num(report.get('time_budget_s'))}",
        f"sparc_smooth: {_num(report.get('sparc_smooth'))}",
        f"sparc_jerky: {_num(report.get('sparc_jerky'))}",
        "weights: speed 0.70, smoothness 0.30, success_floor 0.30, quality 0.70",
    ]
    tasks = report.get("tasks") or {}
    for task_id, task in sorted(tasks.items()):
        lines.extend(
            [
                f"task {task_id}:",
                f"  episodes: {task.get('episodes')}",
                f"  successes: {task.get('successes')}",
                f"  success_rate: {_num(task.get('success_rate'))}",
                f"  speed: {_num(task.get('speed'))}",
                f"  smoothness: {_num(task.get('smoothness'))}",
                f"  quality: {_num(task.get('quality'))}",
                f"  task_score: {_num(task.get('task_score'))}",
            ]
        )
    lines.append("episodes:")
    for row in report.get("episodes") or []:
        lines.append(
            "  "
            + " ".join(
                [
                    str(row.get("job_id")),
                    f"task={row.get('task_id')}",
                    f"scene={row.get('scene_id')}",
                    f"variant={row.get('variant')}",
                    f"success={row.get('success')}",
                    f"steps={row.get('steps')}",
                    f"time_s={_num(row.get('completion_time_s'))}",
                    f"sparc={_num(row.get('sparc'))}",
                    f"speed={_num(row.get('speed'))}",
                    f"smoothness={_num(row.get('smoothness'))}",
                ]
            )
        )
    return "\n".join(lines)


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"
