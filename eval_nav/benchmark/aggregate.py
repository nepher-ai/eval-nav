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
) -> tuple[float, AggregateMetrics, dict[str, float]]:
    """Score records sorted by job id. Returns the score, aggregates, and per-task rates."""
    ordered = sorted(records, key=lambda record: record["job_id"])
    episodes = [
        EpisodeMetrics(
            episode_id=index,
            scene=record.get("scene_id", ""),
            seed=int(record.get("seed", 0)),
            success=bool(record["success"]),
            steps=int(record["steps"]),
            timeout=bool(record.get("timeout", False)),
            extra={"task_id": record.get("task_id", "default")},
        )
        for index, record in enumerate(ordered)
    ]
    metrics = AggregateMetrics.from_episodes(episodes)
    scorer = get_scorer(task_type, scoring_version)
    score = scorer.compute_score(metrics, max_episode_steps, episodes)
    rates = getattr(scorer, "task_rates", {})
    return score, metrics, rates


def write_outputs(
    output_dir: Path,
    score: float,
    metrics: AggregateMetrics,
    records: list[dict[str, Any]],
    *,
    task_rates: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    """Write results.json, summary.txt, and evaluation_result.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record["job_id"])
    summary = (
        f"score: {score:.6f}\n"
        f"episodes: {metrics.total_episodes}\n"
        f"success_rate: {metrics.success_rate:.6f}\n"
    )
    for task_id, rate in sorted(task_rates.items()):
        summary += f"task {task_id}: {rate:.6f}\n"
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps({"records": ordered, "metrics": metrics.to_dict()}, indent=2),
        encoding="utf-8",
    )
    result = {
        "score": score,
        "summary": summary.strip(),
        "metadata": {**metadata, "task_rates": task_rates, "total_episodes": metrics.total_episodes},
    }
    (output_dir / "evaluation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
