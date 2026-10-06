# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Benchmark manifest loaded from a phase bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Variant:
    """One pose distribution inside a scene."""

    name: str
    pose_jitter_m: float = 0.0


@dataclass
class SceneRequest:
    """A composed scene, or a count of seeds of one composer."""

    composer: str
    count: int = 1
    scene_id: str | None = None


@dataclass
class TaskRequest:
    """Episodes of one task family."""

    task_id: str
    scenes: list[SceneRequest]
    variants: list[Variant]
    instruction_pool: str
    episodes: int


@dataclass
class BenchmarkManifest:
    """Hidden phase benchmark. The shard size is pinned and does not follow GPU count."""

    version: str
    shard_size: int
    seed_salt: str
    tasks: list[TaskRequest] = field(default_factory=list)


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load ``benchmark.yaml``."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return manifest_from_dict(data)


def manifest_from_dict(data: dict[str, Any]) -> BenchmarkManifest:
    """Build a manifest from a parsed mapping."""
    tasks = [_task(item) for item in data["tasks"]]
    manifest = BenchmarkManifest(
        version=str(data["version"]),
        shard_size=int(data["shard_size"]),
        seed_salt=str(data["seed_salt"]),
        tasks=tasks,
    )
    if manifest.shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    return manifest


def _task(item: dict[str, Any]) -> TaskRequest:
    scenes = []
    for scene in item["scenes"]:
        if isinstance(scene, str):
            scenes.append(SceneRequest(composer=scene, count=1, scene_id=scene))
        else:
            scenes.append(
                SceneRequest(
                    composer=str(scene["composer"]),
                    count=int(scene.get("count", 1)),
                    scene_id=scene.get("scene_id"),
                )
            )
    variants = [
        Variant(name=str(variant["name"]), pose_jitter_m=float(variant.get("pose_jitter_m", 0.0)))
        for variant in item["variants"]
    ]
    return TaskRequest(
        task_id=str(item["task_id"]),
        scenes=scenes,
        variants=variants,
        instruction_pool=str(item["instruction_pool"]),
        episodes=int(item["episodes"]),
    )
