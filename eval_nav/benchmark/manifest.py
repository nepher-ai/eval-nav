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
class SuccessSpec:
    """Success block stored as the benchmark wrote it. The task package interprets it."""

    body: dict[str, Any]

    @property
    def kind(self) -> str:
        if "kind" in self.body:
            return str(self.body["kind"])
        return str(self.body.get("op", ""))

    @property
    def side(self) -> str:
        return str(self.body.get("side", "left"))

    @property
    def roles(self) -> dict[str, Any]:
        raw = self.body.get("roles") or {}
        return {str(role): name for role, name in raw.items()}


@dataclass
class TaskRequest:
    """One authored task, or a composed family when ``scenes`` is a count."""

    task_id: str
    scenes: list[SceneRequest]
    variants: list[Variant]
    instruction_pool: str
    instruction: str = ""
    success: SuccessSpec | None = None


@dataclass
class BenchmarkManifest:
    """Hidden phase benchmark. The shard size is pinned and does not follow GPU count."""

    version: str
    shard_size: int
    seed_salt: str
    tasks: list[TaskRequest] = field(default_factory=list)
    pack_by: str = "none"
    """``none`` builds one environment per (task, scene, variant). ``assets`` runs every task that
    builds the same Isaac stage in one environment, up to ``shard_size`` episodes at a time."""
    bundle_dir: Path | None = None
    """Directory that holds ``benchmark.yaml`` and the scene files. Set by :func:`load_manifest`."""


PACK_MODES = ("none", "assets")


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load ``benchmark.yaml``."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return manifest_from_dict(data, bundle_dir=path.parent)


def manifest_from_dict(data: dict[str, Any], bundle_dir: str | Path | None = None) -> BenchmarkManifest:
    """Build a manifest from a parsed mapping."""
    tasks = [_task(item) for item in data["tasks"]]
    manifest = BenchmarkManifest(
        version=str(data["version"]),
        shard_size=int(data["shard_size"]),
        seed_salt=str(data["seed_salt"]),
        tasks=tasks,
        pack_by=str(data.get("pack_by", "none")),
        bundle_dir=None if bundle_dir is None else Path(bundle_dir),
    )
    if manifest.shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    if manifest.pack_by not in PACK_MODES:
        raise ValueError(f"pack_by must be one of {PACK_MODES}, got {manifest.pack_by!r}")
    return manifest


def _task(item: dict[str, Any]) -> TaskRequest:
    if "scene" in item:
        scene_id = str(item["scene"])
        scenes = [SceneRequest(composer=scene_id, count=1, scene_id=scene_id)]
        if "success" not in item:
            raise ValueError(f"task {item.get('task_id')} is missing success")
        if not item.get("instruction"):
            raise ValueError(f"task {item.get('task_id')} is missing instruction")
    else:
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
    raw_variants = item.get("variants")
    if raw_variants:
        variants = [
            Variant(name=str(variant["name"]), pose_jitter_m=float(variant.get("pose_jitter_m", 0.0)))
            for variant in raw_variants
        ]
    else:
        variants = [Variant(name="nominal")]
    success = _success(item.get("success"))
    return TaskRequest(
        task_id=str(item["task_id"]),
        scenes=scenes,
        variants=variants,
        instruction_pool=str(item.get("instruction_pool", "")),
        instruction=str(item.get("instruction", "")),
        success=success,
    )


def _success(block: dict[str, Any] | None) -> SuccessSpec | None:
    if block is None:
        return None
    if not isinstance(block, dict) or not block:
        raise ValueError("success must be a mapping")
    return SuccessSpec(body=dict(block))
