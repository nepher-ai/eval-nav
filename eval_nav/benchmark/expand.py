# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Deterministic expansion of a manifest into episode jobs and shards."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .manifest import BenchmarkManifest, SceneRequest


@dataclass(frozen=True)
class EpisodeJob:
    """One episode. ``job_id`` and ``seed`` depend only on the manifest."""

    job_id: str
    task_id: str
    scene_id: str
    composer: str
    variant: str
    pose_jitter_m: float
    instruction_id: str
    episode_index: int
    seed: int


@dataclass(frozen=True)
class Shard:
    """A vectorized batch. Shards never cross a task, scene, or variant."""

    task_id: str
    scene_id: str
    composer: str
    variant: str
    pose_jitter_m: float
    jobs: tuple[EpisodeJob, ...]

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.task_id, self.scene_id, self.variant)


def expand(manifest: BenchmarkManifest) -> list[EpisodeJob]:
    """Expand in a fixed order: task, scene, variant, episode."""
    jobs: list[EpisodeJob] = []
    for task in manifest.tasks:
        for scene_id, composer in _scenes(task.scenes):
            for variant in task.variants:
                for index in range(task.episodes):
                    instruction_id = f"{task.instruction_pool}#{index}"
                    job_id = _digest(
                        "|".join(
                            (
                                manifest.version,
                                task.task_id,
                                scene_id,
                                variant.name,
                                instruction_id,
                                str(index),
                            )
                        )
                    )
                    jobs.append(
                        EpisodeJob(
                            job_id=job_id,
                            task_id=task.task_id,
                            scene_id=scene_id,
                            composer=composer,
                            variant=variant.name,
                            pose_jitter_m=variant.pose_jitter_m,
                            instruction_id=instruction_id,
                            episode_index=index,
                            seed=_seed(job_id, manifest.seed_salt),
                        )
                    )
    return jobs


def make_shards(jobs: list[EpisodeJob], shard_size: int) -> list[Shard]:
    """Pack jobs into fixed-size shards inside each (task, scene, variant) group."""
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    shards: list[Shard] = []
    group: list[EpisodeJob] = []
    for job in jobs:
        if group and (job.task_id, job.scene_id, job.variant) != (
            group[0].task_id,
            group[0].scene_id,
            group[0].variant,
        ):
            shards.extend(_pack(group, shard_size))
            group = []
        group.append(job)
    if group:
        shards.extend(_pack(group, shard_size))
    return shards


def _scenes(scenes: list[SceneRequest]) -> list[tuple[str, str]]:
    expanded = []
    for scene in scenes:
        if scene.scene_id is not None and scene.count == 1:
            expanded.append((scene.scene_id, scene.composer))
            continue
        for index in range(scene.count):
            expanded.append((f"{scene.composer}-{index}", scene.composer))
    return expanded


def _pack(jobs: list[EpisodeJob], shard_size: int) -> list[Shard]:
    shards = []
    for start in range(0, len(jobs), shard_size):
        chunk = tuple(jobs[start : start + shard_size])
        first = chunk[0]
        shards.append(
            Shard(
                task_id=first.task_id,
                scene_id=first.scene_id,
                composer=first.composer,
                variant=first.variant,
                pose_jitter_m=first.pose_jitter_m,
                jobs=chunk,
            )
        )
    return shards


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed(job_id: str, salt: str) -> int:
    raw = hashlib.sha256(f"{job_id}|{salt}".encode("utf-8")).digest()
    value = int.from_bytes(raw[:8], "big") % (2**31 - 1)
    return value or 1
