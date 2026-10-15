# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Deterministic expansion of a manifest into episode jobs and shards."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
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
    instruction: str = ""
    episode_length_s: float = 0.0


@dataclass(frozen=True)
class Shard:
    """A vectorized batch. Shards never cross a task, scene, or variant."""

    task_id: str
    scene_id: str
    composer: str
    variant: str
    pose_jitter_m: float
    jobs: tuple[EpisodeJob, ...]
    pack: str = ""
    """Stage signature when jobs of several tasks share one environment. Empty otherwise."""
    mixed: bool = False
    """The jobs may differ in surface and objects. The environment is built for these jobs, env by env."""

    @property
    def group_key(self) -> tuple[str, str, str]:
        if self.pack:
            return ("pack", self.pack, self.variant)
        return (self.task_id, self.scene_id, self.variant)


def expand(manifest: BenchmarkManifest, episodes: int | None = None) -> list[EpisodeJob]:
    """Expand in a fixed order: task, scene, variant, episode.

    ``episodes`` is the repeat count from the eval config. Every task uses that count.
    """
    count = 1 if episodes is None else int(episodes)
    if count < 1:
        raise ValueError("episodes must be >= 1")
    jobs: list[EpisodeJob] = []
    for task in manifest.tasks:
        for scene_id, composer in _scenes(task.scenes):
            for variant in task.variants:
                for index in range(count):
                    instruction_id = task.instruction or f"{task.instruction_pool}#{index}"
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
                            instruction=task.instruction,
                        )
                    )
    return jobs


def make_shards(
    jobs: list[EpisodeJob],
    shard_size: int,
    pack_key: Callable[[EpisodeJob], str] | None = None,
    mixed: bool = False,
) -> list[Shard]:
    """Pack jobs into fixed-size shards inside each group.

    A group is one (task, scene, variant). With ``pack_key`` it is every job that
    returns the same key and variant, whatever its task, so one environment serves
    them all. Group order follows the first job of each group, and job order
    inside a group follows expansion order, so neither depends on GPU count.

    With ``mixed`` the jobs of a group need not build the same stage. A group of ``n`` jobs
    becomes ``ceil(n / shard_size)`` shards of nearly equal size, and every shard is its own
    environment, because its objects differ from the others'.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    if mixed and pack_key is None:
        raise ValueError("mixed shards need a pack key")
    if pack_key is not None:
        return _make_packed_shards(jobs, shard_size, pack_key, mixed)
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


def _make_packed_shards(
    jobs: list[EpisodeJob], shard_size: int, pack_key: Callable[[EpisodeJob], str], mixed: bool = False
) -> list[Shard]:
    groups: dict[tuple[str, str], list[EpisodeJob]] = {}
    for job in jobs:
        groups.setdefault((pack_key(job), job.variant), []).append(job)
    shards: list[Shard] = []
    for (signature, _variant), group in groups.items():
        if mixed:
            shards.extend(_pack_balanced(group, shard_size, signature))
        else:
            shards.extend(_pack(group, shard_size, pack=signature))
    return shards


def _pack_balanced(jobs: list[EpisodeJob], shard_size: int, key: str) -> list[Shard]:
    """Split a group into the fewest shards of at most ``shard_size`` jobs, sizes within one of each other."""
    count = -(-len(jobs) // shard_size)
    base, extra = divmod(len(jobs), count)
    shards = []
    start = 0
    for index in range(count):
        chunk = tuple(jobs[start : start + base + (1 if index < extra else 0)])
        start += len(chunk)
        first = chunk[0]
        shards.append(
            Shard(
                task_id=first.task_id,
                scene_id=first.scene_id,
                composer=first.composer,
                variant=first.variant,
                pose_jitter_m=first.pose_jitter_m,
                jobs=chunk,
                # Each shard has its own objects, so no two share an environment.
                pack=f"{key}#{index}",
                mixed=True,
            )
        )
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


def _pack(jobs: list[EpisodeJob], shard_size: int, pack: str = "") -> list[Shard]:
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
                pack=pack,
            )
        )
    return shards


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed(job_id: str, salt: str) -> int:
    raw = hashlib.sha256(f"{job_id}|{salt}".encode()).digest()
    value = int.from_bytes(raw[:8], "big") % (2**31 - 1)
    return value or 1
