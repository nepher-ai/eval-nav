# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Assign shard groups to GPUs. The shard list itself does not depend on GPU count."""

from __future__ import annotations

import os
import subprocess

from .expand import Shard


def visible_gpu_count() -> int:
    """GPUs this process may use, from ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi``."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and raw.strip() not in ("", "-1", "none"):
        return max(1, len([part for part in raw.split(",") if part.strip()]))
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1
    if completed.returncode != 0:
        return 1
    count = sum(1 for line in completed.stdout.splitlines() if line.startswith("GPU"))
    return count or 1


def group_shards(shards: list[Shard]) -> list[list[Shard]]:
    """One group per (task, scene, variant), preserving shard order."""
    groups: list[list[Shard]] = []
    for shard in shards:
        if not groups or groups[-1][0].group_key != shard.group_key:
            groups.append([shard])
        else:
            groups[-1].append(shard)
    return groups


def assign_groups(groups: list[list[Shard]], gpu_count: int) -> list[list[list[Shard]]]:
    """Round-robin groups across GPUs. Each bucket runs one group at a time."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    buckets: list[list[list[Shard]]] = [[] for _ in range(gpu_count)]
    for index, group in enumerate(groups):
        buckets[index % gpu_count].append(group)
    return buckets
