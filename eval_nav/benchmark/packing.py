# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Group episodes that build the same Isaac stage so they share one environment.

Building a scene (USD load, convex cooking, camera graph) costs far more than
stepping it. Tasks that differ only in instruction, success test, or object
poses can run side by side as separate environments of one vectorized stage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .expand import EpisodeJob
from .manifest import BenchmarkManifest

_ANCHOR = "anchor"


def scene_signature(bundle_dir: str | Path, task_id: str, scene_id: str) -> str:
    """Digest of everything in a scene file that decides how the stage is built.

    Poses, instructions, and success tests are left out: the environment writes
    them per episode after the stage exists. Anchors are left out because they
    are never spawned. Placement order stays in because it orders sensor filters.

    Raises:
        FileNotFoundError: The bundle has no scene file for this task and scene.
    """
    path = Path(bundle_dir) / "scenes" / f"{task_id}-{scene_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"benchmark has no scene {task_id}-{scene_id} under {bundle_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {
        "background": payload.get("background"),
        "surface": payload.get("surface"),
        "surface_pose": payload.get("surface_pose"),
        "placements": [
            [item.get("slot"), item.get("usd"), item.get("scale"), bool(item.get("kinematic"))]
            for item in payload.get("placements", [])
            if str(item.get("kind")) != _ANCHOR
        ],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def pack_key_for(manifest: BenchmarkManifest) -> Callable[[EpisodeJob], str] | None:
    """Pack key function for a manifest, or ``None`` when it does not pack.

    A key depends only on the bundle, never on the number of GPUs.
    """
    if manifest.pack_by == "none":
        return None
    if manifest.bundle_dir is None:
        raise ValueError("pack_by needs the bundle directory that holds the scene files")
    bundle_dir = manifest.bundle_dir
    cache: dict[tuple[str, str], str] = {}

    def key(job: EpisodeJob) -> str:
        pair = (job.task_id, job.scene_id)
        if pair not in cache:
            cache[pair] = scene_signature(bundle_dir, *pair)
        return cache[pair]

    return key
