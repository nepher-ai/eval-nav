# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Expand an EnvHub bundle and run one Isaac process per GPU.

This process does not start Isaac Sim. ``scripts/evaluate.py`` calls it when
the config says ``runtime: brain``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval_nav.benchmark.aggregate import score_records, write_outputs
from eval_nav.benchmark.expand import expand, make_shards
from eval_nav.benchmark.manifest import load_manifest
from eval_nav.benchmark.scheduler import assign_groups, group_shards, plan_slots
from eval_nav.domain.config import EvalConfig


def main(argv: list[str] | None = None) -> None:
    """Run the brain evaluation described by ``--config``."""
    parser = argparse.ArgumentParser(description="Evaluate a brain submission")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None, help="Override the bundle's benchmark.yaml.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--result-path", default=None)
    args, _unknown = parser.parse_known_args(argv)

    config = EvalConfig.from_yaml(args.config)
    config.validate()
    manifest_path = Path(args.manifest) if args.manifest else _manifest_from_envhub(config)
    manifest = load_manifest(manifest_path)
    output_dir = Path(args.output_dir or config.log_dir or "logs/eval")
    records = run_workers(config, manifest, output_dir)
    max_steps = config.max_episode_steps or 1
    score, metrics, report = score_records(
        records,
        config.task_type,
        config.scoring_version,
        max_steps,
        max_episode_time_s=config.max_episode_time_s,
    )
    write_outputs(
        output_dir,
        score,
        metrics,
        records,
        report=report,
        metadata={"runtime": config.runtime, "version": manifest.version},
    )
    if args.result_path:
        target = Path(args.result_path)
        target.write_text((output_dir / "evaluation_result.json").read_text(encoding="utf-8"), encoding="utf-8")
    print((output_dir / "summary.txt").read_text(encoding="utf-8"))
    print((output_dir / "evaluation_result.json").read_text(encoding="utf-8"))


def run_workers(config: EvalConfig, manifest, output_dir: Path) -> list[dict]:
    """Expand, schedule, and collect JSONL records."""
    shards = make_shards(expand(manifest, config.num_episodes), manifest.shard_size)
    groups = group_shards(shards)
    placement = str((config.brain or {}).get("placement", "paired"))
    slots = plan_slots(placement)
    buckets = assign_groups(groups, len(slots))
    output_dir.mkdir(parents=True, exist_ok=True)
    active = [(slot, bucket) for slot, bucket in zip(slots, buckets) if bucket]
    replicas = max((slot.brain_index for slot, _bucket in active), default=0) + 1
    devices = ", ".join(slot.device_id for slot, _bucket in active)
    print(f"[INFO] Isaac workers: {len(active)} on GPU {devices}. Brain replicas: {replicas}.", flush=True)

    def run_bucket(index: int, slot, bucket: list) -> list[dict]:
        return _run_bucket(config, bucket, output_dir / f"gpu{index}.jsonl", slot)

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(active))) as pool:
        futures = [
            pool.submit(run_bucket, index, slot, bucket) for index, (slot, bucket) in enumerate(active)
        ]
        for future in futures:
            records.extend(future.result())
    return records


def _run_bucket(config: EvalConfig, bucket, output_path: Path, slot) -> list[dict]:
    """One Isaac process for every group assigned to this GPU."""
    payload = {
        "device": "cuda:0",
        "visible_device": slot.device_id,
        "socket": str(Path(config.brain["socket_dir"]) / f"brain-{slot.brain_index}.sock") if config.brain else "",
        "task_module": config.task_module,
        "task_name": config.task_name,
        "cfg_entry": (config.env_config or {}).get("cfg_entry"),
        "benchmark_env_id": config.benchmark_env_id,
        "open_loop_horizon": int(config.brain["open_loop_horizon"]) if config.brain else 1,
        "step_timeout_s": float(config.brain["step_timeout_s"]) if config.brain else 30,
        "max_steps": _step_cap(config),
        "runtime": config.runtime,
        "enable_cameras": config.enable_cameras,
        "policy_path": config.policy_path,
        "groups": [{"shards": [_shard_payload(shard) for shard in group]} for group in bucket],
    }
    group_path = output_path.with_suffix(".group.json")
    group_path.write_text(json.dumps(payload), encoding="utf-8")
    cli = Path(sys.argv[0]).resolve()
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = slot.device_id
    completed = subprocess.run(
        [sys.executable, str(cli), "--worker", "--group", str(group_path), "--output", str(output_path)],
        check=False,
        env=child_env,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"evaluation worker failed for GPU {slot.device_id} with code {completed.returncode}"
        )
    return [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _step_cap(config: EvalConfig) -> int:
    """Step cap for every group. The eval config sets it."""
    return int(config.max_episode_steps or 400)


def _shard_payload(shard) -> dict:
    return {
        "task_id": shard.task_id,
        "scene_id": shard.scene_id,
        "composer": shard.composer,
        "variant": shard.variant,
        "pose_jitter_m": shard.pose_jitter_m,
        "jobs": [
            {
                "job_id": job.job_id,
                "task_id": job.task_id,
                "scene_id": job.scene_id,
                "composer": job.composer,
                "variant": job.variant,
                "pose_jitter_m": job.pose_jitter_m,
                "instruction_id": job.instruction_id,
                "episode_index": job.episode_index,
                "seed": job.seed,
                "instruction": job.instruction,
            }
            for job in shard.jobs
        ],
    }


def _manifest_from_envhub(config: EvalConfig) -> Path:
    from nepher import load_env

    env = load_env(config.benchmark_env_id, category=config.category)
    path = Path(env.cache_path) / "benchmark.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"bundle {config.benchmark_env_id} has no benchmark.yaml")
    return path
