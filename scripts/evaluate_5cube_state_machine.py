#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Franka 5-cube scripted state machine.")
    parser.add_argument("--isaaclab-root", default="/home/a0loshi1/robotics/IsaacLab")
    parser.add_argument("--script", default="/home/a0loshi1/robotics/IsaacLab/scripts/custom/run_5cube_state_machine_vec.py")
    parser.add_argument("--task", default="Isaac-Stack-5Cube-Franka-IK-Rel-v0")
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=9000)
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--log-name", default="scripted_5cube_eval")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    isaaclab_root = Path(args.isaaclab_root)
    script = Path(args.script)

    if not isaaclab_root.exists():
        raise SystemExit(f"IsaacLab root does not exist: {isaaclab_root}")

    if not script.exists():
        raise SystemExit(f"State-machine script does not exist: {script}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = isaaclab_root / "logs" / "franka-5cube-scripted-eval" / f"eval_run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_log = log_dir / "raw_state_machine.log"
    results_json = log_dir / "results.json"
    summary_txt = log_dir / "summary.txt"

    cmd = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(script),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_steps",
        str(args.max_steps),
    ]

    if args.headless:
        cmd.append("--headless")

    env = dict(**__import__("os").environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("[INFO] Running scripted 5-cube evaluation command:")
    print(" ".join(cmd))
    print(f"[INFO] Raw log: {raw_log}")

    start = time.time()

    with raw_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(isaaclab_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )

        assert proc.stdout is not None
        captured = []
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            captured.append(line)

        ret = proc.wait()

    elapsed = time.time() - start
    text = "".join(captured)

    if ret != 0:
        raise SystemExit(f"State-machine script exited with code {ret}. See: {raw_log}")

    results_match = re.search(r"Env results:\s*(\[.*?\])", text, re.S)
    steps_match = re.search(r"Env steps:\s*(\[.*?\])", text, re.S)

    if not results_match or not steps_match:
        raise SystemExit(f"Could not parse Env results / Env steps from log: {raw_log}")

    results = ast.literal_eval(results_match.group(1))
    steps = ast.literal_eval(steps_match.group(1))

    if len(results) != len(steps):
        raise SystemExit("Parsed results and steps have different lengths.")

    n_total = len(results)
    n_success = sum(bool(x) for x in results)
    n_failed = n_total - n_success
    success_rate = n_success / n_total if n_total else 0.0

    step_dt = 0.05
    success_steps = [s for s, ok in zip(steps, results) if ok]
    failed_steps = [s for s, ok in zip(steps, results) if not ok]

    mean_success_steps = mean(success_steps) if success_steps else None
    median_success_steps = median(success_steps) if success_steps else None
    std_success_steps = pstdev(success_steps) if len(success_steps) > 1 else 0.0

    # Simple normalized score: success rate minus a small time penalty.
    # Keeps score close to success_rate but rewards faster solutions.
    max_steps = args.max_steps
    mean_time_fraction = (mean_success_steps / max_steps) if mean_success_steps else 1.0
    score = max(0.0, min(1.0, success_rate * (1.0 - 0.2 * mean_time_fraction)))

    report = {
        "status": "SUCCESS",
        "score": round(score, 4),
        "log_dir": str(log_dir),
        "metadata": {
            "evaluation_type": "scripted_state_machine_vector_eval",
            "task_name": args.task,
            "controller": str(script),
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "step_dt": step_dt,
            "elapsed_seconds": elapsed,
            "raw_log": str(raw_log),
        },
        "aggregate": {
            "total_episodes": n_total,
            "successful_episodes": n_success,
            "failed_episodes": n_failed,
            "timeout_or_failed_episodes": n_failed,
            "success_rate": success_rate,
            "mean_success_steps": mean_success_steps,
            "median_success_steps": median_success_steps,
            "std_success_steps": std_success_steps,
            "mean_success_time_s": mean_success_steps * step_dt if mean_success_steps else None,
            "median_success_time_s": median_success_steps * step_dt if median_success_steps else None,
            "failed_steps": failed_steps,
        },
        "per_env": [
            {"env_index": i, "success": bool(ok), "steps": int(step), "time_s": float(step * step_dt)}
            for i, (ok, step) in enumerate(zip(results, steps))
        ],
    }

    summary = f"""
============================================================
Franka 5-Cube Scripted Controller Evaluation Summary
============================================================

Status: SUCCESS
Final Score: {report["score"]:.4f}

Task:
  {args.task}

Controller:
  {script}

Aggregate Metrics:
------------------------------------------------------------
  Total Envs/Episodes: {n_total}
  Successful: {n_success}
  Failed/Timeout: {n_failed}
  Success Rate: {success_rate * 100:.2f}%
  Mean Success Steps: {mean_success_steps:.2f}
  Median Success Steps: {median_success_steps:.2f}
  Std Success Steps: {std_success_steps:.2f}
  Mean Success Time: {(mean_success_steps * step_dt if mean_success_steps else 0):.2f} s
  Max Steps: {args.max_steps}
  Step dt: {step_dt:.4f} s
  Elapsed Wall Time: {elapsed:.2f} s

Files:
------------------------------------------------------------
  Raw Log: {raw_log}
  JSON: {results_json}
  Summary: {summary_txt}

Interpretation:
------------------------------------------------------------
  This evaluated the scripted Franka 5-cube state-machine controller.
  It did NOT use random actions, robomimic, or RSL-RL checkpoint loading.

============================================================
""".strip()

    results_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_txt.write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"\n[INFO] Results saved to: {log_dir}")
    print(f"[INFO] JSON: {results_json}")
    print(f"[INFO] Summary: {summary_txt}")


if __name__ == "__main__":
    main()
