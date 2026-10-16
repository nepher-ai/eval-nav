#!/usr/bin/env python3
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Evaluate a tournament submission.

The config selects the path. ``runtime: brain`` loads the EnvHub bundle and
starts one Isaac process per GPU. Any other runtime starts Isaac in this
process and runs the checkpoint evaluator.

The per-GPU process is ``evaluate.py --worker``. It is not a separate command.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def _mode(argv: list[str]) -> str:
    """Return ``worker``, ``brain``, or ``checkpoint`` before Isaac starts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", default=None)
    args, _unknown = parser.parse_known_args(argv)
    if args.worker:
        return "worker"
    if not args.config:
        return "checkpoint"
    path = Path(args.config)
    if not path.is_file():
        return "checkpoint"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    except Exception:
        return "checkpoint"
    if data.get("runtime") == "brain":
        return "brain"
    return "checkpoint"


def _run_checkpoint() -> None:
    """Start Isaac Sim and evaluate an in-process checkpoint."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Evaluate IsaacLab navigation environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation configuration YAML file")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    parser.add_argument(
        "--result-path",
        type=str,
        default=None,
        help="Absolute path for evaluation_result.json output (default: cwd)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    # Lab 3 dropped the --enable_cameras flag. The launcher still reads the attribute.
    if _config_enables_cameras(args_cli.config):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from eval_nav import EvalConfig, EvaluationReporter, NavigationEvaluator

    try:
        result = _checkpoint_main(args_cli, EvalConfig, EvaluationReporter, NavigationEvaluator)
        print(f"\n[INFO] Evaluation result: {result}")
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        import traceback

        print(f"\n[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


def _config_enables_cameras(config_path: str | None) -> bool:
    """Return whether the eval YAML sets ``enable_cameras: true``."""
    if not config_path:
        return False
    path = Path(config_path)
    if not path.is_file():
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    except Exception:
        return False
    return bool(data.get("enable_cameras"))


def _checkpoint_main(args_cli, eval_config_cls, reporter_cls, evaluator_cls):
    try:
        config = eval_config_cls.from_yaml(args_cli.config)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    if not config.log_dir:
        raise ValueError("log_dir must be specified in config YAML")

    log_dir = Path(config.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = (Path.cwd() / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"eval_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    original_log_dir = config.log_dir
    config.log_dir = str(run_dir)

    evaluator = evaluator_cls(config, checkpoint_path=config.policy_path)

    if config.policy_path:
        print(f"[INFO] Policy checkpoint specified: {config.policy_path}")
        print("[INFO] Policy will be loaded when first environment is created")
    else:
        print("[INFO] No policy checkpoint specified, using random actions")

    results = evaluator.evaluate(policy=None)
    reporter = reporter_cls(results)

    log_json_path = run_dir / "results.json"
    log_summary_path = run_dir / "summary.txt"

    reporter.save_json(log_json_path)
    reporter.save_summary(log_summary_path)

    config_path = run_dir / "config.yaml"
    config.log_dir = original_log_dir
    with open(config_path, "w", encoding="utf-8", errors="replace") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    if not args_cli.quiet:
        reporter.print_summary()
        print(f"\nResults saved to log directory: {run_dir}")
        print(f"  - JSON: {log_json_path}")
        print(f"  - Summary: {log_summary_path}")
        print(f"  - Config: {config_path}")
        print(f"  - NumPy state logs: {run_dir}/*.npy")

    result = {
        "score": results.get("score", 0),
        "log_dir": str(run_dir),
        "metadata": results.get("metadata", {}),
        "summary": reporter.generate_summary(),
    }

    if args_cli.result_path:
        result_json_path = Path(args_cli.result_path)
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        result_json_path = Path("evaluation_result.json")
    try:
        with open(result_json_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if not args_cli.quiet:
            print(f"  - Result: {result_json_path}")
    except OSError as exc:
        print(f"[WARNING] Failed to save result JSON: {exc}", file=sys.stderr)

    if results.get("status") != "SUCCESS":
        sys.exit(1)

    return result


def main() -> None:
    mode = _mode(sys.argv[1:])
    if mode == "worker":
        from eval_nav.benchmark.worker import main as run_worker

        run_worker()
        return
    if mode == "brain":
        from eval_nav.benchmark.orchestrate import main as run_brain

        run_brain()
        return
    _run_checkpoint()


if __name__ == "__main__":
    main()
