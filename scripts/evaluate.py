#!/usr/bin/env python3
# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Command-line interface for navigation evaluation.

Launch Isaac Sim Simulator first.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser(
    description="Evaluate IsaacLab navigation environments",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to evaluation configuration YAML file",
)

parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress console output",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from eval_nav import EvalConfig, EvaluationReporter, NavigationEvaluator


def main():
    """Main entry point for evaluation CLI.
    
    Returns:
        dict: Dictionary containing core evaluation results with keys:
            - score: float (final evaluation score)
            - log_dir: str (path to the run directory where results are saved)
            - metadata: dict (evaluation metadata as JSON-serializable dict)
            - summary: str (reporter's human-readable summary)
    """
    try:
        config = EvalConfig.from_yaml(args_cli.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
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
    
    evaluator = NavigationEvaluator(config, checkpoint_path=config.policy_path)
    
    if config.policy_path:
        print(f"[INFO] Policy checkpoint specified: {config.policy_path}")
        print("[INFO] Policy will be loaded when first environment is created")
    else:
        print("[INFO] No policy checkpoint specified, using random actions")
    
    results = evaluator.evaluate(policy=None)
    reporter = EvaluationReporter(results)
    
    log_json_path = run_dir / "results.json"
    log_summary_path = run_dir / "summary.txt"
    
    reporter.save_json(log_json_path)
    reporter.save_summary(log_summary_path)
    
    config_path = run_dir / "config.yaml"
    config.log_dir = original_log_dir
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    if not args_cli.quiet:
        reporter.print_summary()
        print(f"\nResults saved to log directory: {run_dir}")
        print(f"  - JSON: {log_json_path}")
        print(f"  - Summary: {log_summary_path}")
        print(f"  - Config: {config_path}")
        print(f"  - NumPy state logs: {run_dir}/*.npy")
    
    if results.get("status") != "SUCCESS":
        sys.exit(1)
    
    result = {
        "score": results.get("score"),
        "log_dir": str(run_dir),
        "metadata": results.get("metadata", {}),
        "summary": reporter.generate_summary(),
    }
    
    result_json_path = "evaluation_result.json"
    try:
        with open(result_json_path, "w") as f:
            json.dump(result, f, indent=2)
        if not args_cli.quiet:
            print(f"  - Result: {result_json_path}")
    except IOError as e:
        print(f"[WARNING] Failed to save result JSON: {e}", file=sys.stderr)
    
    return result


if __name__ == "__main__":
    try:
        result = main()
        print(f"\n[INFO] Evaluation result: {result}")
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()

