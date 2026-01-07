#!/usr/bin/env python3
# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line interface for navigation evaluation."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create argument parser and add AppLauncher args
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

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from eval_nav import EvalConfig, EvaluationReporter, NavigationEvaluator


def main():
    """Main entry point for evaluation CLI."""
    # Load configuration
    try:
        config = EvalConfig.from_yaml(args_cli.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save to log_dir from config (encapsulated in a run subdirectory)
    if not config.log_dir:
        raise ValueError("log_dir must be specified in config YAML")
    
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped run directory to encapsulate all results (including numpy files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"eval_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config.log_dir to point to run_dir so numpy files are saved there
    original_log_dir = config.log_dir
    config.log_dir = str(run_dir)
    
    # Run evaluation
    # Pass checkpoint path to evaluator for lazy loading
    evaluator = NavigationEvaluator(config, checkpoint_path=config.policy_path)
    
    if config.policy_path:
        print(f"[INFO] Policy checkpoint specified: {config.policy_path}")
        print("[INFO] Policy will be loaded when first environment is created")
    else:
        print("[INFO] No policy checkpoint specified, using random actions")
    
    results = evaluator.evaluate(policy=None)
    
    # Generate reports
    reporter = EvaluationReporter(results)
    
    # Save results in the run directory
    log_json_path = run_dir / "results.json"
    log_summary_path = run_dir / "summary.txt"
    
    reporter.save_json(log_json_path)
    reporter.save_summary(log_summary_path)
    
    # Save a copy of the config for reference
    config_path = run_dir / "config.yaml"
    # Restore original log_dir in config before saving
    config.log_dir = original_log_dir
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    # Print summary
    if not args_cli.quiet:
        reporter.print_summary()
        print(f"\nResults saved to log directory: {run_dir}")
        print(f"  - JSON: {log_json_path}")
        print(f"  - Summary: {log_summary_path}")
        print(f"  - Config: {config_path}")
        print(f"  - NumPy state logs: {run_dir}/*.npy")
    
    # Exit with error code if evaluation failed
    if results.get("status") != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close the app
        simulation_app.close()

