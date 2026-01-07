#!/usr/bin/env python3
# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line interface for navigation evaluation."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

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
    "--output-dir",
    type=str,
    default="./eval_results",
    help="Directory to save evaluation results (default: ./eval_results)",
)

parser.add_argument(
    "--policy",
    type=str,
    default=None,
    help="Path to policy checkpoint (optional, uses random actions if not provided)",
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

from nav_eval import EvalConfig, EvaluationReporter, NavigationEvaluator


def main():
    """Main entry point for evaluation CLI."""
    # Load configuration
    try:
        config = EvalConfig.from_yaml(args_cli.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    # Pass checkpoint path to evaluator for lazy loading
    evaluator = NavigationEvaluator(config, checkpoint_path=args_cli.policy)
    
    if args_cli.policy:
        print(f"[INFO] Policy checkpoint specified: {args_cli.policy}")
        print("[INFO] Policy will be loaded when first environment is created")
    
    results = evaluator.evaluate(policy=None)
    
    # Generate reports
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reporter = EvaluationReporter(results)
    
    # Save JSON report
    json_path = output_dir / "results.json"
    reporter.save_json(json_path)
    
    # Save text summary
    summary_path = output_dir / "summary.txt"
    reporter.save_summary(summary_path)
    
    # Print summary
    if not args_cli.quiet:
        reporter.print_summary()
        print(f"\nResults saved to: {output_dir}")
        print(f"  - JSON: {json_path}")
        print(f"  - Summary: {summary_path}")
    
    # Exit with error code if evaluation failed
    if results.get("status") != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # Close the app
        simulation_app.close()

