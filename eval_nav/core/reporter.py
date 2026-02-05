# Copyright (c) 2026, Nepher AI
# All rights reserved.
#
# SPDX-License-Identifier: Proprietary

"""Reporting and output generation for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationReporter:
    """Generates machine-readable and human-readable evaluation reports."""
    
    def __init__(self, results: dict[str, Any]):
        """Initialize reporter.
        
        Args:
            results: Evaluation results dictionary.
        """
        self.results = results
    
    def save_json(self, output_path: str | Path) -> None:
        """Save machine-readable JSON report.
        
        Args:
            output_path: Path to save JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def generate_summary(self) -> str:
        """Generate human-readable text summary.
        
        Returns:
            Text summary string.
        """
        status = self.results.get("status", "UNKNOWN")
        score = self.results.get("score", 0.0)
        
        lines = []
        lines.append("=" * 60)
        lines.append("Navigation Evaluation Summary")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append(f"Status: {status}")
        lines.append(f"Final Score: {score:.4f} (normalized [0, 1])")
        lines.append("")
        
        if status != "SUCCESS":
            error = self.results.get("error", "Unknown error")
            lines.append(f"Error: {error}")
            if "details" in self.results:
                lines.append(f"Details: {json.dumps(self.results['details'], indent=2)}")
            lines.append("")
            return "\n".join(lines)
        
        metrics = self.results.get("metrics", {})
        if metrics:
            lines.append("Aggregate Metrics:")
            lines.append("-" * 60)
            lines.append(f"  Total Episodes: {metrics.get('total_episodes', 0)}")
            lines.append(f"  Successful: {metrics.get('successful_episodes', 0)}")
            lines.append(f"  Failed: {metrics.get('failed_episodes', 0)}")
            lines.append(f"  Timeouts: {metrics.get('timeout_episodes', 0)}")
            lines.append(f"  Success Rate: {metrics.get('success_rate', 0.0):.2%}")
            
            if metrics.get("mean_completion_time") is not None:
                lines.append(f"  Mean Completion Time: {metrics['mean_completion_time']:.2f} steps")
                if metrics.get("std_completion_time") is not None:
                    lines.append(f"  Std Completion Time: {metrics['std_completion_time']:.2f} steps")
            
            lines.append(f"  Mean Steps: {metrics.get('mean_steps', 0.0):.2f}")
            if metrics.get("std_steps") is not None:
                lines.append(f"  Std Steps: {metrics['std_steps']:.2f}")
            lines.append("")
        
        metadata = self.results.get("metadata", {})
        if metadata:
            lines.append("Evaluation Metadata:")
            lines.append("-" * 60)
            lines.append(f"  Task: {metadata.get('task_name', 'N/A')}")
            lines.append(f"  Scoring Version: {metadata.get('scoring_version', 'N/A')}")
            lines.append(f"  Scenes: {metadata.get('scenes', [])}")
            lines.append(f"  Seeds: {metadata.get('seeds', [])}")
            lines.append(f"  Episodes per Scene-Seed: {metadata.get('num_episodes', 0)}")
            lines.append(f"  Total Episodes Run: {metadata.get('total_episodes_run', 0)}")
            if "elapsed_seconds" in metadata:
                lines.append(f"  Elapsed Time: {metadata['elapsed_seconds']:.2f} seconds")
            lines.append("")
        
        lines.append("Interpretation:")
        lines.append("-" * 60)
        if score >= 0.8:
            lines.append("  Excellent performance! High success rate and fast completion.")
        elif score >= 0.6:
            lines.append("  Good performance. Room for improvement in success rate or speed.")
        elif score >= 0.4:
            lines.append("  Moderate performance. Significant improvements needed.")
        else:
            lines.append("  Poor performance. Fundamental navigation issues detected.")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_summary(self, output_path: str | Path) -> None:
        """Save human-readable text summary.
        
        Args:
            output_path: Path to save text file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.generate_summary()
        with open(output_path, "w") as f:
            f.write(summary)
    
    def print_summary(self) -> None:
        """Print summary to console."""
        print(self.generate_summary())

