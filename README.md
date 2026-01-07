# Navigation Evaluation Framework

**A minimal but strong evaluation system for IsaacLab navigation environments.**

## Overview

This framework provides standardized, reproducible evaluation for navigation tasks in IsaacLab. It implements a fixed evaluation campaign system with deterministic execution, comprehensive metrics, and clear reporting.

## Features

### ✅ Core Evaluation Capability
- Load student-provided IsaacLab navigation environments
- Run fixed evaluation campaigns with:
  - Predefined envs-nav scenes
  - Fixed random seeds
  - Fixed number of episodes
  - Fixed time horizon per episode
- Fully automatic, non-interactive execution
- Deterministic results

### ✅ Minimal but Strong Config System
- Select target Gym/envs-nav environment
- Configure scenes/environments list
- Set seeds & episode count
- Define scoring version (MVP = V1 only)
- Basic evaluation parameters

### ✅ MVP Scoring System (V1)
- Focus on fundamental navigation capability
- Metrics:
  - Success rate (70% weight)
  - Normalized time-to-completion (30% weight)
- Clear, simple final score normalized to [0, 1]
- Failures heavily penalized but well-defined

### ✅ Essential Metric Collection
- Per-episode data:
  - Success/fail status
  - Steps to completion
  - Timeout indicator
  - Basic performance stats
- Aggregate summary:
  - Overall success rate
  - Mean completion time (successful episodes)
  - Final score

### ✅ Basic Failure Handling
- Evaluation always produces structured outcome
- States: `SUCCESS`, `ENV_ERROR`, `EVAL_ERROR`, `TIMEOUT`
- Each includes short explanation and lightweight log output
- No silent failures

### ✅ Reproducibility Baseline
- Pinned IsaacLab version expectation
- Fixed seeds
- Same configured scenes for everyone
- Stable output format
- Runtime metadata recorded

### ✅ Output & Reporting
- **Machine-readable**: JSON report with status, score, metrics, episode stats, metadata
- **Human-readable**: Text summary with performance interpretation

## Installation

```bash
cd source/nav-eval
pip install -e .
```

## Quick Start

### 1. Create Evaluation Config

Create a YAML configuration file (see `configs/` for examples):

```yaml
task_name: "Nepher-Animal-WaypointNav-Envs-Play-v0"

scenes:
  - 0
  - 1
  - 2

seeds:
  - 42
  - 123

num_episodes: 10
scoring_version: "v1"

env_config:
  nav_env_id: "waypoint-benchmark-v1"
  nav_scene: 0
```

### 2. Run Evaluation

```bash
python scripts/evaluate.py --config configs/task-animal-nav.yaml --output-dir results/
```

### 3. View Results

Results are saved in two formats:
- `results.json`: Machine-readable JSON report
- `summary.txt`: Human-readable text summary

## Example Configurations

### task-animal-nav

Evaluates the ANYmal B waypoint navigation environment:

```bash
python scripts/evaluate.py --config configs/task-animal-nav.yaml --output-dir results/animal-nav/
```

### task-leatherback-waypointnav

Evaluates the Leatherback waypoint navigation environment:

```bash
python scripts/evaluate.py --config configs/task-leatherback-waypointnav.yaml --output-dir results/leatherback/
```

## Configuration Reference

### Required Fields

- `task_name`: Gymnasium task name (e.g., `"Nepher-Animal-WaypointNav-Envs-Play-v0"`)

### Optional Fields

- `scenes`: List of scene IDs to evaluate (default: `[0]`)
- `seeds`: List of random seeds (default: `[42]`)
- `num_episodes`: Episodes per scene-seed combination (default: `10`)
- `max_episode_steps`: Maximum steps per episode (default: `None` = use env default)
- `scoring_version`: Scoring version, MVP supports `"v1"` only (default: `"v1"`)
- `env_config`: Environment-specific configuration dict
- `timeout_seconds`: Maximum wall-clock time for evaluation (default: `None`)

## Output Format

### JSON Report Structure

```json
{
  "status": "SUCCESS",
  "score": 0.85,
  "metrics": {
    "total_episodes": 90,
    "successful_episodes": 78,
    "failed_episodes": 12,
    "timeout_episodes": 5,
    "success_rate": 0.8667,
    "mean_completion_time": 245.3,
    "std_completion_time": 45.2,
    "mean_steps": 280.5,
    "std_steps": 120.3
  },
  "episodes": [...],
  "metadata": {...}
}
```

### Status Values

- `SUCCESS`: Evaluation completed successfully
- `ENV_ERROR`: Environment failed to load
- `EVAL_ERROR`: Runtime error during evaluation
- `TIMEOUT`: Evaluation exceeded time limit

## Scoring System (V1)

The V1 scoring system computes a final score in [0, 1] range:

```
score = 0.7 * success_rate + 0.3 * time_component
```

Where:
- `success_rate`: Fraction of successful episodes [0, 1]
- `time_component`: Normalized time-to-completion for successful episodes [0, 1]
  - Faster completion = higher time_component
  - Episodes exceeding max_normalized_time get 0

## MVP Non-Goals

The following are **not** included in the MVP (intentionally kept lean):

- ❌ V2 multi-factor scoring
- ❌ Safety or robustness metrics
- ❌ Multi-objective benchmarking
- ❌ UI
- ❌ Upload handling
- ❌ Docker orchestration
- ❌ Visual analytics
- ❌ Leaderboard system

## Development

### Project Structure

```
nav-eval/
├── nav_eval/
│   ├── __init__.py
│   ├── config.py          # Configuration system
│   ├── evaluator.py       # Core evaluation runner
│   ├── scorer.py          # V1 scoring system
│   ├── metrics.py         # Metric collection
│   ├── reporter.py        # Output generation
│   └── errors.py          # Failure handling
├── scripts/
│   └── evaluate.py        # CLI entry point
├── configs/
│   ├── task-animal-nav.yaml
│   └── task-leatherback-waypointnav.yaml
├── setup.py
├── pyproject.toml
└── README.md
```

## License

BSD-3-Clause

## Authors

Nepher Team

