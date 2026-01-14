# eval-nav: Navigation Evaluation Framework for IsaacLab

Standardized, reproducible evaluation for IsaacLab navigation environments.

## Overview

`eval-nav` evaluates trained navigation policies across multiple envhub scenes with:

- **Deterministic execution**: Fixed seeds and reproducible results
- **V1 scoring**: Success rate (70%) + completion time (30%)
- **Multi-scene evaluation**: Test across different `env_id` and `scene` combinations
- **Structured output**: JSON results, text summaries, and per-episode state logs

## Requirements

- IsaacLab 2.3+
- Isaac Sim 5.1+
- envhub (nepher) installed (`pip install -e source/envhub`)
- Target environments pre-downloaded (via envhub)

## Installation

```bash
# 1. Install envhub (nepher) (required for navigation environments)
# See: source/envhub/README.md
cd source/envhub
pip install -e .

# 2. Pre-download target environments
# See available environments: nepher list
nepher download waypoint-benchmark-v1 waypoint-sample-v1

# 3. Install eval-nav
cd source/eval-nav
pip install -e .
```

For envhub (nepher) setup and CLI usage, see [envhub README](https://github.com/nepher-ai/envhub).

## Quick Start

### 1. Create Configuration

```yaml
# config.yaml
task_name: "Nepher-Leatherback-WaypointNav-Envhub-Play-v0"
task_module: "leatherbacknav"

env_scenes:
  - env_id: "waypoint-benchmark-v1"
    scene: 0
  - env_id: "waypoint-sample-v1"
    scene: 0

seeds: [42]
num_episodes: 10
max_episode_steps: 900
num_envs: 10
scoring_version: "v1"

log_dir: "logs/my-eval"
enable_logging: true
policy_path: "default"
```

### 2. Run Evaluation

```bash
python scripts/evaluate.py --config config.yaml --headless
```

### 3. View Results

```
logs/my-eval/eval_run_YYYYMMDD_HHMMSS/
├── results.json    # Machine-readable results
├── summary.txt     # Human-readable summary
├── config.yaml     # Evaluation configuration
└── data/          # Episode state logs (.npy)
```

## Configuration Reference

### Required

```yaml
task_name: "Nepher-Task-Name-v0"      # Gymnasium task ID
task_module: "modulename"             # Import module for env registration
num_envs: 10                          # Parallel environments
log_dir: "logs/eval"                  # Output directory
env_scenes:                           # Scene combinations
  - env_id: "waypoint-benchmark-v1"
    scene: 0
```

### Optional

```yaml
seeds: [42]                           # Random seeds (default: [42])
num_episodes: 10                      # Episodes per scene-seed (default: 10)
max_episode_steps: 900                # Max steps per episode (default: env default)
scoring_version: "v1"                 # Scoring version (default: "v1")
timeout_seconds: null                 # Evaluation timeout (default: none)
enable_logging: false                 # Save episode states (default: false)
policy_path: "default"                # Policy checkpoint path (default: none)
                                      # "default" = auto-detect from task module
                                      # null = random actions
```

**Total episodes:** `len(env_scenes) × len(seeds) × num_episodes`

## Output Format

### results.json

```json
{
  "status": "SUCCESS",
  "score": 0.9537,
  "metrics": {
    "total_episodes": 20,
    "successful_episodes": 19,
    "success_rate": 0.95,
    "mean_completion_time": 245.3,
    "std_completion_time": 45.2
  },
  "episodes": [...],
  "metadata": {...}
}
```

### summary.txt

```
Navigation Evaluation Summary
Status: SUCCESS
Final Score: 0.9537
Success Rate: 95.00%
Mean Completion Time: 245.30 steps
```

## Scoring (V1)

```
score = 0.7 × success_rate + 0.3 × time_component
```

- **Success rate** (70%): Fraction of episodes reaching goal
- **Time component** (30%): `1.0 - (mean_time / max_steps)` for successful episodes

## Example

**Config:** `configs/task-leatherback-waypointnav.yaml`

```yaml
task_name: "Nepher-Leatherback-WaypointNav-Envhub-Play-v0"
task_module: "leatherbacknav"
env_scenes:
  - env_id: "waypoint-benchmark-v1"
    scene: 0
  - env_id: "waypoint-sample-v1"
    scene: 0
seeds: [42]
num_episodes: 1
num_envs: 1
log_dir: "logs/leatherback-waypointnav"
policy_path: "default"
```

**Run:**

```bash
python scripts/evaluate.py --config configs/task-leatherback-waypointnav.yaml --headless
```

## Programmatic API

```python
from eval_nav import EvalConfig, NavigationEvaluator, EvaluationReporter

# Load and run
config = EvalConfig.from_yaml("config.yaml")
evaluator = NavigationEvaluator(config, checkpoint_path=config.policy_path)
results = evaluator.evaluate(policy=None)

# Generate reports
reporter = EvaluationReporter(results)
reporter.save_json("results.json")
reporter.save_summary("summary.txt")
reporter.print_summary()

# Access results
print(f"Score: {results['score']:.4f}")
print(f"Success Rate: {results['metrics']['success_rate']:.2%}")
```

## Policy Loading

```yaml
# Auto-detect from task module (looks for best_policy/best_policy.pt)
policy_path: "default"

# Explicit path
policy_path: "/path/to/policy.pt"

# Random actions
policy_path: null
```

## Isaac Sim Options

```bash
# Headless mode
python scripts/evaluate.py --config config.yaml --headless

# Enable cameras
python scripts/evaluate.py --config config.yaml --enable_cameras

# Custom experience file
python scripts/evaluate.py --config config.yaml --experience isaaclab.python.headless.kit
```

## Troubleshooting

### Environment not found
```bash
cd source/task-your-nav
pip install -e .
```

### Policy not found
- Set `policy_path: null` to use random actions
- Or provide explicit path to `.pt` file

## License
Nepher License