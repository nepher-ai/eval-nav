# Nepher Navigation Evaluation Framework

Standardized, reproducible evaluation framework for IsaacLab navigation environments.

## Purpose

This public evaluation repository is required for the **Nepher subnet** (Bittensor Subnet 49). Validators clone this repository to evaluate miner solutions during tournament evaluation phases. It ensures:

- **Standardized scoring**: Consistent evaluation across all validators
- **Reproducible results**: Fixed seeds and deterministic execution
- **Public transparency**: Open-source evaluation logic for miner verification
- **EnvHub integration**: Evaluates across standardized Nepher EnvHub benchmark environments

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

See [config examples](configs/) for reference configurations.

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
scoring_version: "v1"  # Scoring version (see config examples)

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

## Isaac Sim Options

```bash
# Headless mode (default for validators)
python scripts/evaluate.py --config config.yaml --headless

# Enable cameras
python scripts/evaluate.py --config config.yaml --enable_cameras

# Custom experience file
python scripts/evaluate.py --config config.yaml --experience isaaclab.python.headless.kit
```

## Troubleshooting

**Environment not found:**
```bash
cd source/task-your-nav
pip install -e .
```

**Policy not found:**
- Set `policy_path: null` to use random actions
- Or provide explicit path to `.pt` file

## License

Nepher License