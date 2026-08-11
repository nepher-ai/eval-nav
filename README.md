# Nepher Task Evaluation Framework

Evaluation harness for IsaacLab `task-*` projects on EnvHub benchmarks (navigation, manipulation, and more).

Used by validators on the **Nepher subnet** (Bittensor Subnet 49) for standardized, reproducible tournament scoring.

## Requirements

- IsaacLab 2.3+, Isaac Sim 5.1+
- `nepher`, the target `task-*` package, and its EnvHub assets

## Installation

```bash
pip install nepher
nepher download <env_ids...>

cd eval-nav && pip install -e .
cd ../<task-*> && pip install -e .
```

Configs for each campaign: [configs/](configs/).

## Usage

```bash
python scripts/evaluate.py --config configs/<task>.yaml --headless
```

Results are written under `log_dir` as `results.json`, `summary.txt`, and `config.yaml`.

```python
from eval_nav import EvalConfig, NavigationEvaluator, EvaluationReporter

config = EvalConfig.from_yaml("config.yaml")
results = NavigationEvaluator(config, checkpoint_path=config.policy_path).evaluate(policy=None)
EvaluationReporter(results).print_summary()
```

## License

Proprietary. See [LICENSE](LICENSE).
