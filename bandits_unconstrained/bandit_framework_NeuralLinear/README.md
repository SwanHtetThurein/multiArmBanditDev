# Multi-Dimensional Bandit Experiment Framework

Modularized version of `new_global_Dynamic_Parameters.py`. The testing
structure (random problem generation, 9 tests x 20 runs x 6 noise levels,
per-round performance CSV) is unchanged; the recommendation algorithm is now
a swappable plug-in.

## Layout

```
bandit_framework/
├── run_experiment.py        # entry point (interactive or CLI flags)
├── experiment.py            # harness: problem/test generation, sweep loop, CSV
├── environment.py           # hidden optimal arms + noisy reward function
└── algorithms/
    ├── __init__.py          # registry (name -> class)
    ├── base.py              # RecommendationAlgorithm interface + ProblemConfig
    ├── dreamteam.py         # the original algorithm, logic preserved exactly
    └── random_baseline.py   # minimal example plug-in / performance floor
```

## Running

```bash
# Interactive prompts (like the original script)
python run_experiment.py

# Non-interactive
python run_experiment.py --bandits 9 --rounds 100
python run_experiment.py --bandits 12 --rounds 100 --seed 42
python run_experiment.py --bandits 9 --rounds 100 --algorithm random
python run_experiment.py --bandits 9 --rounds 50 --runs 5 --noise 0.0 0.5 1.0
```

Output CSV has the same schema as before —
`[rowid, runID, noise_level, round, performance]` — saved to
`Global results/` with the algorithm name in the filename, e.g.
`dreamteam_results_rounds100_dims9_tests9_<timestamp>.csv`.

## Adding a new recommendation algorithm

1. Create `algorithms/my_algo.py`:

```python
from .base import RecommendationAlgorithm

class MyAlgorithm(RecommendationAlgorithm):
    name = "myalgo"

    def choose(self, round_num):
        # return one arm index per bandit, e.g. using self.config.arm_counts,
        # self.config.bandit_types, self.initial_bias, self.total_rounds
        ...

    def update(self, arms_chosen, reward):
        # incorporate the observed reward (float in [0, 1])
        ...

    def predict_best(self):
        # return your current best-guess arm per bandit (used for scoring)
        ...
```

2. Register it in `algorithms/__init__.py`:

```python
from .my_algo import MyAlgorithm
ALGORITHMS[MyAlgorithm.name] = MyAlgorithm
```

3. Run it: `python run_experiment.py --bandits 9 --rounds 100 --algorithm myalgo`

The contract: the algorithm never sees the optimal arms or the noise level —
only the rewards it receives via `update()`. Scoring compares
`predict_best()` against the hidden optimum each round.

## Design notes

- **Environment vs. algorithm separation**: `TeamRewardEnvironment` owns the
  optimal-arm vector and reward noise, so algorithms physically cannot peek
  at the answer. Alternative reward structures (per-dimension credit,
  relative thresholds, etc.) can be added there without touching algorithms.
- **Fresh state per run**: the harness constructs a new algorithm instance
  per run, which replaces the original `clone_bandits()` deep-copy dance.
- **Known quirks preserved in `dreamteam.py`** (documented as class
  constants so they're easy to experiment with): the fixed `0.1` reward
  threshold, the `+1000` beta penalty, and the global budget peak `m=2`.
