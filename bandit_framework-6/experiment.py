"""Experiment harness — the testing structure from the original script,
made algorithm-agnostic.

Responsibilities:
  - Generate a random problem (bandit count, arm counts 2-5, equal thirds of
    early/late/ongoing types).
  - Generate random test scenarios (initial_bias, optimal_arm pairs).
  - Sweep noise levels x repetitions x tests, running a fresh algorithm
    instance against a fresh environment for each run.
  - Record per-round performance and save the same CSV format as the
    original: [rowid, runID, noise_level, round, performance].

The algorithm is injected by name (see algorithms/__init__.py), so swapping
in a new recommendation algorithm requires zero changes here.
"""

import csv
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from algorithms import get_algorithm, ProblemConfig
from environment import TeamRewardEnvironment


DEFAULT_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class ExperimentSettings:
    algorithm: str = "dreamteam"
    n_bandits: int = 9                # must be divisible by 3
    total_rounds: int = 100
    n_tests: int = 9
    runs_per_test: int = 20
    noise_levels: List[float] = field(default_factory=lambda: list(DEFAULT_NOISE_LEVELS))
    min_arms: int = 2
    max_arms: int = 5
    seed: Optional[int] = None
    output_dir: str = "Global results"

    def validate(self):
        if self.n_bandits <= 0 or self.n_bandits % 3 != 0:
            raise ValueError(f"n_bandits must be a positive multiple of 3, got {self.n_bandits}")
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")


def generate_problem(settings: ExperimentSettings) -> ProblemConfig:
    """Random arm counts and an equal early/late/ongoing type split (shuffled)."""
    group_size = settings.n_bandits // 3
    type_pool = (['early'] * group_size + ['late'] * group_size
                 + ['ongoing'] * group_size)
    random.shuffle(type_pool)
    arm_counts = [random.randint(settings.min_arms, settings.max_arms)
                  for _ in range(settings.n_bandits)]
    return ProblemConfig(arm_counts=arm_counts, bandit_types=type_pool)


def generate_tests(arm_counts: List[int], n_tests: int) -> List[List[List[int]]]:
    """Random (initial_bias, optimal_arm) pairs — same as the original."""
    tests = []
    for _ in range(n_tests):
        initial_bias = [random.randint(0, n - 1) for n in arm_counts]
        optimal_arm = [random.randint(0, n - 1) for n in arm_counts]
        tests.append([initial_bias, optimal_arm])
    return tests


def run_single(algo_cls, config: ProblemConfig, initial_bias: List[int],
               optimal_arm: List[int], total_rounds: int, noise: float,
               return_switches: bool = False):
    """One run: fresh algorithm vs fresh environment. Returns per-round
    performance (fraction of dimensions whose best-guess arm is optimal).
    If return_switches, also returns the per-round number of dimensions
    whose PLAYED arm changed vs. the previous round."""
    algorithm = algo_cls(config, initial_bias, total_rounds)
    environment = TeamRewardEnvironment(optimal_arm, noise)

    performances = []
    switches = []
    prev_arms = None
    for round_num in range(1, total_rounds + 1):
        arms_chosen = algorithm.choose(round_num)
        if prev_arms is not None:
            switches.append(sum(1 for a, b in zip(arms_chosen, prev_arms) if a != b))
        prev_arms = list(arms_chosen)
        reward = environment.reward(arms_chosen)
        algorithm.update(arms_chosen, reward)
        correct = environment.evaluate_prediction(algorithm.predict_best())
        performances.append(correct / config.n_bandits)
    if return_switches:
        return performances, switches
    return performances


def run_experiment(settings: ExperimentSettings,
                   config: Optional[ProblemConfig] = None,
                   tests: Optional[List] = None,
                   verbose: bool = True):
    """Full sweep. Returns (rows, config, tests). rows match the original CSV:
    [rowid, runID, noise_level, round, performance]."""
    settings.validate()
    if settings.seed is not None:
        import numpy as np
        random.seed(settings.seed)
        np.random.seed(settings.seed)

    algo_cls = get_algorithm(settings.algorithm)

    if config is None:
        config = generate_problem(settings)
    if tests is None:
        tests = generate_tests(config.arm_counts, settings.n_tests)

    if verbose:
        print(f"Algorithm: {settings.algorithm}")
        for i, (n, t) in enumerate(zip(config.arm_counts, config.bandit_types), 1):
            print(f"  bandit_{i}: type={t}, arms={n}")
        print(f"\nGenerated {len(tests)} tests:")
        for t in tests:
            print(f"  Initial bias: {t[0]}, Optimal arm: {t[1]}")

    rows = []
    rowid = 0
    run_id = 0
    for noise in settings.noise_levels:
        for _ in range(settings.runs_per_test):
            for initial_bias, optimal_arm in tests:
                performances = run_single(
                    algo_cls, config, initial_bias[:], optimal_arm[:],
                    settings.total_rounds, noise,
                )
                for round_num, perf in enumerate(performances, start=1):
                    rows.append([rowid, run_id, noise, round_num, perf])
                    rowid += 1
                run_id += 1
        if verbose:
            print(f"noise={noise}: done")

    return rows, config, tests


def save_results(rows, settings: ExperimentSettings, base_dir: Optional[str] = None) -> str:
    """Save CSV with the same schema as the original, algorithm name included
    in the filename for side-by-side comparisons later."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (f"{settings.algorithm}_results_rounds{settings.total_rounds}"
                f"_dims{settings.n_bandits}_tests{settings.n_tests}_{timestamp}.csv")
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, settings.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rowid', 'runID', 'noise_level', 'round', 'performance'])
        writer.writerows(rows)
    return file_path
