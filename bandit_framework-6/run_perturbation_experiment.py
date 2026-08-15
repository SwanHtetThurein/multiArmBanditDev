"""300-round perturbation experiment driver.

Runs a non-stationary bandit benchmark: the same multi-dimensional team
optimization setup as run_experiment.py, but the hidden optimal arm is
swapped at round 101 (configurable). Algorithms keep their state across
the swap — this is the classical non-stationary bandit problem. We
measure how quickly each algorithm recovers.

Usage:
    python run_perturbation_experiment.py --algorithm dreamteam --seed 42
    python run_perturbation_experiment.py --algorithm bocs --bandits 9 --rounds 300
    python run_perturbation_experiment.py --algorithm neuralucb --perturbation_round 151 --p_perturb_dims 3

The 'phase' column in the output CSV marks rounds as 'pre' (1..perturbation_round)
or 'post' (perturbation_round+1..total_rounds). The performance metric is the
fraction of dimensions whose predict_best() matches the CURRENT optimal arm at
each round, so performance naturally drops at the perturbation round and recovers
later.

Output schema (per-round):
    rowid, runID, noise_level, round, performance, phase, baseline
where baseline = mean(performance over rounds 80..100) for the run.
Two CSVs are written: one per-algorithm full CSV, and one per-algorithm summary
CSV with the recovery metrics. The aggregator (aggregate_perturbation.py) reads
the per-algorithm full CSVs.
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from algorithms import ALGORITHMS, get_algorithm, ProblemConfig
from environment import TeamRewardEnvironment
from experiment import DEFAULT_NOISE_LEVELS, generate_problem, generate_tests


# Default output directory (created on first save).
DEFAULT_OUTPUT_DIR = "Perturbation results"


@dataclass
class PerturbationSettings:
    algorithm: str = "dreamteam"
    n_bandits: int = 9
    total_rounds: int = 300
    perturbation_round: int = 101              # round AFTER which the swap happens
    p_perturb_dims: int = 1                    # number of dimensions to flip
    n_tests: int = 9
    runs_per_test: int = 20
    noise_levels: List[float] = field(default_factory=lambda: list(DEFAULT_NOISE_LEVELS))
    min_arms: int = 2
    max_arms: int = 5
    seed: Optional[int] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    baseline_window: int = (80, 100)          # inclusive pre-perturbation window

    def validate(self):
        if self.n_bandits <= 0 or self.n_bandits % 3 != 0:
            raise ValueError(f"n_bandits must be a positive multiple of 3, got {self.n_bandits}")
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if not (1 <= self.perturbation_round < self.total_rounds):
            raise ValueError(
                f"perturbation_round must be in [1, {self.total_rounds - 1}], got {self.perturbation_round}"
            )
        if not (1 <= self.p_perturb_dims <= self.n_bandits):
            raise ValueError(
                f"p_perturb_dims must be in [1, {self.n_bandits}], got {self.p_perturb_dims}"
            )


def make_perturbed_optimal(original: List[int], arm_counts: List[int],
                           p: int, rng: random.Random) -> List[int]:
    """Return a new optimal arm that differs from `original` in exactly p
    randomly chosen dimensions. For each flipped dimension, the new arm is
    drawn uniformly from {0..n-1} \\ {original[d]} until it differs."""
    new = list(original)
    candidates = list(range(len(original)))
    rng.shuffle(candidates)
    flipped = 0
    for d in candidates:
        if flipped >= p:
            break
        n = arm_counts[d]
        # uniform draw from arms that differ from current
        options = [a for a in range(n) if a != original[d]]
        new[d] = rng.choice(options)
        flipped += 1
    assert flipped == p
    return new


def run_single_with_perturbation(
    algo_cls, config: ProblemConfig, initial_bias: List[int],
    optimal_arm: List[int], total_rounds: int, noise: float,
    perturbation_round: int, perturbed_optimal: List[int],
    return_switches: bool = False,
):
    """One run: fresh algorithm, fresh environment. At round `perturbation_round`,
    the environment's optimal arm is swapped to `perturbed_optimal`. The algorithm
    is never reset.

    Returns a list of per-round performance values (length total_rounds).
    If return_switches is True, also returns the per-round number of dimensions
    whose PLAYED arm changed vs. the previous round (length total_rounds - 1).
    """
    algorithm = algo_cls(config, initial_bias, total_rounds)
    environment = TeamRewardEnvironment(optimal_arm, noise)

    performances = []
    switches = []
    prev_arms = None
    for round_num in range(1, total_rounds + 1):
        # Inject the perturbation between rounds (perturbation_round..perturbation_round+1).
        # After this block executes, the next iteration's reward/evaluate_prediction
        # both read the new optimal_arm.
        if round_num == perturbation_round + 1:
            environment.optimal_arm = perturbed_optimal

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


def run_perturbation_experiment(settings: PerturbationSettings,
                                config: Optional[ProblemConfig] = None,
                                tests: Optional[List] = None,
                                verbose: bool = True) -> tuple:
    """Full sweep. Returns (rows, summary_rows, config, tests)."""
    settings.validate()

    # Seed ONCE so all 12 algorithms (when run with the same seed) see the
    # same problem, tests, and *perturbation events*. The per-algorithm
    # perturbation draw uses a seeded RNG so the events are deterministic.
    if settings.seed is not None:
        random.seed(settings.seed)
        np.random.seed(settings.seed)
    # Separate RNG for the perturbation events so perturbation draws are
    # decoupled from the algorithm's Thompson sampling / weight draws.
    perturb_rng = random.Random((settings.seed or 0) * 9973 + 1)

    algo_cls = get_algorithm(settings.algorithm)

    if config is None:
        config = generate_problem(settings)
    if tests is None:
        tests = generate_tests(config.arm_counts, settings.n_tests)

    if verbose:
        print(f"Algorithm: {settings.algorithm}")
        for i, (n, t) in enumerate(zip(config.arm_counts, config.bandit_types), 1):
            print(f"  bandit_{i}: type={t}, arms={n}")
        print(f"\nGenerated {len(tests)} tests (each = (initial_bias, optimal_arm)):")
        for t in tests:
            print(f"  Initial bias: {t[0]}, Optimal arm: {t[1]}")
        print(f"\nPerturbation: at round {settings.perturbation_round}, "
              f"flip {settings.p_perturb_dims} of {settings.n_bandits} dimensions "
              f"of the hidden optimal arm. Algorithm state does NOT reset.")

    # Pre-compute the perturbed optimal arm for each test (deterministic given seed).
    perturbed_optima = [
        make_perturbed_optimal(opt, config.arm_counts, settings.p_perturb_dims, perturb_rng)
        for _, opt in tests
    ]

    rows = []               # per-round per-run rows
    summary_rows = []       # one row per (noise, runID)
    rowid = 0
    run_id = 0
    base_lo, base_hi = settings.baseline_window
    for noise in settings.noise_levels:
        for _ in range(settings.runs_per_test):
            for test_idx, (initial_bias, optimal_arm) in enumerate(tests):
                perturbed = perturbed_optima[test_idx]
                performances = run_single_with_perturbation(
                    algo_cls, config, initial_bias[:], optimal_arm[:],
                    settings.total_rounds, noise,
                    settings.perturbation_round, perturbed,
                )

                # Per-run baseline = mean performance over [base_lo, base_hi].
                baseline = sum(performances[base_lo - 1:base_hi]) / (base_hi - base_lo + 1)

                for round_num, perf in enumerate(performances, start=1):
                    phase = "pre" if round_num <= settings.perturbation_round else "post"
                    rows.append([rowid, run_id, noise, round_num, perf, phase, baseline])
                    rowid += 1

                # Recovery metrics (per run).
                summary = compute_recovery(
                    performances, baseline, settings.perturbation_round,
                    settings.total_rounds,
                )
                summary_rows.append([settings.algorithm, noise, run_id, baseline,
                                     summary["recovery_threshold_rounds"],
                                     summary["time_to_baseline_rounds"],
                                     summary["max_post_dip"],
                                     summary["recovered_within_300"]])
                run_id += 1
        if verbose:
            print(f"noise={noise}: done")

    return rows, summary_rows, config, tests


def compute_recovery(performances: List[float], baseline: float,
                     perturbation_round: int, total_rounds: int) -> dict:
    """Compute the four recovery metrics from a single run's performance series.

    recovery_threshold_rounds: smallest t >= 101 with performance[t] >= max(0.9, baseline)
    time_to_baseline_rounds:    smallest t >= 101 with performance[t] >= baseline
    max_post_dip:               1 - min(performance[t] for t in 101..200)
    recovered_within_300:       bool — did the algorithm reach threshold by round 300?
    """
    POST_START = perturbation_round + 1
    POST_WINDOW_END = min(200, total_rounds)
    threshold = max(0.9, baseline)

    recovery_threshold = None
    time_to_baseline = None
    max_post_dip = 0.0
    for t in range(POST_START, total_rounds + 1):
        perf = performances[t - 1]
        if recovery_threshold is None and perf >= threshold:
            recovery_threshold = t - perturbation_round  # rounds after perturbation
        if time_to_baseline is None and perf >= baseline:
            time_to_baseline = t - perturbation_round
        if t <= POST_WINDOW_END:
            dip = 1.0 - perf
            if dip > max_post_dip:
                max_post_dip = dip

    return {
        "recovery_threshold_rounds": (recovery_threshold if recovery_threshold is not None
                                      else float("nan")),
        "time_to_baseline_rounds": (time_to_baseline if time_to_baseline is not None
                                    else float("nan")),
        "max_post_dip": max_post_dip,
        "recovered_within_300": recovery_threshold is not None,
    }


def save_results(rows: List, settings: PerturbationSettings,
                 base_dir: Optional[str] = None) -> str:
    """Save per-algorithm full CSV (per-round performance with phase + baseline)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (f"{settings.algorithm}_perturbation_rounds{settings.total_rounds}"
                f"_dims{settings.n_bandits}_tests{settings.n_tests}"
                f"_perturb{settings.p_perturb_dims}_t{settings.perturbation_round}_{timestamp}.csv")
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, settings.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rowid", "runID", "noise_level", "round",
                         "performance", "phase", "baseline"])
        writer.writerows(rows)
    return file_path


def save_summary(summary_rows: List, settings: PerturbationSettings,
                 base_dir: Optional[str] = None) -> str:
    """Save per-algorithm summary CSV (recovery metrics)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (f"{settings.algorithm}_perturbation_summary_rounds{settings.total_rounds}"
                f"_dims{settings.n_bandits}_tests{settings.n_tests}"
                f"_perturb{settings.p_perturb_dims}_t{settings.perturbation_round}_{timestamp}.csv")
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, settings.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "noise_level", "runID", "baseline",
                         "recovery_threshold_rounds", "time_to_baseline_rounds",
                         "max_post_dip", "recovered_within_300"])
        for row in summary_rows:
            out = list(row)
            # Format NaN as empty string for CSV friendliness.
            if isinstance(out[4], float) and math.isnan(out[4]):
                out[4] = ""
            if isinstance(out[5], float) and math.isnan(out[5]):
                out[5] = ""
            writer.writerow(out)
    return file_path


def parse_args() -> PerturbationSettings:
    parser = argparse.ArgumentParser(
        description="300-round perturbation experiment (non-stationary bandit)"
    )
    parser.add_argument("--algorithm", default=None, choices=sorted(ALGORITHMS),
                        help="which recommendation algorithm to run (default: dreamteam)")
    parser.add_argument("--bandits", type=int, default=9,
                        help="number of bandits (must be divisible by 3)")
    parser.add_argument("--rounds", type=int, default=300,
                        help="total rounds per run")
    parser.add_argument("--perturbation_round", type=int, default=101,
                        help="round AFTER which the optimal arm is swapped")
    parser.add_argument("--p_perturb_dims", type=int, default=1,
                        help="number of dimensions whose optimal arm is flipped")
    parser.add_argument("--tests", type=int, default=9, help="number of random test scenarios")
    parser.add_argument("--runs", type=int, default=20, help="repetitions per test per noise level")
    parser.add_argument("--noise", type=float, nargs="+", default=None,
                        help="noise levels to sweep (default: 0.0 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    settings = PerturbationSettings(
        algorithm=args.algorithm or "dreamteam",
        n_bandits=args.bandits,
        total_rounds=args.rounds,
        perturbation_round=args.perturbation_round,
        p_perturb_dims=args.p_perturb_dims,
        n_tests=args.tests,
        runs_per_test=args.runs,
        seed=args.seed,
    )
    if args.noise is not None:
        settings.noise_levels = args.noise
    return settings


def main():
    settings = parse_args()
    rows, summary_rows, config, tests = run_perturbation_experiment(settings)
    full_path = save_results(rows, settings)
    summary_path = save_summary(summary_rows, settings)
    print(f"\nSaved {len(rows)} per-round rows to {full_path}")
    print(f"Saved {len(summary_rows)} per-run summary rows to {summary_path}")


if __name__ == "__main__":
    main()
