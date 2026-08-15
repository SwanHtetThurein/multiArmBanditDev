"""300-round unconstrained perturbation experiment driver.

Runs an UNCONSTRAINED bandit benchmark on 7 algorithms: DreamTeam, random,
BOCS, COMBO, NeuralLinear, NeuralUCB, NeuralTS. The hidden optimal arm is
swapped at TWO rounds (101 and 201 by default), each time flipping 1 of the 9
dimensions. Algorithms keep their state across both swaps — this is the
non-stationary regime. We measure how quickly each algorithm recovers after
each perturbation.

Imports algorithms and environment from the parent bandit_framework-6/ folder
so we don't duplicate algorithm code.

Usage:
    python run_perturbation_experiment.py --algorithm dreamteam --seed 42
    python run_perturbation_experiment.py --algorithm bocs --bandits 9 --rounds 300
    python run_perturbation_experiment.py --algorithm neuralucb --perturbation_rounds 101 201

The 'phase' column in the output CSV marks rounds as 'pre' (1..99),
'post1' (100..199), or 'post2' (200..300). The performance metric is the
fraction of dimensions whose predict_best() matches the CURRENT optimal arm
at each round, so performance naturally drops at each perturbation and
recovers later.

Output schema (per-round):
    rowid, runID, noise_level, round, performance, phase, baseline
where baseline = mean(performance over rounds 80..100) for the run.

Two CSVs are written per algorithm: a full per-round CSV and a per-run
summary CSV with the recovery metrics. The aggregator reads the per-round
CSVs.

Default output dir: "Unconstrained Perturbation Results" (relative to this
script).
"""

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

# Make the parent bandit_framework-6/ importable so we can reuse algorithms,
# environment, and experiment helpers without copying them.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from algorithms import ALGORITHMS, get_algorithm  # noqa: E402
from environment import TeamRewardEnvironment  # noqa: E402
from experiment import DEFAULT_NOISE_LEVELS, generate_problem, generate_tests  # noqa: E402


# Default output directory (created on first save).
DEFAULT_OUTPUT_DIR = "Unconstrained Perturbation Results"

# Default perturbations: rounds 101 and 201 (each flips 1 of 9 dimensions).
DEFAULT_PERTURBATION_ROUNDS = (101, 201)
DEFAULT_PERTURB_DIMS = 1


@dataclass
class PerturbationSettings:
    algorithm: str = "dreamteam"
    n_bandits: int = 9
    total_rounds: int = 300
    perturbation_rounds: Tuple[int, ...] = DEFAULT_PERTURBATION_ROUNDS
    p_perturb_dims: int = DEFAULT_PERTURB_DIMS
    n_tests: int = 9
    runs_per_test: int = 20
    noise_levels: List[float] = field(default_factory=lambda: list(DEFAULT_NOISE_LEVELS))
    min_arms: int = 2
    max_arms: int = 5
    seed: Optional[int] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    baseline_window: Tuple[int, int] = (80, 100)   # inclusive pre-perturbation window

    def validate(self):
        if self.n_bandits <= 0 or self.n_bandits % 3 != 0:
            raise ValueError(f"n_bandits must be a positive multiple of 3, got {self.n_bandits}")
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if not self.perturbation_rounds:
            raise ValueError("perturbation_rounds must be non-empty")
        for pr in self.perturbation_rounds:
            if not (1 <= pr < self.total_rounds):
                raise ValueError(
                    f"perturbation_round {pr} must be in [1, {self.total_rounds - 1}]"
                )
        # Must be strictly increasing
        if list(self.perturbation_rounds) != sorted(self.perturbation_rounds):
            raise ValueError("perturbation_rounds must be in ascending order")
        if not (1 <= self.p_perturb_dims <= self.n_bandits):
            raise ValueError(
                f"p_perturb_dims must be in [1, {self.n_bandits}], got {self.p_perturb_dims}"
            )


def make_perturbed_optimal(previous: List[int], arm_counts: List[int],
                           p: int, rng: random.Random) -> List[int]:
    """Return a new optimal arm that differs from `previous` in exactly p
    randomly chosen dimensions. For each flipped dimension, the new arm is
    drawn uniformly from {0..n-1} \\ {previous[d]} until it differs."""
    new = list(previous)
    candidates = list(range(len(previous)))
    rng.shuffle(candidates)
    flipped = 0
    for d in candidates:
        if flipped >= p:
            break
        n = arm_counts[d]
        options = [a for a in range(n) if a != previous[d]]
        new[d] = rng.choice(options)
        flipped += 1
    assert flipped == p
    return new


def run_single_with_perturbations(
    algo_cls, config, initial_bias: List[int],
    optimal_arm: List[int], total_rounds: int, noise: float,
    perturbation_rounds: Tuple[int, ...], perturbed_optima: List[List[int]],
):
    """One run: fresh algorithm, fresh environment. At each round in
    `perturbation_rounds`, the environment's optimal arm is swapped to the
    corresponding entry in `perturbed_optima`. The algorithm is never reset.

    `perturbed_optima[i]` is the new optimal arm to install AFTER round
    perturbation_rounds[i] (i.e., it takes effect at round perturbation_rounds[i]+1).

    Returns a list of per-round performance values (length total_rounds).
    """
    algorithm = algo_cls(config, initial_bias, total_rounds)
    environment = TeamRewardEnvironment(optimal_arm, noise)

    # Schedule the swaps: map round_num -> new_optimal_arm.
    # round_num = perturbation_rounds[i] + 1 is when the new optimum takes effect.
    swap_at = {pr + 1: perturbed_optima[i] for i, pr in enumerate(perturbation_rounds)}

    performances = []
    for round_num in range(1, total_rounds + 1):
        if round_num in swap_at:
            environment.optimal_arm = swap_at[round_num]

        arms_chosen = algorithm.choose(round_num)
        reward = environment.reward(arms_chosen)
        algorithm.update(arms_chosen, reward)
        correct = environment.evaluate_prediction(algorithm.predict_best())
        performances.append(correct / config.n_bandits)
    return performances


def phase_for_round(round_num: int, perturbation_rounds: Tuple[int, ...]) -> str:
    """Return 'pre' for rounds before the first perturbation, 'post1' for
    rounds between the first and second perturbation, 'post2' for rounds
    after the second, etc."""
    for i, pr in enumerate(perturbation_rounds):
        if round_num <= pr:
            return "pre" if i == 0 else f"post{i}"
    return f"post{len(perturbation_rounds)}"


def run_perturbation_experiment(settings: PerturbationSettings,
                                config=None,
                                tests=None,
                                verbose: bool = True) -> tuple:
    """Full sweep. Returns (rows, summary_rows, config, tests)."""
    settings.validate()

    # Seed once so all 6 algorithms (when run with the same seed) see the
    # same problem, tests, and perturbation events. The per-algorithm
    # perturbation draw uses a separate seeded RNG so the events are
    # deterministic and decoupled from the algorithm's Thompson draws.
    if settings.seed is not None:
        random.seed(settings.seed)
        np.random.seed(settings.seed)
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
        print(
            f"\nPerturbations: at rounds {list(settings.perturbation_rounds)}, "
            f"flip {settings.p_perturb_dims} of {settings.n_bandits} dimensions of "
            f"the hidden optimal arm. Algorithm state does NOT reset."
        )

    # Pre-compute the perturbed optimal arm chain for each test.
    # perturbed_optima[test_idx][i] is the optimum that takes effect after
    # perturbation_rounds[i]. Start from the original optimum and chain
    # flips: each event flips p_perturb_dims dimensions of the PREVIOUS
    # optimum, so two perturbations may hit the same dim (reverting it) or
    # different dims.
    perturbed_optima_per_test = []
    for _, opt in tests:
        chain = []
        current = list(opt)
        for _ in settings.perturbation_rounds:
            new = make_perturbed_optimal(current, config.arm_counts,
                                        settings.p_perturb_dims, perturb_rng)
            chain.append(new)
            current = new
        perturbed_optima_per_test.append(chain)

    rows = []
    summary_rows = []
    rowid = 0
    run_id = 0
    base_lo, base_hi = settings.baseline_window
    for noise in settings.noise_levels:
        for _ in range(settings.runs_per_test):
            for test_idx, (initial_bias, optimal_arm) in enumerate(tests):
                perturbed_chain = perturbed_optima_per_test[test_idx]
                performances = run_single_with_perturbations(
                    algo_cls, config, initial_bias[:], optimal_arm[:],
                    settings.total_rounds, noise,
                    settings.perturbation_rounds, perturbed_chain,
                )

                baseline = sum(performances[base_lo - 1:base_hi]) / (base_hi - base_lo + 1)

                for round_num, perf in enumerate(performances, start=1):
                    phase = phase_for_round(round_num, settings.perturbation_rounds)
                    rows.append([rowid, run_id, noise, round_num, perf, phase, baseline])
                    rowid += 1

                summary = compute_recovery_summary(
                    performances, baseline, settings.perturbation_rounds,
                    settings.total_rounds, settings.n_bandits,
                    settings.p_perturb_dims,
                )
                summary_rows.append([settings.algorithm, noise, run_id, baseline] + summary)
                run_id += 1
        if verbose:
            print(f"noise={noise}: done")

    return rows, summary_rows, config, tests


def compute_recovery_summary(performances: List[float], baseline: float,
                             perturbation_rounds: Tuple[int, ...],
                             total_rounds: int, n_bandits: int,
                             p_perturb_dims: int) -> list:
    """For each perturbation event, compute recovery metrics.

    Returns a list of CSV-ready values, one entry per perturbation (in order):
        [recovery_threshold_rounds_1, time_to_baseline_rounds_1, max_dip_1,
         recovered_within_300_1_bool, ...]

    The recovery threshold is the achievable post-perturbation ceiling:
    after k perturbation events (each flipping p_perturb_dims dimensions),
    the best an algorithm can do is (n_bandits - k*p_perturb_dims)/n_bandits
    of the original baseline. Using max(0.9, baseline) as the threshold is
    wrong here because for 1 flip on 9 bandits the ceiling is 8/9 ≈ 0.889,
    below 0.9 — recovery would be impossible to measure.
    """
    out = []
    for i, pr in enumerate(perturbation_rounds):
        post_start = pr + 1
        if i + 1 < len(perturbation_rounds):
            post_end = perturbation_rounds[i + 1]   # stop before next perturbation
        else:
            post_end = total_rounds
        # Achievable ceiling after i+1 perturbation events.
        flips = (i + 1) * p_perturb_dims
        ceiling_frac = max(0.0, (n_bandits - flips) / n_bandits)
        threshold = baseline * ceiling_frac
        recovery_threshold = None
        time_to_baseline = None
        max_dip = 0.0
        for t in range(post_start, post_end + 1):
            perf = performances[t - 1]
            if recovery_threshold is None and perf >= threshold:
                recovery_threshold = t - pr
            if time_to_baseline is None and perf >= baseline:
                time_to_baseline = t - pr
            if t <= min(pr + 100, post_end):  # dip window: 100 rounds after perturbation
                dip = 1.0 - perf
                if dip > max_dip:
                    max_dip = dip
        out.extend([
            recovery_threshold if recovery_threshold is not None else float("nan"),
            time_to_baseline if time_to_baseline is not None else float("nan"),
            max_dip,
            int(recovery_threshold is not None),
        ])
    return out


def save_results(rows: List, settings: PerturbationSettings,
                 base_dir: Optional[str] = None) -> str:
    """Save per-algorithm full CSV (per-round performance with phase + baseline)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perturbs_str = "perturbs" + str(len(settings.perturbation_rounds))
    rounds_str = "_t" + "_t".join(str(pr) for pr in settings.perturbation_rounds)
    filename = (
        f"{settings.algorithm}_perturbation_rounds{settings.total_rounds}"
        f"_dims{settings.n_bandits}_tests{settings.n_tests}"
        f"_{perturbs_str}{rounds_str}_{timestamp}.csv"
    )
    if base_dir is None:
        base_dir = HERE
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
    """Save per-algorithm summary CSV (recovery metrics per perturbation)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perturbs_str = "perturbs" + str(len(settings.perturbation_rounds))
    rounds_str = "_t" + "_t".join(str(pr) for pr in settings.perturbation_rounds)
    filename = (
        f"{settings.algorithm}_perturbation_summary_rounds{settings.total_rounds}"
        f"_dims{settings.n_bandits}_tests{settings.n_tests}"
        f"_{perturbs_str}{rounds_str}_{timestamp}.csv"
    )
    if base_dir is None:
        base_dir = HERE
    save_dir = os.path.join(base_dir, settings.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    # Build header dynamically: one block of 4 cols per perturbation.
    header = ["algorithm", "noise_level", "runID", "baseline"]
    for i, pr in enumerate(settings.perturbation_rounds, start=1):
        header += [
            f"recovery_threshold_rounds_{i}",
            f"time_to_baseline_rounds_{i}",
            f"max_dip_{i}",
            f"recovered_within_300_{i}",
        ]

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in summary_rows:
            out = list(row)
            # Format NaN as empty string for CSV friendliness.
            for j in range(4, len(out)):
                if isinstance(out[j], float) and math.isnan(out[j]):
                    out[j] = ""
            writer.writerow(out)
    return file_path


def parse_args() -> PerturbationSettings:
    parser = argparse.ArgumentParser(
        description="300-round unconstrained perturbation experiment (non-stationary bandit)"
    )
    parser.add_argument("--algorithm", default=None, choices=sorted(ALGORITHMS),
                        help="which recommendation algorithm to run (default: dreamteam)")
    parser.add_argument("--bandits", type=int, default=9,
                        help="number of bandits (must be divisible by 3)")
    parser.add_argument("--rounds", type=int, default=300,
                        help="total rounds per run")
    parser.add_argument("--perturbation_rounds", type=int, nargs="+",
                        default=list(DEFAULT_PERTURBATION_ROUNDS),
                        help="rounds AFTER which the optimal arm is swapped "
                             "(default: 101 201)")
    parser.add_argument("--p_perturb_dims", type=int, default=DEFAULT_PERTURB_DIMS,
                        help="number of dimensions whose optimal arm is flipped at each perturbation")
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
        perturbation_rounds=tuple(args.perturbation_rounds),
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
