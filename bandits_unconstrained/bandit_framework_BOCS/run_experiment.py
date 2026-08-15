"""Entry point. Two ways to run:

Interactive (same prompts as the original script):
    python run_experiment.py

Non-interactive:
    python run_experiment.py --bandits 9 --rounds 100
    python run_experiment.py --bandits 12 --rounds 100 --algorithm dreamteam --seed 42
    python run_experiment.py --bandits 9 --rounds 100 --algorithm random   # baseline

Swap algorithms with --algorithm <name>; see algorithms/__init__.py for the
registry of available names and how to add new ones.
"""

import argparse

from bandits_unconstrained.bandit_framework_BOCS.algorithms import ALGORITHMS
from bandits_unconstrained.bandit_framework_BOCS.experiment import ExperimentSettings, run_experiment, save_results


def prompt_interactive() -> ExperimentSettings:
    print("\n=== Bandit Experiment Configuration ===\n")

    names = ', '.join(sorted(ALGORITHMS))
    while True:
        algo = input(f"Algorithm [{names}] (default: dreamteam): ").strip() or "dreamteam"
        if algo in ALGORITHMS:
            break
        print(f"  ✗ Unknown algorithm '{algo}'. Choose from: {names}\n")

    while True:
        n_bandits = int(input("How many bandits? (must be divisible by 3 for equal early/late/ongoing split): "))
        if n_bandits > 0 and n_bandits % 3 == 0:
            break
        print(f"  ✗ {n_bandits} is not divisible by 3. Please re-enter a valid number (e.g. 3, 6, 9, 12...).\n")

    while True:
        total_rounds = int(input("How many rounds per experiment? "))
        if total_rounds > 0:
            break
        print("  ✗ Rounds must be a positive integer. Please try again.\n")

    return ExperimentSettings(algorithm=algo, n_bandits=n_bandits, total_rounds=total_rounds)


def parse_args() -> ExperimentSettings:
    parser = argparse.ArgumentParser(description="Multi-dimensional bandit experiment harness")
    parser.add_argument("--algorithm", default=None, choices=sorted(ALGORITHMS),
                        help="which recommendation algorithm to run (default: dreamteam)")
    parser.add_argument("--bandits", type=int, default=None,
                        help="number of bandits (must be divisible by 3)")
    parser.add_argument("--rounds", type=int, default=None, help="rounds per run")
    parser.add_argument("--tests", type=int, default=9, help="number of random test scenarios")
    parser.add_argument("--runs", type=int, default=20, help="repetitions per test per noise level")
    parser.add_argument("--noise", type=float, nargs="+", default=None,
                        help="noise levels to sweep (default: 0.0 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    # No CLI config given -> fall back to interactive prompts, as the original did.
    if args.bandits is None and args.rounds is None and args.algorithm is None:
        return prompt_interactive()

    settings = ExperimentSettings(
        algorithm=args.algorithm or "dreamteam",
        n_bandits=args.bandits if args.bandits is not None else 9,
        total_rounds=args.rounds if args.rounds is not None else 100,
        n_tests=args.tests,
        runs_per_test=args.runs,
        seed=args.seed,
    )
    if args.noise is not None:
        settings.noise_levels = args.noise
    return settings


def main():
    settings = parse_args()
    rows, config, tests = run_experiment(settings)
    path = save_results(rows, settings)
    print(f"\nSaved {len(rows)} rows to '{path}'")


if __name__ == "__main__":
    main()
