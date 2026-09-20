"""Entry point for the stationary sweep.

    python3 run_experiment.py --algorithm bocs
    python3 run_experiment.py --algorithm bocs --n-cores 8
    python3 run_experiment.py --algorithm bocs --switch-limit flat --max-changes 2
    python3 run_experiment.py --all --n-cores 8          # every algorithm, in turn

Protocol: 500 sampled settings x 6 noise levels = 3000 runs per algorithm.
Every algorithm reads the same settings file, so they all face identical
problems; see sampling.py.

Resuming: press Ctrl-C at any time. Progress is checkpointed per run, and
rerunning the same command picks up where it stopped. Use --no-resume to
start the run over from scratch.

Output is Parquet: a full per-round trace, a small results projection, and a
manifest recording how the run was produced. --trace controls how much of the
trace is kept (default: full).
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PKG = "bandits_unconstrained.bandit_framework_all_algos"

# BLAS threads must be capped before numpy is imported anywhere, or each
# worker spawns its own thread pool and more cores makes the sweep slower.
from bandits_unconstrained.bandit_framework_all_algos.parallel import (  # noqa: E402
    limit_blas_threads, resolve_n_cores,
)


def parse_args():
    from bandits_unconstrained.bandit_framework_all_algos.algorithms import ALGORITHMS
    from bandits_unconstrained.bandit_framework_all_algos.environment import (
        DEFAULT_MAX_CHANGES, NO_LIMIT, SWITCH_LIMIT_MODES,
    )
    from bandits_unconstrained.bandit_framework_all_algos import sampling, tracing

    p = argparse.ArgumentParser(
        description="Multi-dimensional bandit sweep (500-setting protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algorithm", default=None, choices=sorted(ALGORITHMS),
                   help="which algorithm to run")
    p.add_argument("--all", action="store_true",
                   help="run every registered algorithm, one after another")
    p.add_argument("--rounds", type=int, default=100, help="rounds per run")
    p.add_argument("--noise", type=float, nargs="+", default=None,
                   help="noise levels (default: 0.0 0.2 0.4 0.6 0.8 1.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="master seed for per-run RNG streams")
    p.add_argument("--settings-seed", type=int, default=sampling.DEFAULT_SETTINGS_SEED,
                   help="seed of the shared settings file (changes the benchmark)")
    p.add_argument("--n-settings", type=int, default=sampling.N_SETTINGS,
                   help="how many settings to sample")
    p.add_argument("--settings-path", default=None,
                   help="explicit path to a settings JSON file")
    p.add_argument("--switch-limit", dest="switch_limit", default=NO_LIMIT,
                   choices=list(SWITCH_LIMIT_MODES),
                   help="cap on role changes per round, applied by the environment "
                        "to every algorithm equally")
    p.add_argument("--max-changes", dest="max_changes", type=float,
                   default=DEFAULT_MAX_CHANGES,
                   help="absolute cap for 'flat', or the mid-run peak for 'parabolic'")
    p.add_argument("--n-cores", dest="n_cores", type=int, default=None,
                   help="worker processes (default: all cores but one)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="discard any existing progress and start over")
    p.add_argument("--trace", default=tracing.DEFAULT_TRACE_LEVEL,
                   choices=list(tracing.TRACE_LEVELS),
                   help="how much per-round detail to record: 'off' writes no "
                        "trace, 'basic' records the teams and the constraint's "
                        "effect, 'full' adds each algorithm's internal state")
    p.add_argument("--output-dir", default="Global results")
    return p.parse_args(), ALGORITHMS


def main():
    args, ALGORITHMS = parse_args()

    if not args.algorithm and not args.all:
        print("Pick an algorithm with --algorithm NAME, or --all for every one.\n")
        print("Available: " + ", ".join(sorted(ALGORITHMS)))
        return 2

    limit_blas_threads(resolve_n_cores(args.n_cores))

    from bandits_unconstrained.bandit_framework_all_algos.experiment import (
        ExperimentSettings, run_experiment,
    )

    algos = sorted(ALGORITHMS) if args.all else [args.algorithm]
    results = []
    for i, algo in enumerate(algos, start=1):
        if len(algos) > 1:
            print("=" * 62)
            print(f"[{i}/{len(algos)}] {algo}")
            print("=" * 62)
        s = ExperimentSettings(
            algorithm=algo,
            total_rounds=args.rounds,
            seed=args.seed,
            settings_seed=args.settings_seed,
            n_settings=args.n_settings,
            settings_path=args.settings_path,
            switch_limit=args.switch_limit,
            max_changes=args.max_changes,
            n_cores=args.n_cores,
            resume=args.resume,
            output_dir=args.output_dir,
            trace=args.trace,
        )
        if args.noise is not None:
            s.noise_levels = args.noise
        r = run_experiment(s)
        results.append(r)
        if r["interrupted"]:
            print("\nInterrupted. Rerun the same command to resume.")
            break
        print()

    if len(results) > 1:
        done = sum(1 for r in results if r["complete"])
        print("=" * 62)
        print(f"{done} of {len(results)} algorithms complete.")
        if done < len(results):
            print("Rerun the same command to finish the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
