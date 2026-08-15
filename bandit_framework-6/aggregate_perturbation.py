"""Aggregator for the perturbation experiment.

Reads the per-algorithm full CSVs (or per-algorithm summary CSVs) from
bandit_framework-6/Perturbation results/ and writes:

  1. A cross-algorithm summary CSV (`perturbation_summary_all_algorithms.csv`)
     with one row per (algorithm, noise_level) and columns:
       algorithm, noise_level, n_runs, baseline_mean, baseline_std,
       recovery_threshold_rounds_mean, recovery_threshold_rounds_std,
       time_to_baseline_rounds_mean, time_to_baseline_rounds_std,
       max_post_dip_mean, max_post_dip_std,
       recovered_pct

  2. A Markdown table (`perturbation_summary_all_algorithms.md`) for the writeup.

The aggregator derives its numbers from the full per-round CSVs (more
reliable than the per-algorithm summary CSVs, which only carry one row per
run), so it can recover even if a run was run with different baseline-window
settings later.

Usage:
    python aggregate_perturbation.py --results_dir "Perturbation results"
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple


# Filename pattern from run_perturbation_experiment.py:
#   {algo}_perturbation_rounds300_dims9_tests9_perturb1_t101_<timestamp>.csv
FILENAME_RE = re.compile(
    r"^(?P<algo>[a-z_]+)_perturbation_rounds(?P<rounds>\d+)_dims(?P<dims>\d+)"
    r"_tests(?P<tests>\d+)_perturb(?P<perturb>\d+)_t(?P<tround>\d+)_(?P<ts>\d{8}_\d{6})\.csv$"
)


def discover_files(results_dir: str) -> List[Tuple[str, str]]:
    """Return list of (algorithm, full_csv_path). One entry per algorithm
    (uses the most recent timestamp if multiple files exist)."""
    files_by_algo: dict = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*_perturbation_rounds*_*.csv")):
        # Skip summary files
        if "_perturbation_summary_" in path:
            continue
        basename = os.path.basename(path)
        m = FILENAME_RE.match(basename)
        if not m:
            continue
        algo = m.group("algo")
        ts = m.group("ts")
        files_by_algo[algo].append((ts, path))
    out = []
    for algo, lst in files_by_algo.items():
        lst.sort()
        out.append((algo, lst[-1][1]))
    return sorted(out)


def compute_recovery_from_rows(rows: List[dict], perturbation_round: int,
                               total_rounds: int) -> dict:
    """Re-derive recovery metrics from a per-round table.

    `rows` is a list of dicts with keys: runID, round, performance.
    Returns one summary entry per runID with the four recovery metrics.
    """
    base_lo, base_hi = 80, 100
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["runID"]].append(r)
    out = []
    for run_id in sorted(by_run.keys()):
        runs = sorted(by_run[run_id], key=lambda r: r["round"])
        perfs = [r["performance"] for r in runs]
        baseline = sum(perfs[base_lo - 1:base_hi]) / (base_hi - base_lo + 1)

        post_start = perturbation_round + 1
        threshold = max(0.9, baseline)

        recovery_threshold = None
        time_to_baseline = None
        max_post_dip = 0.0
        for idx, perf in enumerate(perfs):
            t = idx + 1  # round number
            if t < post_start:
                continue
            if recovery_threshold is None and perf >= threshold:
                recovery_threshold = t - perturbation_round
            if time_to_baseline is None and perf >= baseline:
                time_to_baseline = t - perturbation_round
            if t <= min(200, total_rounds):
                dip = 1.0 - perf
                if dip > max_post_dip:
                    max_post_dip = dip

        out.append({
            "runID": run_id,
            "baseline": baseline,
            "recovery_threshold_rounds": (recovery_threshold if recovery_threshold is not None
                                          else float("nan")),
            "time_to_baseline_rounds": (time_to_baseline if time_to_baseline is not None
                                        else float("nan")),
            "max_post_dip": max_post_dip,
            "recovered_within_300": recovery_threshold is not None,
        })
    return out


def aggregate_one(path: str, perturbation_round: int, total_rounds: int) -> List[dict]:
    """Read one per-algorithm CSV, return per-(noise_level, runID) rows with
    the recovery metrics added."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "runID": int(r["runID"]),
                "noise_level": float(r["noise_level"]),
                "round": int(r["round"]),
                "performance": float(r["performance"]),
            })
    by_noise = defaultdict(list)
    for r in rows:
        by_noise[r["noise_level"]].append(r)
    out = []
    for noise, noise_rows in by_noise.items():
        by_run = defaultdict(list)
        for r in noise_rows:
            by_run[r["runID"]].append(r)
        for run_id in sorted(by_run.keys()):
            runs = sorted(by_run[run_id], key=lambda r: r["round"])
            metrics = compute_recovery_from_rows(runs, perturbation_round, total_rounds)[0]
            out.append({"algorithm": os.path.basename(path).split("_")[0],
                        "noise_level": noise, **metrics})
    return out


def safe_mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else float("nan"))


def write_cross_summary(per_algo: dict, output_dir: str, timestamp: str) -> str:
    """Write the cross-algorithm summary CSV and Markdown table."""
    csv_path = os.path.join(output_dir, f"perturbation_summary_all_algorithms_{timestamp}.csv")
    md_path = os.path.join(output_dir, f"perturbation_summary_all_algorithms_{timestamp}.md")

    rows = []
    for algo in sorted(per_algo.keys()):
        by_noise = defaultdict(list)
        for r in per_algo[algo]:
            by_noise[r["noise_level"]].append(r)
        for noise in sorted(by_noise.keys()):
            runs = by_noise[noise]
            n = len(runs)
            baselines = [r["baseline"] for r in runs]
            recovery = [r["recovery_threshold_rounds"] for r in runs]
            time_to_base = [r["time_to_baseline_rounds"] for r in runs]
            dips = [r["max_post_dip"] for r in runs]
            recovered = [r["recovered_within_300"] for r in runs]
            rows.append({
                "algorithm": algo,
                "noise_level": noise,
                "n_runs": n,
                "baseline_mean": safe_mean(baselines),
                "baseline_std": safe_std(baselines),
                "recovery_threshold_rounds_mean": safe_mean(recovery),
                "recovery_threshold_rounds_std": safe_std(recovery),
                "time_to_baseline_rounds_mean": safe_mean(time_to_base),
                "time_to_baseline_rounds_std": safe_std(time_to_base),
                "max_post_dip_mean": safe_mean(dips),
                "max_post_dip_std": safe_std(dips),
                "recovered_pct": sum(recovered) / n * 100.0,
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            # format floats for readability
            r2 = {}
            for k, v in r.items():
                if isinstance(v, float):
                    r2[k] = f"{v:.4f}" if not math.isnan(v) else ""
                else:
                    r2[k] = v
            writer.writerow(r2)

    with open(md_path, "w") as f:
        f.write("# Perturbation Experiment — Cross-Algorithm Summary\n\n")
        f.write("Numbers below are means across the 180 runs per (algorithm, noise_level).\n")
        f.write("Recovery thresholds are relative to perturbation_round (round 101).\n")
        f.write("`recovered_pct` is the fraction of runs that hit the recovery threshold by round 300.\n\n")
        f.write("| algorithm | noise | n | baseline (mean) | recovery_rounds (mean ± std) | time_to_baseline (mean ± std) | max_post_dip (mean) | recovered_pct |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v, p=1):
                return f"{v:.{p}f}" if not (isinstance(v, float) and math.isnan(v)) else "—"
            f.write(
                f"| {r['algorithm']} | {r['noise_level']:.1f} | {r['n_runs']} | "
                f"{fmt(r['baseline_mean'], 3)} | "
                f"{fmt(r['recovery_threshold_rounds_mean'])} ± {fmt(r['recovery_threshold_rounds_std'])} | "
                f"{fmt(r['time_to_baseline_rounds_mean'])} ± {fmt(r['time_to_baseline_rounds_std'])} | "
                f"{fmt(r['max_post_dip_mean'], 3)} | "
                f"{fmt(r['recovered_pct'], 1)}% |\n"
            )
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-algorithm perturbation CSVs into a cross-algorithm summary."
    )
    parser.add_argument("--results_dir", default="Perturbation results",
                        help="directory containing the per-algorithm CSVs")
    parser.add_argument("--perturbation_round", type=int, default=101)
    parser.add_argument("--total_rounds", type=int, default=300)
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"results directory not found: {args.results_dir}")

    files = discover_files(args.results_dir)
    if not files:
        raise SystemExit(f"no per-algorithm CSVs found in {args.results_dir}")
    print(f"Found {len(files)} algorithms:")
    for algo, path in files:
        print(f"  {algo}: {os.path.basename(path)}")

    per_algo = {}
    for algo, path in files:
        per_algo[algo] = aggregate_one(path, args.perturbation_round, args.total_rounds)
        print(f"  aggregated {algo}: {len(per_algo[algo])} (noise, runID) rows")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, md_path = write_cross_summary(per_algo, args.results_dir, timestamp)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
