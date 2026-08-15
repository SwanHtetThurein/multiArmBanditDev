"""Aggregator for the unconstrained perturbation experiment.

Reads the per-algorithm full CSVs from the unconstrained perturbation
results directory and writes:

  1. A cross-algorithm summary CSV with one row per (algorithm, noise_level)
     and columns:
       algorithm, noise_level, n_runs,
       baseline_mean, baseline_std,
       recovery_threshold_rounds_<i>_mean, recovery_threshold_rounds_<i>_std,
       time_to_baseline_rounds_<i>_mean, time_to_baseline_rounds_<i>_std,
       max_dip_<i>_mean, max_dip_<i>_std,
       recovered_pct_<i>
     (one block of metrics per perturbation i).

  2. A Markdown table for the writeup.

The aggregator derives its numbers from the full per-round CSVs (more
reliable than the per-algorithm summary CSVs), so it can recover even if a
run was conducted with different baseline-window settings later.

Usage:
    python aggregate_perturbation.py --results_dir "Unconstrained Perturbation Results"
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
from typing import List, Tuple


# Filename pattern from run_perturbation_experiment.py:
#   {algo}_perturbation_rounds300_dims9_tests9_perturbs2_t101_t201_<ts>.csv
FILENAME_RE = re.compile(
    r"^(?P<algo>[a-z_]+)_perturbation_rounds(?P<rounds>\d+)_dims(?P<dims>\d+)"
    r"_tests(?P<tests>\d+)_perturbs(?P<perturbs>\d+)"
    r"(?P<round_str>(?:_t\d+)+)_(?P<ts>\d{8}_\d{6})\.csv$"
)

# Pull the perturbation rounds out of the trailing _tNNN_tNNN_... segment.
ROUNDS_RE = re.compile(r"_t(\d+)")


def discover_files(results_dir: str):
    """Return list of (algorithm, full_csv_path, perturbation_rounds, n_bandits,
    p_perturb_dims). Uses the most recent timestamp if multiple files exist
    for the same algorithm."""
    files_by_algo = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*_perturbation_rounds*.csv")):
        if "_perturbation_summary_" in path:
            continue
        basename = os.path.basename(path)
        m = FILENAME_RE.match(basename)
        if not m:
            continue
        algo = m.group("algo")
        ts = m.group("ts")
        n_bandits = int(m.group("dims"))
        rounds = [int(x) for x in ROUNDS_RE.findall(m.group("round_str"))]
        n_perturbs = int(m.group("perturbs"))
        # p_perturb_dims is encoded in the filename as the count of _tNNN
        # segments = n_perturbs. To recover p_perturb_dims we don't have it
        # directly; default to 1 (the experiment's documented default).
        p_perturb_dims = 1
        files_by_algo[algo].append(
            (ts, path, rounds, n_bandits, p_perturb_dims, n_perturbs))
    out = []
    for algo, lst in files_by_algo.items():
        lst.sort()
        ts, path, rounds, n_bandits, p_perturb_dims, n_perturbs = lst[-1]
        out.append((algo, path, rounds, n_bandits, p_perturb_dims))
    return sorted(out)


def compute_recovery_for_runs(perfs: List[float], baseline: float,
                              perturbation_rounds: List[int],
                              total_rounds: int, n_bandits: int,
                              p_perturb_dims: int) -> List[dict]:
    """For each perturbation, compute recovery metrics within its window.

    The recovery threshold is the achievable post-perturbation ceiling:
    after k perturbation events (each flipping p_perturb_dims dimensions),
    the best an algorithm can do is (n_bandits - k*p_perturb_dims)/n_bandits
    of its pre-perturbation baseline. Using max(0.9, baseline) is wrong
    here because for 1 flip on 9 bandits the ceiling is 8/9 ≈ 0.889,
    below 0.9 — recovery would be impossible to measure.
    """
    out = []
    for i, pr in enumerate(perturbation_rounds):
        post_start = pr + 1
        post_end = (perturbation_rounds[i + 1]
                    if i + 1 < len(perturbation_rounds)
                    else total_rounds)
        flips = (i + 1) * p_perturb_dims
        ceiling_frac = max(0.0, (n_bandits - flips) / n_bandits)
        threshold = baseline * ceiling_frac
        rec_thr = None
        rec_base = None
        max_dip = 0.0
        for t in range(post_start, post_end + 1):
            perf = perfs[t - 1]
            if rec_thr is None and perf >= threshold:
                rec_thr = t - pr
            if rec_base is None and perf >= baseline:
                rec_base = t - pr
            if t <= min(pr + 100, post_end):
                dip = 1.0 - perf
                if dip > max_dip:
                    max_dip = dip
        out.append({
            "recovery_threshold_rounds": rec_thr if rec_thr is not None else float("nan"),
            "time_to_baseline_rounds": rec_base if rec_base is not None else float("nan"),
            "max_dip": max_dip,
            "recovered_within_300": rec_thr is not None,
        })
    return out


def aggregate_one(path: str, perturbation_rounds: List[int],
                  total_rounds: int, n_bandits: int,
                  p_perturb_dims: int) -> List[dict]:
    """Read one per-algorithm CSV, return per-(noise_level, runID) rows with
    the recovery metrics added (one block per perturbation)."""
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
    base_lo, base_hi = 80, 100
    for noise, noise_rows in by_noise.items():
        by_run = defaultdict(list)
        for r in noise_rows:
            by_run[r["runID"]].append(r)
        for run_id in sorted(by_run.keys()):
            runs = sorted(by_run[run_id], key=lambda r: r["round"])
            perfs = [r["performance"] for r in runs]
            baseline = sum(perfs[base_lo - 1:base_hi]) / (base_hi - base_lo + 1)
            metrics = compute_recovery_for_runs(perfs, baseline,
                                                perturbation_rounds, total_rounds,
                                                n_bandits, p_perturb_dims)
            row = {
                "algorithm": os.path.basename(path).split("_")[0],
                "noise_level": noise,
                "runID": run_id,
                "baseline": baseline,
            }
            for i, m in enumerate(metrics, start=1):
                row[f"recovery_threshold_rounds_{i}"] = m["recovery_threshold_rounds"]
                row[f"time_to_baseline_rounds_{i}"] = m["time_to_baseline_rounds"]
                row[f"max_dip_{i}"] = m["max_dip"]
                row[f"recovered_within_300_{i}"] = m["recovered_within_300"]
            out.append(row)
    return out


def safe_mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


def safe_std(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else float("nan"))


def write_cross_summary(per_algo: dict, perturbation_rounds: List[int],
                        output_dir: str, timestamp: str) -> Tuple[str, str]:
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
            out = {
                "algorithm": algo,
                "noise_level": noise,
                "n_runs": n,
                "baseline_mean": safe_mean([r["baseline"] for r in runs]),
                "baseline_std": safe_std([r["baseline"] for r in runs]),
            }
            for i in range(1, len(perturbation_rounds) + 1):
                out[f"recovery_threshold_rounds_{i}_mean"] = safe_mean(
                    [r[f"recovery_threshold_rounds_{i}"] for r in runs])
                out[f"recovery_threshold_rounds_{i}_std"] = safe_std(
                    [r[f"recovery_threshold_rounds_{i}"] for r in runs])
                out[f"time_to_baseline_rounds_{i}_mean"] = safe_mean(
                    [r[f"time_to_baseline_rounds_{i}"] for r in runs])
                out[f"time_to_baseline_rounds_{i}_std"] = safe_std(
                    [r[f"time_to_baseline_rounds_{i}"] for r in runs])
                out[f"max_dip_{i}_mean"] = safe_mean([r[f"max_dip_{i}"] for r in runs])
                out[f"max_dip_{i}_std"] = safe_std([r[f"max_dip_{i}"] for r in runs])
                out[f"recovered_pct_{i}"] = sum(r[f"recovered_within_300_{i}"]
                                                for r in runs) / n * 100.0
            rows.append(out)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r2 = {}
            for k, v in r.items():
                if isinstance(v, float):
                    r2[k] = f"{v:.4f}" if not math.isnan(v) else ""
                else:
                    r2[k] = v
            writer.writerow(r2)

    with open(md_path, "w") as f:
        f.write("# Unconstrained Perturbation Experiment — Cross-Algorithm Summary\n\n")
        f.write(f"Two perturbations at rounds {list(perturbation_rounds)}, "
                f"each flipping 1 of 9 dimensions. Algorithm state carries over.\n")
        f.write("Numbers are means across runs per (algorithm, noise_level).\n\n")
        # Build a readable table that combines both perturbations.
        header = ("| algorithm | noise | n | baseline (mean) | "
                  "recov_1 (mean ± std) | t_to_base_1 (mean ± std) | max_dip_1 | recovered_pct_1 | "
                  "recov_2 (mean ± std) | t_to_base_2 (mean ± std) | max_dip_2 | recovered_pct_2 |")
        sep = "|".join(["---"] * 12)
        f.write(header + "\n" + "|" + sep + "|\n")
        for r in rows:
            def fmt(v, p=1):
                return f"{v:.{p}f}" if not (isinstance(v, float) and math.isnan(v)) else "—"
            line = (
                f"| {r['algorithm']} | {r['noise_level']:.1f} | {r['n_runs']} | "
                f"{fmt(r['baseline_mean'], 3)} | "
                f"{fmt(r['recovery_threshold_rounds_1_mean'])} ± {fmt(r['recovery_threshold_rounds_1_std'])} | "
                f"{fmt(r['time_to_baseline_rounds_1_mean'])} ± {fmt(r['time_to_baseline_rounds_1_std'])} | "
                f"{fmt(r['max_dip_1_mean'], 3)} | "
                f"{fmt(r['recovered_pct_1'], 1)}% | "
                f"{fmt(r['recovery_threshold_rounds_2_mean'])} ± {fmt(r['recovery_threshold_rounds_2_std'])} | "
                f"{fmt(r['time_to_baseline_rounds_2_mean'])} ± {fmt(r['time_to_baseline_rounds_2_std'])} | "
                f"{fmt(r['max_dip_2_mean'], 3)} | "
                f"{fmt(r['recovered_pct_2'], 1)}% |"
            )
            f.write(line + "\n")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-algorithm unconstrained-perturbation CSVs."
    )
    parser.add_argument("--results_dir", default="Unconstrained Perturbation Results",
                        help="directory containing the per-algorithm CSVs")
    parser.add_argument("--total_rounds", type=int, default=300)
    parser.add_argument("--p_perturb_dims", type=int, default=1,
                        help="dims flipped per perturbation event (must match "
                             "the run_perturbation_experiment.py setting; "
                             "default 1)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"results directory not found: {args.results_dir}")

    files = discover_files(args.results_dir)
    if not files:
        raise SystemExit(f"no per-algorithm CSVs found in {args.results_dir}")
    # All files should share the same perturbation pattern; use the first.
    algo0, path0, perturbation_rounds, n_bandits, _ = files[0]
    print(f"Found {len(files)} algorithms; perturbation rounds = {perturbation_rounds}")
    print(f"  Using n_bandits={n_bandits}, p_perturb_dims={args.p_perturb_dims}")
    for algo, path, _r, _n, _p in files:
        print(f"  {algo}: {os.path.basename(path)}")

    per_algo = {}
    for algo, path, rounds, nb, _p in files:
        per_algo[algo] = aggregate_one(path, rounds, args.total_rounds,
                                       nb, args.p_perturb_dims)
        print(f"  aggregated {algo}: {len(per_algo[algo])} (noise, runID) rows")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, md_path = write_cross_summary(per_algo, perturbation_rounds,
                                            args.results_dir, timestamp)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
