"""Plotter for the perturbation experiment.

Reads per-algorithm full CSVs from bandit_framework-6/Perturbation results/
and produces:

  * <algo>_perturbation.png  — per-algorithm per-round performance curve with
    vertical line at the perturbation round, panels per noise level.
  * recovery_summary.png — cross-algorithm bar chart of median recovery time
    by noise level, with separate panels for unconstrained vs constrained
    variants.

Usage:
    python graph_perturbation.py --results_dir "Perturbation results" \
        --output_dir "Graph generation"
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Filename pattern from run_perturbation_experiment.py
FILENAME_RE = re.compile(
    r"^(?P<algo>[a-z_]+)_perturbation_rounds(?P<rounds>\d+)_dims(?P<dims>\d+)"
    r"_tests(?P<tests>\d+)_perturb(?P<perturb>\d+)_t(?P<tround>\d+)_(?P<ts>\d{8}_\d{6})\.csv$"
)

# Display order for algorithms in plots
ALGO_ORDER = [
    "dreamteam",
    "random",
    "bocs",
    "combo",
    "neurallinear",
    "neuralucb",
    "neuralts",
    "bocs_constrained",
    "combo_constrained",
    "neurallinear_constrained",
    "neuralucb_constrained",
    "neuralts_constrained",
]


def discover_files(results_dir: str):
    """Return list of (algorithm, full_csv_path). Uses the most recent
    timestamp if multiple files exist for the same algorithm."""
    files_by_algo = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*_perturbation_rounds*_*.csv")):
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
    return sorted(out, key=lambda x: ALGO_ORDER.index(x[0]) if x[0] in ALGO_ORDER else 999)


def plot_per_algorithm(df, algo, perturbation_round, output_path):
    """Per-algorithm plot: per-round mean performance, panels by noise level."""
    noise_levels = sorted(df["noise_level"].unique())
    n_cols = min(3, len(noise_levels))
    n_rows = (len(noise_levels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             sharex=True, sharey=True)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        sub = df[df["noise_level"] == noise]
        # group by round to get mean and std
        grouped = sub.groupby("round")["performance"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["round"], grouped["mean"], color="C0", lw=1.5, label="mean")
        ax.fill_between(grouped["round"],
                         grouped["mean"] - grouped["std"],
                         grouped["mean"] + grouped["std"],
                         color="C0", alpha=0.2, label="± 1 std")
        ax.axvline(perturbation_round, color="red", lw=1.2, ls="--",
                   label="perturbation" if idx == 0 else None)
        # Shade pre/post regions
        ax.axvspan(0, perturbation_round, color="grey", alpha=0.05)
        ax.axvspan(perturbation_round, grouped["round"].max() + 5, color="orange", alpha=0.05)
        ax.set_title(f"noise = {noise:.1f}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("round")
        if idx % n_cols == 0:
            ax.set_ylabel("performance")

    # Hide unused axes
    for j in range(len(noise_levels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{algo} — perturbation at round {perturbation_round}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_recovery(df, perturbation_round):
    """For each (noise_level, runID), compute recovery rounds relative to
    the perturbation round. Returns a DataFrame with one row per run."""
    out = []
    for (noise, run_id), sub in df.groupby(["noise_level", "runID"]):
        sub = sub.sort_values("round")
        perfs = sub["performance"].values
        baseline = perfs[79:100].mean()  # rounds 80..100 (0-indexed 79..99)
        threshold = max(0.9, baseline)
        post_start = perturbation_round + 1
        rec_thr = None
        rec_base = None
        max_dip = 0.0
        for t in range(post_start, len(perfs) + 1):
            p = perfs[t - 1]
            if rec_thr is None and p >= threshold:
                rec_thr = t - perturbation_round
            if rec_base is None and p >= baseline:
                rec_base = t - perturbation_round
            if t <= min(200, len(perfs)):
                dip = 1.0 - p
                if dip > max_dip:
                    max_dip = dip
        out.append({
            "algorithm": sub["algorithm"].iloc[0] if "algorithm" in sub.columns else "",
            "noise_level": noise,
            "runID": run_id,
            "baseline": baseline,
            "recovery_threshold_rounds": rec_thr if rec_thr is not None else np.nan,
            "time_to_baseline_rounds": rec_base if rec_base is not None else np.nan,
            "max_post_dip": max_dip,
            "recovered_within_300": rec_thr is not None,
        })
    return pd.DataFrame(out)


def plot_recovery_summary(recovery_df, output_path, perturbation_round):
    """Cross-algorithm bar chart of median recovery time by noise level."""
    noise_levels = sorted(recovery_df["noise_level"].unique())

    # Order: unconstrained first (in ALGO_ORDER), then constrained
    present_algos = [a for a in ALGO_ORDER if a in recovery_df["algorithm"].unique()]
    unconstrained = [a for a in present_algos if "_constrained" not in a]
    constrained = [a for a in present_algos if "_constrained" in a]

    n_panels = len(noise_levels)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             sharey=True)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    bar_width = 0.35
    n_un = len(unconstrained)
    n_co = len(constrained)
    x_un = np.arange(n_un)
    x_co = np.arange(n_co) + n_un + 1.0  # leave a gap between groups

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        sub = recovery_df[recovery_df["noise_level"] == noise]

        un_meds, un_errs, un_pcts = [], [], []
        for a in unconstrained:
            s = sub[sub["algorithm"] == a]["recovery_threshold_rounds"].dropna()
            un_meds.append(s.median() if len(s) else np.nan)
            un_errs.append((s.quantile(0.25) if len(s) else np.nan,
                            s.quantile(0.75) if len(s) else np.nan))
            un_pcts.append(100.0 * sub[sub["algorithm"] == a]["recovered_within_300"].mean())

        co_meds, co_errs, co_pcts = [], [], []
        for a in constrained:
            s = sub[sub["algorithm"] == a]["recovery_threshold_rounds"].dropna()
            co_meds.append(s.median() if len(s) else np.nan)
            co_errs.append((s.quantile(0.25) if len(s) else np.nan,
                            s.quantile(0.75) if len(s) else np.nan))
            co_pcts.append(100.0 * sub[sub["algorithm"] == a]["recovered_within_300"].mean())

        un_meds = np.array(un_meds)
        co_meds = np.array(co_meds)
        # asymmetric error bars: lower is median - q25, upper is q75 - median
        def errs(meds, q):
            lower = np.array([(meds[i] - q[i][0]) if not np.isnan(meds[i]) else 0 for i in range(len(meds))])
            upper = np.array([(q[i][1] - meds[i]) if not np.isnan(meds[i]) else 0 for i in range(len(meds))])
            return lower, upper
        un_lo, un_hi = errs(un_meds, un_errs)
        co_lo, co_hi = errs(co_meds, co_errs)

        # Replace NaN medians with 0 for plotting so the bar shows even if no recovery
        un_plot = np.where(np.isnan(un_meds), 0, un_meds)
        co_plot = np.where(np.isnan(co_meds), 0, co_meds)
        ax.bar(x_un, un_plot, bar_width, color="C0",
               yerr=[un_lo, un_hi], capsize=2, label="unconstrained")
        ax.bar(x_co, co_plot, bar_width, color="C1",
               yerr=[co_lo, co_hi], capsize=2, label="constrained")

        ax.set_xticks(list(x_un) + list(x_co))
        labels = unconstrained + constrained
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"noise = {noise:.1f}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("recovery rounds (median ± IQR)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Recovery time across algorithms — perturbation at round {perturbation_round}",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot perturbation experiment results.")
    parser.add_argument("--results_dir", default="Perturbation results",
                        help="directory containing per-algorithm CSVs")
    parser.add_argument("--output_dir", default="Graph generation",
                        help="where to write the PNGs")
    parser.add_argument("--perturbation_round", type=int, default=101)
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"results directory not found: {args.results_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    files = discover_files(args.results_dir)
    if not files:
        raise SystemExit(f"no per-algorithm CSVs found in {args.results_dir}")

    print(f"Found {len(files)} algorithms.")
    all_recovery_rows = []
    for algo, path in files:
        print(f"  {algo}: {os.path.basename(path)}")
        df = pd.read_csv(path)
        df["algorithm"] = algo
        # Per-algorithm plot
        per_algo_out = os.path.join(args.output_dir,
                                    f"perturbation_{algo}.png")
        plot_per_algorithm(df, algo, args.perturbation_round, per_algo_out)
        # Compute recovery for the summary plot
        rec = compute_recovery(df, args.perturbation_round)
        all_recovery_rows.append(rec)

    recovery_df = pd.concat(all_recovery_rows, ignore_index=True)
    summary_out = os.path.join(args.output_dir, "recovery_summary.png")
    plot_recovery_summary(recovery_df, summary_out, args.perturbation_round)
    print(f"\nWrote per-algorithm plots to {args.output_dir}/perturbation_<algo>.png")
    print(f"Wrote summary plot to {summary_out}")


if __name__ == "__main__":
    main()