"""Plotter for the unconstrained perturbation experiment.

Reads per-algorithm full CSVs from
  bandits_framework_all_algos_nonstationary/Unconstrained Perturbation Results/
and produces:

  * <algo>_perturbation.png  — per-algorithm per-round performance curve with
    vertical lines at the two perturbation rounds, panels per noise level.
  * recovery_summary.png — cross-algorithm grouped bar chart of median
    recovery time per algorithm, with one group per (noise_level,
    perturbation). Unconstrained algorithms only.

Usage:
    python graph_perturbation.py \
        --results_dir "Unconstrained Perturbation Results" \
        --output_dir "Graphs"
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Filename pattern from run_perturbation_experiment.py:
#   {algo}_perturbation_rounds300_dims9_tests9_perturbs2_t101_t201_<ts>.csv
FILENAME_RE = re.compile(
    r"^(?P<algo>[a-z_]+)_perturbation_rounds(?P<rounds>\d+)_dims(?P<dims>\d+)"
    r"_tests(?P<tests>\d+)_perturbs(?P<perturbs>\d+)"
    r"(?P<round_str>(?:_t\d+)+)_(?P<ts>\d{8}_\d{6})\.csv$"
)
ROUNDS_RE = re.compile(r"_t(\d+)")

# Display order for algorithms in plots (unconstrained-only suite).
ALGO_ORDER = [
    # reference points
    "dreamteam",
    "dreamteam_orig",
    "random",
    # model-free combinatorial search
    "sa",
    "ols",
    "regevo",
    # surrogate-model Bayesian optimization
    "bocs",
    "bocs_hs",
    "combo",
    "combo_slice",
    "smac",
    "cocabo",
    "casmopolitan",
    # generalized linear model
    "glm_fpl",
    # neural surrogates
    "neurallinear",
    "bootnn",
    # acquisition rules on the shared linear surrogate
    "linucb",
    "kg",
    "purexp",
    "sts",
    # acquisition rules on the shared GP surrogate
    "gp_onehot",
    "gp_nei",
    "gp_ucb",
    "gp_ts",
    # fixed-budget best-arm identification
    "bayesgap",
    # combinatorial bandits
    "cucb",
    "cts",
]


def discover_files(results_dir: str):
    """Return list of (algorithm, full_csv_path, perturbation_rounds).
    Uses the most recent timestamp if multiple files exist for the same algorithm."""
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
        rounds = [int(x) for x in ROUNDS_RE.findall(m.group("round_str"))]
        files_by_algo[algo].append((ts, path, rounds))
    out = []
    for algo, lst in files_by_algo.items():
        lst.sort()
        out.append((algo, lst[-1][1], lst[-1][2]))
    # Order by ALGO_ORDER; unknowns go last.
    return sorted(out, key=lambda x: ALGO_ORDER.index(x[0]) if x[0] in ALGO_ORDER else 999)


def plot_per_algorithm(df, algo, perturbation_rounds, output_path):
    """Per-algorithm plot: per-round mean performance, panels by noise level.
    Vertical lines mark both perturbation rounds; pre/post1/post2 regions shaded."""
    noise_levels = sorted(df["noise_level"].unique())
    n_cols = min(3, len(noise_levels))
    n_rows = (len(noise_levels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             sharex=True, sharey=True)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    pr1, pr2 = perturbation_rounds[0], perturbation_rounds[1]

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        sub = df[df["noise_level"] == noise]
        grouped = sub.groupby("round")["performance"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["round"], grouped["mean"], color="C0", lw=1.5, label="mean")
        ax.fill_between(grouped["round"],
                         grouped["mean"] - grouped["std"],
                         grouped["mean"] + grouped["std"],
                         color="C0", alpha=0.2, label="± 1 std")
        # Two perturbation lines.
        ax.axvline(pr1, color="red", lw=1.2, ls="--",
                   label=f"perturbation @ {pr1}" if idx == 0 else None)
        ax.axvline(pr2, color="darkred", lw=1.2, ls="--",
                   label=f"perturbation @ {pr2}" if idx == 0 else None)
        # Three regions: pre, post1, post2.
        ax.axvspan(0, pr1, color="grey", alpha=0.05)
        ax.axvspan(pr1, pr2, color="orange", alpha=0.05)
        ax.axvspan(pr2, grouped["round"].max() + 5, color="red", alpha=0.05)
        ax.set_title(f"noise = {noise:.1f}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("round")
        if idx % n_cols == 0:
            ax.set_ylabel("performance")

    for j in range(len(noise_levels), len(axes)):
        axes[j].set_visible(False)

    rounds_str = ", ".join(str(r) for r in perturbation_rounds)
    fig.suptitle(f"{algo} — perturbations at rounds {rounds_str}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_recovery(df, perturbation_rounds, n_bandits=9, p_perturb_dims=1):
    """For each (noise_level, runID), compute recovery rounds relative to
    each perturbation round. Returns a DataFrame with one row per run.

    The recovery threshold is the achievable post-perturbation ceiling:
    after k perturbation events (each flipping p_perturb_dims dimensions),
    the best an algorithm can do is (n_bandits - k*p_perturb_dims)/n_bandits
    of its pre-perturbation baseline. Using max(0.9, baseline) is wrong here
    because for 1 flip on 9 bandits the ceiling is 8/9 ≈ 0.889,
    below 0.9 — recovery would be impossible to measure.
    """
    out = []
    for (noise, run_id), sub in df.groupby(["noise_level", "runID"]):
        sub = sub.sort_values("round")
        perfs = sub["performance"].values
        baseline = perfs[79:100].mean()  # rounds 80..100 (0-indexed 79..99)
        algo = sub["algorithm"].iloc[0] if "algorithm" in sub.columns else ""
        row = {
            "algorithm": algo,
            "noise_level": noise,
            "runID": run_id,
            "baseline": baseline,
        }
        for i, pr in enumerate(perturbation_rounds, start=1):
            post_start = pr + 1
            post_end = (perturbation_rounds[i]
                        if i < len(perturbation_rounds)
                        else len(perfs))
            flips = i * p_perturb_dims
            ceiling_frac = max(0.0, (n_bandits - flips) / n_bandits)
            threshold = baseline * ceiling_frac
            rec_thr = None
            rec_base = None
            max_dip = 0.0
            for t in range(post_start, post_end + 1):
                p = perfs[t - 1]
                if rec_thr is None and p >= threshold:
                    rec_thr = t - pr
                if rec_base is None and p >= baseline:
                    rec_base = t - pr
                if t <= min(pr + 100, post_end):
                    dip = 1.0 - p
                    if dip > max_dip:
                        max_dip = dip
            row[f"recovery_threshold_rounds_{i}"] = (
                rec_thr if rec_thr is not None else np.nan)
            row[f"time_to_baseline_rounds_{i}"] = (
                rec_base if rec_base is not None else np.nan)
            row[f"max_dip_{i}"] = max_dip
            row[f"recovered_within_300_{i}"] = rec_thr is not None
        out.append(row)
    return pd.DataFrame(out)


def plot_recovery_summary(recovery_df, perturbation_rounds, output_path):
    """Cross-algorithm grouped bar chart of median recovery time by noise level.
    One cluster per noise level; within a cluster, two bars per algorithm
    (one per perturbation round)."""
    noise_levels = sorted(recovery_df["noise_level"].unique())
    present_algos = [a for a in ALGO_ORDER
                     if a in recovery_df["algorithm"].unique()]
    n_perturbs = len(perturbation_rounds)

    n_panels = len(noise_levels)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharey=True)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    bar_width = 0.8 / n_perturbs
    x_base = np.arange(len(present_algos))

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]
        sub = recovery_df[recovery_df["noise_level"] == noise]
        for k, pr in enumerate(perturbation_rounds, start=1):
            meds = []
            lo_errs = []
            hi_errs = []
            for a in present_algos:
                s = sub[sub["algorithm"] == a][f"recovery_threshold_rounds_{k}"].dropna()
                meds.append(s.median() if len(s) else np.nan)
                lo_errs.append(s.quantile(0.25) if len(s) else np.nan)
                hi_errs.append(s.quantile(0.75) if len(s) else np.nan)
            meds = np.array(meds)
            lo_errs = np.array(lo_errs)
            hi_errs = np.array(hi_errs)
            # Replace NaN medians with 0 for plotting so the bar shows even if no recovery.
            plot_vals = np.where(np.isnan(meds), 0, meds)
            lower = np.where(np.isnan(meds), 0, meds - lo_errs)
            upper = np.where(np.isnan(meds), 0, hi_errs - meds)
            offsets = x_base + (k - (n_perturbs + 1) / 2) * bar_width
            ax.bar(offsets, plot_vals, bar_width,
                   yerr=[lower, upper], capsize=2,
                   label=f"perturbation @ {pr}", color=f"C{k - 1}")
        ax.set_xticks(x_base)
        ax.set_xticklabels(present_algos, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"noise = {noise:.1f}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("recovery rounds (median ± IQR)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Recovery time across unconstrained algorithms — perturbations at "
        f"{list(perturbation_rounds)}", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot unconstrained perturbation experiment results.")
    parser.add_argument("--results_dir", default="Unconstrained Perturbation Results",
                        help="directory containing per-algorithm CSVs")
    parser.add_argument("--output_dir", default="Graphs",
                        help="where to write the PNGs")
    parser.add_argument("--n_bandits", type=int, default=9,
                        help="number of bandits (must match the run config)")
    parser.add_argument("--p_perturb_dims", type=int, default=1,
                        help="dims flipped per perturbation event (must match "
                             "the run config; default 1)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"results directory not found: {args.results_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    files = discover_files(args.results_dir)
    if not files:
        raise SystemExit(f"no per-algorithm CSVs found in {args.results_dir}")

    # All files should share the same perturbation pattern; use the first.
    perturbation_rounds = files[0][2]
    print(f"Found {len(files)} algorithms; perturbation rounds = {perturbation_rounds}")
    print(f"  Using n_bandits={args.n_bandits}, p_perturb_dims={args.p_perturb_dims}")
    for algo, path, _ in files:
        print(f"  {algo}: {os.path.basename(path)}")

    all_recovery_rows = []
    for algo, path, rounds in files:
        df = pd.read_csv(path)
        df["algorithm"] = algo
        per_algo_out = os.path.join(args.output_dir,
                                    f"perturbation_{algo}.png")
        plot_per_algorithm(df, algo, rounds, per_algo_out)
        rec = compute_recovery(df, rounds,
                               n_bandits=args.n_bandits,
                               p_perturb_dims=args.p_perturb_dims)
        all_recovery_rows.append(rec)

    recovery_df = pd.concat(all_recovery_rows, ignore_index=True)
    summary_out = os.path.join(args.output_dir, "recovery_summary.png")
    plot_recovery_summary(recovery_df, perturbation_rounds, summary_out)
    print(f"\nWrote per-algorithm plots to {args.output_dir}/perturbation_<algo>.png")
    print(f"Wrote summary plot to {summary_out}")


if __name__ == "__main__":
    main()
