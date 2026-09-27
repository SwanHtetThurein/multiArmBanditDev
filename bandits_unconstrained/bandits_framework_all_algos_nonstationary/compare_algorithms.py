"""Paired statistical comparison between algorithms.

Every algorithm was run on the *same* sampled team settings with the same random
seeds, so the comparison is paired: for a given (setting, noise level) both
algorithms faced an identical problem and identical measurement noise. That is
what makes the tests below legitimate — and far more sensitive than comparing
two independent samples of means.

Outputs:
  pairwise_vs_best_<ts>.csv   every algorithm vs the overall winner: mean paired
                              difference, 95% CI, Wilcoxon signed-rank p,
                              Holm-corrected p, Cliff's delta
  winrate_matrix_<ts>.csv     head-to-head win rate for every ordered pair
  winrate_matrix.png          the same matrix as a heatmap
  significance_vs_best.png    gap behind the winner with 95% CIs

Usage:
    python compare_algorithms.py --results_dir "Global results" --output_dir Analysis
    python compare_algorithms.py --results_dir "Unconstrained Perturbation Results" \
        --output_dir Analysis --phase post1 --noise 0.4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import (
    PHASE_CHOICES, apply_phase, cliffs_delta, describe, discover, holm, load,
    phase_note, phase_suffix,
)

WANT = ["setting_id", "noise_level", "round", "performance", "phase"]


def final_scores(files, phase, noise=None):
    """Matrix of final performance: rows = (setting_id, noise_level), cols = algorithm."""
    cols = {}
    for i, e in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {e['label']}", flush=True)
        df = apply_phase(load(e["path"], WANT), phase)
        df = df[df["round"] == int(df["round"].max())]
        if noise is not None:
            df = df[np.isclose(df["noise_level"], noise)]
        s = df.set_index(["setting_id", "noise_level"])["performance"]
        cols[e["label"]] = s[~s.index.duplicated()]
    return pd.DataFrame(cols).dropna(how="any")


def vs_best(mat):
    means = mat.mean().sort_values(ascending=False)
    best = means.index[0]
    rows, pvals = [], []
    for cond in means.index:
        if cond == best:
            continue
        a, b = mat[best].to_numpy(), mat[cond].to_numpy()
        diff = a - b
        p = 1.0 if np.allclose(diff, 0) else float(
            stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        sem = diff.std(ddof=1) / np.sqrt(diff.size)
        rows.append({
            "algorithm": cond,
            "mean_performance": float(b.mean()),
            "best_algorithm": best,
            "best_mean_performance": float(a.mean()),
            "mean_difference": float(diff.mean()),
            "ci95_low": float(diff.mean() - 1.96 * sem),
            "ci95_high": float(diff.mean() + 1.96 * sem),
            "n_paired": int(diff.size),
            "wilcoxon_p": p,
            "cliffs_delta": cliffs_delta(a, b),
        })
        pvals.append(p)
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(pvals)
    out["significant_at_0.05"] = out["holm_p"] < 0.05
    return best, out.sort_values("mean_performance", ascending=False)


def winrate(mat):
    conds = list(mat.columns)
    arr = mat.to_numpy()
    n = len(conds)
    w = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = arr[:, i] - arr[:, j]
                w[i, j] = (np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size
    return pd.DataFrame(w, index=conds, columns=conds)


def plot_winrate(wr, out_dir, order, note, sfx):
    wr = wr.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(0.34 * len(order) + 4, 0.34 * len(order) + 3))
    im = ax.imshow(wr.values, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_title("Head-to-head win rate\n"
                 f"(row beats column, paired on identical settings){note}", fontsize=11)
    fig.colorbar(im, ax=ax, label="P(row > column)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"winrate_matrix{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vs_best(tab, best, out_dir, note, sfx):
    tab = tab.sort_values("mean_difference")
    y = range(len(tab))
    colors = ["tab:red" if s else "0.6" for s in tab["significant_at_0.05"]]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(tab) + 2.5))
    ax.errorbar(tab["mean_difference"], y,
                xerr=[tab["mean_difference"] - tab["ci95_low"],
                      tab["ci95_high"] - tab["mean_difference"]],
                fmt="none", ecolor="0.4", lw=0.9)
    ax.scatter(tab["mean_difference"], y, c=colors, s=28, zorder=3)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(tab["algorithm"], fontsize=7)
    ax.set_xlabel(f"mean performance gap behind {best}  (95% CI)")
    ax.set_title(f"Paired difference from the winner ({best}){note}\n"
                 "red = significantly worse after Holm correction", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"significance_vs_best{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", default="Analysis")
    ap.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    ap.add_argument("--noise", type=float, default=None,
                    help="Restrict to one noise level (default: pool all of them).")
    args = ap.parse_args(argv)

    files = discover(args.results_dir)
    if not files:
        print(f"No results files found in {args.results_dir!r}.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading final scores from {describe(files)}")

    mat = final_scores(files, args.phase, args.noise)
    print(f"Paired on {len(mat)} (setting, noise) cells x {mat.shape[1]} algorithms.")

    best, tab = vs_best(mat)
    wr = winrate(mat)
    order = mat.mean().sort_values(ascending=False).index.tolist()
    note = phase_note(args.phase)
    sfx = phase_suffix(args.phase) + ("" if args.noise is None else f"_noise{args.noise:g}")

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p1 = os.path.join(args.output_dir, f"pairwise_vs_best{sfx}_{stamp}.csv")
    p2 = os.path.join(args.output_dir, f"winrate_matrix{sfx}_{stamp}.csv")
    tab.to_csv(p1, index=False)
    wr.reindex(index=order, columns=order).to_csv(p2)
    p3 = plot_winrate(wr, args.output_dir, order, note, sfx)
    p4 = plot_vs_best(tab, best, args.output_dir, note, sfx)

    n_sig = int(tab["significant_at_0.05"].sum())
    print(f"\nWinner: {best} (mean final performance {mat[best].mean():.4f})")
    print(f"{n_sig} of {len(tab)} other algorithms are significantly worse "
          "after Holm correction.")
    not_sig = tab[~tab["significant_at_0.05"]]["algorithm"].tolist()
    if not_sig:
        print("Statistically indistinguishable from the winner: " + ", ".join(not_sig))
    for p in (p1, p2, p3, p4):
        print("Wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
