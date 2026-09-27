"""Figures for a finished sweep (stationary or perturbation).

Reads the per-algorithm results Parquet files (the small ones, not ``_trace`` or
``_summary``) and writes:

  learning_curves_all.png        every algorithm's mean curve, one panel per noise
  learning_<algo>.png            one algorithm, mean +/- 1 sd, one panel per noise
  final_ranking.png              final performance bar ranking, one panel per noise
  noise_sensitivity.png          final performance vs noise, one line per algorithm
  heatmap_final_performance.png  algorithms x noise levels
  team_size_scaling.png          final performance vs number of roles (3/6/9)
  sample_efficiency.png          rounds needed to reach 90% of own final score

On perturbation results the curves span all 300 rounds and the swap rounds are
drawn as vertical lines. Pass ``--phase`` to scope everything to one stretch of
the run instead.

Usage:
    python graph_results.py --results_dir "Global results" --output_dir Graphs
    python graph_results.py --results_dir "Unconstrained Perturbation Results" \
        --output_dir Graphs --phase pre
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (
    PHASE_CHOICES, apply_phase, describe, discover, final_round, load,
    phase_note, phase_suffix, ranking_by_final, slug,
)

WANT = ["run_id", "setting_id", "n_bandits", "noise_level", "round", "performance", "phase"]


def load_all(files: list[dict], phase: str):
    """Reduce every results file to the two small frames the plots actually need.

    Holding all of it at once does not scale: 26 algorithms x 3000 runs x 300
    rounds is 23 million rows, which is enough to get the process killed on a
    normal laptop. Nothing here plots individual runs, so each file is collapsed
    as it is read and only the summaries are kept:

      curves  mean and sd of performance per (condition, noise, round)
      finals  one row per run at the last round, for the ranking, the team-size
              breakdown and the standard errors
    """
    curve_frames, final_frames = [], []
    for i, e in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {e['label']}", flush=True)
        df = apply_phase(load(e["path"], WANT), phase)
        last = int(df["round"].max())

        c = (df.groupby(["noise_level", "round"])["performance"]
               .agg(["mean", "std", "count"]).reset_index())
        c["condition"] = e["label"]
        c["algorithm"] = e["algo"]
        curve_frames.append(c)

        f = df.loc[df["round"] == last,
                   ["setting_id", "n_bandits", "noise_level", "performance"]].copy()
        f["round"] = last
        f["condition"] = e["label"]
        f["algorithm"] = e["algo"]
        final_frames.append(f)
        del df, c, f

    return (pd.concat(curve_frames, ignore_index=True),
            pd.concat(final_frames, ignore_index=True))


def _panel_grid(n_levels: int):
    ncols = 3 if n_levels >= 3 else n_levels
    nrows = int(np.ceil(n_levels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.2 * nrows), squeeze=False)
    return fig, axes.ravel()


def _mark_swaps(ax, swaps):
    for r in swaps:
        ax.axvline(r, color="tab:red", ls="--", lw=1.0, alpha=0.8)


def plot_learning_curves_all(curves_df, out_dir, ranking, swaps, note, sfx):
    levels = sorted(curves_df["noise_level"].unique())
    fig, axes = _panel_grid(len(levels))
    cmap = plt.get_cmap("turbo")
    colors = {c: cmap(i / max(1, len(ranking) - 1)) for i, c in enumerate(ranking)}

    for ax, lv in zip(axes, levels):
        sub = curves_df[curves_df["noise_level"] == lv]
        curves = sub.pivot(index="round", columns="condition", values="mean")
        for cond in ranking:
            if cond in curves.columns:
                ax.plot(curves.index, curves[cond], lw=1.3, color=colors[cond], label=cond)
        _mark_swaps(ax, swaps)
        ax.set_title(f"noise = {lv:g}")
        ax.set_xlabel("round")
        ax.set_ylabel("mean performance")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    for ax in axes[len(levels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Learning curves — all algorithms (ordered best to worst){note}", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = os.path.join(out_dir, f"learning_curves_all{sfx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_algorithm(curves_df, out_dir, swaps, sfx):
    levels = sorted(curves_df["noise_level"].unique())
    paths = []
    for cond, sub in curves_df.groupby("condition"):
        fig, axes = _panel_grid(len(levels))
        for ax, lv in zip(axes, levels):
            g = sub[sub["noise_level"] == lv].sort_values("round")
            mean = pd.Series(g["mean"].values, index=g["round"].values)
            sd = pd.Series(g["std"].fillna(0.0).values, index=g["round"].values)
            ax.plot(mean.index, mean.values, color="tab:blue", lw=1.6)
            ax.fill_between(mean.index, mean - sd, mean + sd, color="tab:blue", alpha=0.20)
            _mark_swaps(ax, swaps)
            ax.set_title(f"noise = {lv:g}")
            ax.set_xlabel("round")
            ax.set_ylabel("performance")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        for ax in axes[len(levels):]:
            ax.axis("off")
        fig.suptitle(f"{cond} — mean performance ± 1 sd", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(out_dir, f"learning_{slug(cond)}{sfx}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_final_ranking(finals, out_dir, note, sfx):
    levels = sorted(finals["noise_level"].unique())
    last = final_round(finals)
    fig, axes = _panel_grid(len(levels))
    for ax, lv in zip(axes, levels):
        g = finals[finals["noise_level"] == lv].groupby("condition")["performance"]
        mean = g.mean().sort_values()
        sem = g.sem().reindex(mean.index)
        ax.barh(range(len(mean)), mean.values, xerr=sem.values,
                color="tab:blue", error_kw={"lw": 0.8})
        ax.set_yticks(range(len(mean)))
        ax.set_yticklabels(mean.index, fontsize=6.5)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(f"performance at round {last}")
        ax.set_title(f"noise = {lv:g}")
        ax.grid(axis="x", alpha=0.3)
    for ax in axes[len(levels):]:
        ax.axis("off")
    fig.suptitle(f"Final performance ranking (round {last}, ±1 sem){note}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"final_ranking{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_noise_sensitivity(finals, out_dir, ranking, note, sfx, highlight=8):
    last = final_round(finals)
    pivot = finals.pivot_table(
        index="noise_level", columns="condition", values="performance")
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for cond in pivot.columns:
        if cond not in ranking[:highlight]:
            ax.plot(pivot.index, pivot[cond], color="0.80", lw=1.0, zorder=1)
    for i, cond in enumerate(ranking[:highlight]):
        if cond in pivot.columns:
            ax.plot(pivot.index, pivot[cond], marker="o", lw=2.0,
                    color=cmap(i % 10), label=cond, zorder=3)
    if "random" in pivot.columns:
        ax.plot(pivot.index, pivot["random"], marker="s", lw=1.8, ls="--",
                color="k", label="random (baseline)", zorder=3)
    ax.set_xlabel("noise level")
    ax.set_ylabel(f"mean performance at round {last}")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Noise sensitivity — top {highlight} highlighted, rest in grey{note}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"noise_sensitivity{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap(finals, out_dir, ranking, note, sfx):
    last = final_round(finals)
    pivot = finals.pivot_table(
        index="condition", columns="noise_level", values="performance")
    pivot = pivot.reindex([c for c in ranking if c in pivot.index])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("noise level")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label=f"performance at round {last}")
    ax.set_title(f"Final performance by algorithm and noise level{note}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"heatmap_final_performance{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_team_size(finals, out_dir, ranking, note, sfx, top=10):
    last = final_round(finals)
    keep = ranking[:top] + (["random"] if "random" in ranking else [])
    d = finals[finals["condition"].isin(keep)]
    pivot = d.pivot_table(index="n_bandits", columns="condition", values="performance")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, cond in enumerate(pivot.columns):
        style = dict(ls="--", color="k") if cond == "random" else dict(color=cmap(i % 10))
        ax.plot(pivot.index, pivot[cond], marker="o", lw=1.8, label=cond, **style)
    ax.set_xticks(sorted(finals["n_bandits"].unique()))
    ax.set_xlabel("number of roles in the team (n_bandits)")
    ax.set_ylabel(f"mean performance at round {last}")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Does performance hold up as the team gets bigger?{note}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"team_size_scaling{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sample_efficiency(curves_df, out_dir, note, sfx):
    last = int(curves_df["round"].max())
    rows = []
    for cond, sub in curves_df.groupby("condition"):
        # Pool the per-noise means back together, weighted by run count, so this
        # matches a plain mean over every run at that round.
        pooled = sub.assign(w=sub["mean"] * sub["count"]).groupby("round")[["w", "count"]].sum()
        curve = pooled["w"] / pooled["count"]
        hit = curve[curve >= 0.9 * curve.loc[last]]
        rows.append({
            "condition": cond,
            "rounds_to_90pct_of_final": float(hit.index[0]) if len(hit) else np.nan,
            "final": float(curve.loc[last]),
        })
    eff = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(eff["rounds_to_90pct_of_final"], eff["final"], s=40, color="tab:blue")
    for r in eff.itertuples():
        ax.annotate(r.condition, (r.rounds_to_90pct_of_final, r.final),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("rounds to reach 90% of its own final score  (lower = faster)")
    ax.set_ylabel(f"final performance at round {last}  (higher = better)")
    ax.grid(alpha=0.3)
    ax.set_title(f"Speed vs. quality{note}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"sample_efficiency{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    eff.sort_values("final", ascending=False).to_csv(
        os.path.join(out_dir, f"sample_efficiency{sfx}.csv"), index=False)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", default="Graphs")
    ap.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    ap.add_argument("--skip_per_algorithm", action="store_true",
                    help="Skip the one-PNG-per-algorithm learning curves.")
    args = ap.parse_args(argv)

    files = discover(args.results_dir)
    if not files:
        print(f"No results files found in {args.results_dir!r}.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {describe(files)}")

    curves, finals = load_all(files, args.phase)
    ranking = ranking_by_final(finals)
    note, sfx = phase_note(args.phase), phase_suffix(args.phase)
    # Only mark the swap rounds when they fall inside the window being plotted.
    swaps = [r for r in (files[0]["perturb_rounds"] or [])
             if curves["round"].min() <= r <= curves["round"].max()]

    print("Plotting...")
    written = [
        plot_learning_curves_all(curves, args.output_dir, ranking, swaps, note, sfx),
        plot_final_ranking(finals, args.output_dir, note, sfx),
        plot_noise_sensitivity(finals, args.output_dir, ranking, note, sfx),
        plot_heatmap(finals, args.output_dir, ranking, note, sfx),
        plot_team_size(finals, args.output_dir, ranking, note, sfx),
        plot_sample_efficiency(curves, args.output_dir, note, sfx),
    ]
    if not args.skip_per_algorithm:
        written += plot_per_algorithm(curves, args.output_dir, swaps, sfx)

    print(f"\nWrote {len(written)} file(s) to {args.output_dir!r}:")
    for p in written:
        print("  ", os.path.basename(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
