"""Behavioural analysis from the full traces.

The small results files only record *how good* each algorithm's recommendation
was. The ``_trace.parquet`` files record what it actually did each round: which
team it asked for, which team it was allowed to play, how many roles it changed,
how far the played team was from the optimum, and per-algorithm diagnostics.
This script turns that into the "how does it search?" half of the story.

Outputs:
  trace_summary_<ts>.csv      one row per (algorithm, noise level) with churn,
                              exploration and blocking statistics
  churn_over_time.png         roles changed per round, all algorithms
  churn_vs_performance.png    late-run churn against final score
  hamming_over_time.png       distance of the *played* team from the optimum
  blocked_by_limit.png        how often the switching limit bit (only written
                              when a limit was actually in force)

Usage:
    python analyze_traces.py --results_dir "Global results" --output_dir Analysis
    python analyze_traces.py --results_dir "Unconstrained Perturbation Results" \
        --output_dir Analysis --phase post1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (
    PHASE_CHOICES, apply_phase, describe, discover, load, phase_note, phase_suffix,
)

# Columns we need; everything else (the team vectors, the diag_* fields) stays
# on disk so a 400 MB directory remains cheap to scan.
WANT = [
    "setting_id", "n_bandits", "noise_level", "round", "phase",
    "n_requested_changes", "n_played_changes", "allowance",
    "performance", "hamming_to_optimum",
]

# How many rounds at the end of the window count as "settled behaviour".
LATE_WINDOW = 20


# --------------------------------------------------------------------------
# Per-file cache.
#
# Scanning a few hundred MB of traces takes minutes, and on some machines the
# shell running this has a time limit. Each trace file's summary is therefore
# cached on disk the moment it is computed, keyed by the file's name, size and
# modification time. Re-running the script picks up where it stopped; delete
# the cache directory to force a full recompute.
# --------------------------------------------------------------------------
def cache_key(path: str, phase: str, noise) -> str:
    st = os.stat(path)
    base = os.path.basename(path).replace(".parquet", "")
    return f"{base}__{phase}__{noise}__{st.st_size}_{int(st.st_mtime)}.json"


def cache_read(cache_dir: str, key: str):
    p = os.path.join(cache_dir, key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None  # corrupt or half-written: just recompute it


def cache_write(cache_dir: str, key: str, payload: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    tmp = os.path.join(cache_dir, key + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, os.path.join(cache_dir, key))  # atomic: never a torn file


def summarize(df, entry, phase):
    last = int(df["round"].max())
    late = df[df["round"] > last - LATE_WINDOW]

    def one(d, d_late, noise):
        # A "blocked" round is one where the algorithm asked for more changes
        # than the switching limit allowed.
        blocked = (
            float(np.mean(d["n_requested_changes"] > d["n_played_changes"]) * 100.0)
            if "n_requested_changes" in d else np.nan
        )
        return {
            "algorithm": entry["algo"],
            "switch_limit": entry["limit"],
            "condition": entry["label"],
            "phase": phase,
            "noise_level": noise,
            "n_rows": int(len(d)),
            # Churn: how many roles it swaps per round. High late churn means it
            # never settles; near-zero late churn means it froze.
            "churn_mean": float(d["n_played_changes"].mean()),
            "churn_late_mean": float(d_late["n_played_changes"].mean()),
            "churn_late_frac_of_team": float(
                (d_late["n_played_changes"] / d_late["n_bandits"]).mean()),
            # Distance of the team it actually played from the true optimum.
            "hamming_mean": float(d["hamming_to_optimum"].mean())
            if "hamming_to_optimum" in d else np.nan,
            "hamming_late_mean": float(d_late["hamming_to_optimum"].mean())
            if "hamming_to_optimum" in d_late else np.nan,
            "pct_rounds_played_optimum": float(
                np.mean(d_late["hamming_to_optimum"] == 0) * 100.0)
            if "hamming_to_optimum" in d_late else np.nan,
            "pct_rounds_blocked_by_limit": blocked,
            "requested_changes_mean": float(d["n_requested_changes"].mean())
            if "n_requested_changes" in d else np.nan,
            "final_perf_mean": float(d[d["round"] == last]["performance"].mean()),
        }

    rows = [one(sub, late[late["noise_level"] == noise], float(noise))
            for noise, sub in df.groupby("noise_level")]
    rows.append(one(df, late, -1.0))
    return rows


def _curve_plot(curves, ranking, ylabel, title, path, swaps):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("turbo")
    for i, cond in enumerate(ranking):
        if cond in curves:
            c = curves[cond]
            ax.plot(c.index, c.values, lw=1.2,
                    color=cmap(i / max(1, len(ranking) - 1)), label=cond)
    for r in swaps:
        ax.axvline(r, color="tab:red", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_churn_vs_perf(tab, out_dir, note, sfx):
    d = tab[tab["noise_level"] < 0]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(d["churn_late_frac_of_team"], d["final_perf_mean"], s=40, color="tab:blue")
    for r in d.itertuples():
        ax.annotate(r.condition, (r.churn_late_frac_of_team, r.final_perf_mean),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"fraction of the team swapped per round, last {LATE_WINDOW} rounds")
    ax.set_ylabel("final performance")
    ax.grid(alpha=0.3)
    ax.set_title(f"Settling down vs. freezing — late churn against final score{note}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"churn_vs_performance{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_blocked(tab, out_dir, note, sfx):
    d = tab[(tab["noise_level"] < 0) & (tab["switch_limit"] != "none")]
    if d.empty:
        return None
    d = d.sort_values("pct_rounds_blocked_by_limit")
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(d) + 2.5))
    ax.barh(range(len(d)), d["pct_rounds_blocked_by_limit"], color="tab:orange")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["condition"], fontsize=7)
    ax.set_xlabel("% of rounds where the switching limit blocked a requested change")
    ax.set_title(f"How often the global constraint actually bit{note}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"blocked_by_limit{sfx}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", default="Analysis")
    ap.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    ap.add_argument("--noise", type=float, default=None,
                    help="Restrict the over-time plots to one noise level.")
    ap.add_argument("--cache_dir", default=None,
                    help="Where per-file results are cached so an interrupted run "
                         "can resume (default: <output_dir>/.trace_cache).")
    ap.add_argument("--no_cache", action="store_true",
                    help="Recompute everything and do not read the cache.")
    args = ap.parse_args(argv)

    files = discover(args.results_dir, kind="trace")
    if not files:
        print(f"No *_trace.parquet files found in {args.results_dir!r}.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, ".trace_cache")
    print(f"Reading traces: {describe(files)}")

    rows, churn_curves, hamming_curves, finals = [], {}, {}, {}
    for i, e in enumerate(files, 1):
        key = cache_key(e["path"], args.phase, args.noise)
        cached = None if args.no_cache else cache_read(cache_dir, key)
        if cached is not None:
            print(f"  [{i}/{len(files)}] {e['label']} (cached)", flush=True)
        else:
            print(f"  [{i}/{len(files)}] {e['label']} "
                  f"({os.path.getsize(e['path']) / 1e6:.0f} MB)", flush=True)
            df = apply_phase(load(e["path"], WANT), args.phase)
            d = df if args.noise is None else df[np.isclose(df["noise_level"], args.noise)]
            last = int(df["round"].max())
            ham = (d.groupby("round")["hamming_to_optimum"].mean()
                   if "hamming_to_optimum" in d else None)
            cached = {
                "rows": summarize(df, e, args.phase),
                "churn": d.groupby("round")["n_played_changes"].mean().to_dict(),
                "hamming": ham.to_dict() if ham is not None else None,
                "final": float(df[df["round"] == last]["performance"].mean()),
            }
            cache_write(cache_dir, key, cached)
            del df, d

        rows.extend(cached["rows"])
        churn_curves[e["label"]] = pd.Series(
            {int(k): v for k, v in cached["churn"].items()}).sort_index()
        if cached["hamming"]:
            hamming_curves[e["label"]] = pd.Series(
                {int(k): v for k, v in cached["hamming"].items()}).sort_index()
        finals[e["label"]] = cached["final"]

    tab = pd.DataFrame(rows)
    ranking = sorted(finals, key=finals.get, reverse=True)
    note, sfx = phase_note(args.phase), phase_suffix(args.phase)
    lo = min(c.index.min() for c in churn_curves.values())
    hi = max(c.index.max() for c in churn_curves.values())
    swaps = [r for r in (files[0]["perturb_rounds"] or []) if lo <= r <= hi]

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"trace_summary{sfx}_{stamp}.csv")
    tab.to_csv(csv_path, index=False)

    written = [
        csv_path,
        _curve_plot(churn_curves, ranking, "roles changed per round (mean)",
                    f"Churn over time — how much each algorithm reshuffles the team{note}",
                    os.path.join(args.output_dir, f"churn_over_time{sfx}.png"), swaps),
        plot_churn_vs_perf(tab, args.output_dir, note, sfx),
    ]
    if hamming_curves:
        written.append(_curve_plot(
            hamming_curves, ranking, "roles wrong in the team actually played (mean)",
            f"Distance from the optimal team over time{note}",
            os.path.join(args.output_dir, f"hamming_over_time{sfx}.png"), swaps))
    blocked = plot_blocked(tab, args.output_dir, note, sfx)
    if blocked:
        written.append(blocked)

    print(f"\nWrote {len(written)} file(s):")
    for p in written:
        print("  ", os.path.basename(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
