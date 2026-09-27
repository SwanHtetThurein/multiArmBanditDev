"""Cross-algorithm summary table.

Reads the per-algorithm results Parquet files written by ``run_experiment.py``
or ``run_perturbation_experiment.py`` (the small ones, not ``_trace`` or
``_summary``) and produces one row per (algorithm, switching-limit, noise level),
plus a human-readable Markdown report.

``performance`` is the fraction of roles that the algorithm's *recommended* team
gets right at that round (``n_correct / n_bandits``), so 1.0 means the optimal
team was identified exactly.

On perturbation results, pass ``--phase`` to scope the analysis to one stretch
of the run. Analysing all 300 rounds at once mostly measures recovery from the
last swap, which is rarely the question you meant to ask.

Usage:
    python aggregate_results.py --results_dir "Global results" --output_dir Analysis
    python aggregate_results.py --results_dir "Unconstrained Perturbation Results" \
        --output_dir Analysis --phase pre
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

from analysis_common import (
    PHASE_CHOICES, apply_phase, describe, discover, load, phase_note,
    phase_suffix, rounds_to,
)

# Thresholds (on the mean learning curve) for the "rounds to reach" metrics.
SPEED_THRESHOLDS = (0.50, 0.75, 0.90, 0.95)


def summarize_group(df: pd.DataFrame) -> dict:
    """Metrics for one (algorithm, limit, noise level) cell."""
    first_round = int(df["round"].min())
    last_round = int(df["round"].max())
    span = last_round - first_round + 1
    final = df[df["round"] == last_round]["performance"].to_numpy()
    per_run_mean = df.groupby("run_id")["performance"].mean().to_numpy()
    mean_curve = df.groupby("round")["performance"].mean()

    row = {
        "n_runs": int(final.size),
        "first_round": first_round,
        "final_round": last_round,
        # How good is the team it finally recommends?
        "final_perf_mean": float(np.mean(final)),
        "final_perf_std": float(np.std(final, ddof=1)) if final.size > 1 else 0.0,
        "final_perf_sem": (
            float(np.std(final, ddof=1) / np.sqrt(final.size)) if final.size > 1 else 0.0
        ),
        "final_perf_median": float(np.median(final)),
        # Simple regret: distance from a perfect team.
        "simple_regret_mean": float(1.0 - np.mean(final)),
        # How often is the recommendation exactly right / completely wrong?
        "pct_exact_optimal": float(np.mean(final >= 1.0) * 100.0),
        "pct_zero_correct": float(np.mean(final <= 0.0) * 100.0),
        # Area under the learning curve: rewards getting good *early*, not just
        # ending well. Mean over rounds, so it stays on the same 0-1 scale.
        "auc_mean": float(np.mean(per_run_mean)),
        "auc_std": float(np.std(per_run_mean, ddof=1)) if per_run_mean.size > 1 else 0.0,
        # Where it stands at the quarter marks of the window being analysed.
        "perf_at_25pct_rounds": float(
            mean_curve.get(first_round + max(0, span // 4 - 1), np.nan)),
        "perf_at_50pct_rounds": float(
            mean_curve.get(first_round + max(0, span // 2 - 1), np.nan)),
    }
    for thr in SPEED_THRESHOLDS:
        row[f"rounds_to_{int(thr * 100)}pct"] = rounds_to(mean_curve, thr)

    # Break the final score out by team size, so scaling behaviour is visible.
    for nb, sub in df[df["round"] == last_round].groupby("n_bandits"):
        row[f"final_perf_nb{int(nb)}"] = float(sub["performance"].mean())

    # Perturbation runs carry the pre-swap baseline; how much is still missing?
    if "baseline" in df.columns:
        base = df[df["round"] == last_round]["baseline"].to_numpy()
        row["baseline_mean"] = float(np.mean(base))
        row["gap_to_baseline"] = float(np.mean(base) - np.mean(final))
    return row


def build_table(files: list[dict], phase: str) -> pd.DataFrame:
    rows = []
    for i, entry in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {entry['label']}", flush=True)
        df = apply_phase(load(entry["path"]), phase)
        for noise, sub in df.groupby("noise_level"):
            row = {
                "algorithm": entry["algo"],
                "switch_limit": entry["limit"],
                "condition": entry["label"],
                "phase": phase,
                "noise_level": float(noise),
                "total_rounds": entry["rounds"],
            }
            row.update(summarize_group(sub))
            rows.append(row)
        # Pooled row across every noise level, for the headline ranking.
        row = {
            "algorithm": entry["algo"],
            "switch_limit": entry["limit"],
            "condition": entry["label"],
            "phase": phase,
            "noise_level": -1.0,  # sentinel: "all noise levels pooled"
            "total_rounds": entry["rounds"],
        }
        row.update(summarize_group(df))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["noise_level", "final_perf_mean"], ascending=[True, False])


def noise_robustness(table: pd.DataFrame) -> pd.DataFrame:
    """How much does each algorithm lose as noise rises from 0.0 to the max?"""
    per_noise = table[table["noise_level"] >= 0]
    rows = []
    for cond, sub in per_noise.groupby("condition"):
        sub = sub.sort_values("noise_level")
        x = sub["noise_level"].to_numpy()
        y = sub["final_perf_mean"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else float("nan")
        rows.append(
            {
                "condition": cond,
                "algorithm": sub["algorithm"].iloc[0],
                "final_perf_at_noise_0": float(y[0]),
                "final_perf_at_noise_max": float(y[-1]),
                "drop_clean_to_noisy": float(y[0] - y[-1]),
                "noise_slope": slope,
                "final_perf_mean_all_noise": float(np.mean(y)),
            }
        )
    return pd.DataFrame(rows).sort_values("final_perf_mean_all_noise", ascending=False)


def write_markdown(path: str, table: pd.DataFrame, robust: pd.DataFrame, meta: dict) -> None:
    pooled = table[table["noise_level"] < 0].sort_values("final_perf_mean", ascending=False)
    lines = [
        f"# Cross-algorithm summary{meta['phase_note']}",
        "",
        f"Generated {meta['generated']}",
        f"Results directory: `{meta['results_dir']}`",
        f"{meta['describe']}",
        "",
        "`performance` is the fraction of roles the algorithm's recommended team "
        "gets right, so 1.0 = the optimal team was identified exactly.",
        "",
        "## Overall ranking (all noise levels pooled)",
        "",
        "| # | algorithm | final perf | ± sem | AUC | % exactly optimal | rounds to 90% |",
        "|---|-----------|-----------|-------|-----|-------------------|---------------|",
    ]
    for i, r in enumerate(pooled.itertuples(), 1):
        r90 = r.rounds_to_90pct
        r90s = "—" if not np.isfinite(r90) else f"{r90:.0f}"
        lines.append(
            f"| {i} | {r.condition} | {r.final_perf_mean:.3f} | {r.final_perf_sem:.3f} | "
            f"{r.auc_mean:.3f} | {r.pct_exact_optimal:.1f}% | {r90s} |"
        )

    lines += [
        "",
        "## Noise robustness",
        "",
        "| algorithm | perf @ noise 0 | perf @ max noise | drop | slope |",
        "|-----------|----------------|------------------|------|-------|",
    ]
    for r in robust.itertuples():
        lines.append(
            f"| {r.condition} | {r.final_perf_at_noise_0:.3f} | "
            f"{r.final_perf_at_noise_max:.3f} | {r.drop_clean_to_noisy:.3f} | "
            f"{r.noise_slope:+.3f} |"
        )

    lines += ["", "## Per-noise-level final performance", ""]
    per_noise = table[table["noise_level"] >= 0]
    levels = sorted(per_noise["noise_level"].unique())
    lines += [
        "| algorithm | " + " | ".join(f"noise {lv:g}" for lv in levels) + " |",
        "|" + "---|" * (len(levels) + 1),
    ]
    pivot = per_noise.pivot_table(
        index="condition", columns="noise_level", values="final_perf_mean")
    for cond in pooled["condition"].tolist():
        if cond not in pivot.index:
            continue
        cells = " | ".join(f"{pivot.loc[cond, lv]:.3f}" for lv in levels)
        lines.append(f"| {cond} | {cells} |")

    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", default="Analysis")
    ap.add_argument("--phase", choices=PHASE_CHOICES, default="all",
                    help="Perturbation results only: restrict to one phase.")
    args = ap.parse_args(argv)

    files = discover(args.results_dir)
    if not files:
        print(f"No results files found in {args.results_dir!r}.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {describe(files)}")

    table = build_table(files, args.phase)
    robust = noise_robustness(table)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sfx = phase_suffix(args.phase)
    csv_path = os.path.join(args.output_dir, f"summary_all_algorithms{sfx}_{stamp}.csv")
    rob_path = os.path.join(args.output_dir, f"noise_robustness{sfx}_{stamp}.csv")
    md_path = os.path.join(args.output_dir, f"summary_all_algorithms{sfx}_{stamp}.md")

    table.to_csv(csv_path, index=False)
    robust.to_csv(rob_path, index=False)
    write_markdown(
        md_path, table, robust,
        {
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "results_dir": args.results_dir,
            "describe": describe(files),
            "phase_note": phase_note(args.phase),
        },
    )

    for p in (csv_path, rob_path, md_path):
        print("Wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
