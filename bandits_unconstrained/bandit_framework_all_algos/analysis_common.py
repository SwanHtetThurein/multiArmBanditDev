"""Shared plumbing for the analysis scripts.

`aggregate_results.py`, `graph_results.py`, `compare_algorithms.py` and
`analyze_traces.py` all need to do the same three boring things: find the
results files in a directory, work out which algorithm and which switching-limit
condition each one belongs to, and (for the non-stationary experiment) restrict
the analysis to one phase of the run. That lives here so the four scripts agree.

This module is deliberately identical in the stationary and non-stationary
folders, so an analysis script copied between them keeps working.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# Every results filename either of the two runners can produce:
#
#   stationary      {algo}_limit-{mode}_rounds{R}_settings{N}_seed{S}[_trace].parquet
#   non-stationary  {algo}_limit-{mode}_perturbation_rounds{R}_settings{N}
#                   _perturbs{K}_t{r1}_t{r2}_seed{S}[_trace|_summary].parquet
#
# Older runs used `_dims{D}_tests{T}` instead of `_settings{N}`, and predate the
# `_limit-{mode}` tag; both still parse, and a missing tag means 'none'.
#
# The hyphen in `_limit-` matters: the algorithm group can contain underscores
# (`combo_slice`, `gp_ucb`, `bocs_hs`), so an underscore-joined tag would be
# ambiguous and files would be silently dropped from the analysis.
FILENAME_RE = re.compile(
    r"^(?P<algo>[a-z0-9_]+?)"
    r"(?:_limit-(?P<limit>[a-z0-9p]+))?"
    r"(?P<perturbation>_perturbation)?"
    r"_rounds(?P<rounds>\d+)"
    r"(?:_settings(?P<settings>\d+)|_dims(?P<dims>\d+)_tests(?P<tests>\d+))"
    r"(?:_perturbs(?P<perturbs>\d+)(?P<round_str>(?:_t\d+)+))?"
    r"(?:_seed(?P<seed>\d+))?"
    r"(?:_(?P<ts>\d{8}_\d{6}))?"
    r"(?P<kind>_trace|_summary)?"
    r"\.(?P<ext>parquet|csv)$"
)

# Phase names used by run_perturbation_experiment.py. 'all' is not a real phase
# value — it means "do not filter".
PHASE_CHOICES = ("all", "pre", "post1", "post2")


def condition_label(algo: str, limit: str | None) -> str:
    """Display name for one (algorithm, switching-limit) condition.

    With no limit the label is just the algorithm name, so the stationary
    unconstrained results read the way they always did. Once a limited sweep
    exists alongside it, the two appear as `bocs` and `bocs [flat]` and every
    script treats them as separate conditions automatically.
    """
    if not limit or limit == "none":
        return algo
    return f"{algo} [{limit}]"


def slug(label: str) -> str:
    """Filesystem-safe form of a condition label, for plot filenames."""
    return re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")


def discover(results_dir: str, kind: str | None = None) -> list[dict]:
    """Find results files in `results_dir`.

    `kind` selects which flavour of file to return:
      None      the per-round results files (the small ones)
      'trace'   the full `_trace.parquet` files
      'summary' the per-run `_summary.parquet` files (non-stationary only)
    """
    want = f"_{kind}" if kind else None
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*"))):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m or m.group("kind") != want:
            continue
        found.append(
            {
                "path": path,
                "algo": m.group("algo"),
                "limit": m.group("limit") or "none",
                "rounds": int(m.group("rounds")),
                "perturbation": bool(m.group("perturbation")),
                "perturb_rounds": [int(x) for x in re.findall(r"\d+", m.group("round_str") or "")],
                "label": condition_label(m.group("algo"), m.group("limit")),
            }
        )
    return found


def load(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one results file, asking for only the columns we need."""
    if path.endswith(".parquet"):
        if columns is not None:
            import pyarrow.parquet as pq

            have = set(pq.read_schema(path).names)
            columns = [c for c in columns if c in have]
        return pd.read_parquet(path, columns=columns)
    df = pd.read_csv(path)
    return df[[c for c in columns if c in df.columns]] if columns else df


def apply_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Restrict a non-stationary results frame to one phase of the run.

    The perturbation runner tags every row `pre`, `post1` or `post2` — before
    the first role swap, between the two swaps, and after the second. Analysing
    a whole 300-round run as if it were one thing is usually wrong: performance
    at round 300 is "how well it recovered from the second swap", not "how well
    it solved the problem".

    On stationary results there is no `phase` column and nothing to do.
    """
    if phase in (None, "all") or "phase" not in df.columns:
        return df
    out = df[df["phase"] == phase]
    if out.empty:
        raise SystemExit(
            f"No rows for phase {phase!r}. Available: "
            f"{sorted(df['phase'].unique())}"
        )
    return out


def phase_suffix(phase: str | None) -> str:
    """Filename suffix so per-phase outputs do not overwrite each other."""
    return "" if phase in (None, "all") else f"_{phase}"


def phase_note(phase: str | None) -> str:
    """Human-readable tag for plot titles."""
    return "" if phase in (None, "all") else f"  [phase: {phase}]"


def describe(files: list[dict]) -> str:
    """One line about what was found, for the scripts to print."""
    kinds = {"perturbation" if f["perturbation"] else "stationary" for f in files}
    limits = sorted({f["limit"] for f in files})
    return (
        f"{len(files)} condition(s), {'/'.join(sorted(kinds))}, "
        f"{files[0]['rounds']} rounds, switching limit(s): {', '.join(limits)}"
    )


def final_round(df: pd.DataFrame) -> int:
    return int(df["round"].max())


def rounds_to(mean_curve: pd.Series, threshold: float) -> float:
    """First round at which the mean curve reaches `threshold`, else NaN."""
    hit = mean_curve[mean_curve >= threshold]
    return float(hit.index[0]) if len(hit) else float("nan")


def ranking_by_final(df: pd.DataFrame) -> list[str]:
    """Conditions ordered best-to-worst by mean performance at the last round."""
    last = final_round(df)
    return (
        df[df["round"] == last]
        .groupby("condition")["performance"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def cliffs_delta(a, b) -> float:
    """Non-parametric paired effect size in [-1, 1]; 0 = interchangeable."""
    d = np.asarray(a) - np.asarray(b)
    return float((np.sum(d > 0) - np.sum(d < 0)) / d.size)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down correction for a family of p-values."""
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj
