# ANALYSIS.md — turning a finished run into tables and figures

The runners (`run_experiment.py`, `run_perturbation_experiment.py`) produce raw
data and nothing else. The scripts described here sit on top of that data.

This file is identical in the stationary and non-stationary folders, and so are
the five scripts it describes, so anything you learn in one folder transfers to
the other. The two perturbation-only scripts are covered at the end.

All of them are read-only: they never modify the results, so re-run them as
often as you like.

---

## Quick start

```bash
# stationary folder
python aggregate_results.py  --results_dir "Global results" --output_dir Analysis
python graph_results.py      --results_dir "Global results" --output_dir Graphs
python compare_algorithms.py --results_dir "Global results" --output_dir Analysis
python analyze_traces.py     --results_dir "Global results" --output_dir Analysis

# non-stationary folder — same scripts, plus --phase
python aggregate_results.py  --results_dir "Unconstrained Perturbation Results" \
    --output_dir Analysis --phase pre
python graph_results.py      --results_dir "Unconstrained Perturbation Results" \
    --output_dir Graphs --phase post1 --skip_per_algorithm
```

The first three read only the small results files and take seconds.
`analyze_traces.py` reads the large `_trace.parquet` files and takes a few
minutes; it caches its work (see below), so an interrupted run resumes.

`analysis_common.py` holds the parts all four share: the filename parser, the
condition labels, the phase filter and the statistics helpers. It is not run
directly, but the other four will not work without it.

---

## `--phase`: which part of a perturbation run you mean

A perturbation run is three experiments in one. The runner tags every row:

| phase | rounds | what it measures |
|---|---|---|
| `pre` | before the first role swap | how well the algorithm solves a *fixed* problem |
| `post1` | between the two swaps | how well it recovers from one change |
| `post2` | after the second swap | how well it recovers from a second change |

`--phase all` (the default) treats the whole run as one stretch. That is usually
**not** what you want for a headline number: performance at round 300 answers
"how well did it recover from the last swap", not "how well does it solve the
problem". Run the phases separately and compare them — the difference between
them is the interesting result, and outputs are suffixed (`_pre`, `_post1`,
`_post2`) so they never overwrite each other.

On stationary results there is no `phase` column and the flag does nothing.

---

## 1. `aggregate_results.py` — the summary table

One row per (algorithm, switching-limit, noise level), plus a pooled row per
algorithm where `noise_level = -1`.

| column | meaning |
|---|---|
| `final_perf_mean/std/sem/median` | fraction of roles the recommended team gets right at the last round. 1.0 = the optimal team was identified exactly |
| `simple_regret_mean` | `1 - final_perf_mean` |
| `pct_exact_optimal` | % of runs that ended with a perfect team |
| `pct_zero_correct` | % of runs that ended with nothing right |
| `auc_mean/std` | mean performance across *all* rounds in the window. Rewards getting good early, not just ending well |
| `perf_at_25pct_rounds`, `perf_at_50pct_rounds` | where the mean curve stands a quarter and half way through the window |
| `rounds_to_50pct / 75pct / 90pct / 95pct` | first round at which the mean curve crosses that absolute level. Blank (`NaN`) when it never does |
| `final_perf_nb3 / nb6 / nb9` | final score split by team size |
| `baseline_mean`, `gap_to_baseline` | perturbation runs only: the pre-swap level and how much of it is still missing |

Also writes `noise_robustness_<ts>.csv` — performance at zero noise, at maximum
noise, the drop between them, and the slope of a line fitted through the six
levels. A shallow slope means the algorithm degrades gracefully.

And `summary_all_algorithms_<ts>.md`: the ranking, the robustness table and a
per-noise grid, paste-ready for a draft.

## 2. `graph_results.py` — the figures

| file | what it shows |
|---|---|
| `learning_curves_all.png` | every algorithm's mean curve, one panel per noise level, coloured best-to-worst. The one-figure overview |
| `learning_<algo>.png` | one algorithm, mean ± 1 sd band, one panel per noise level |
| `final_ranking.png` | horizontal bar ranking with ±1 sem, one panel per noise level |
| `noise_sensitivity.png` | final score against noise, top 8 highlighted, rest in grey, `random` dashed |
| `heatmap_final_performance.png` | algorithms × noise levels, numbers printed in the cells |
| `team_size_scaling.png` | final score against 3 / 6 / 9 roles |
| `sample_efficiency.png` | speed against quality: rounds to reach 90% of its *own* final score on the x-axis, final score on the y-axis. Top-left is fast and good |

On perturbation data the swap rounds are drawn as red dashed lines.

`--skip_per_algorithm` suppresses the individual curve plots. In the
non-stationary folder they largely duplicate `perturbation_<algo>.png`, so
skipping them there is the normal choice.

The script never holds the raw rows: each results file is reduced to per-round
means and last-round rows as it is read. 26 algorithms × 3000 runs × 300 rounds
is 23 million rows, and loading that at once is enough to get the process killed
on a normal laptop.

## 3. `compare_algorithms.py` — is the gap real?

Every algorithm saw the same sampled settings with the same measurement noise,
so comparisons are **paired**, which is far more sensitive than comparing
independent means. This script exploits that:

* `pairwise_vs_best_<ts>.csv` — every algorithm against the overall winner:
  mean paired difference, 95% CI, Wilcoxon signed-rank p, Holm-corrected p,
  and Cliff's delta (effect size in [-1, 1]).
* `winrate_matrix_<ts>.csv` / `winrate_matrix.png` — for every ordered pair, the
  fraction of settings on which the row algorithm beat the column algorithm
  (ties count half).
* `significance_vs_best.png` — the gap behind the winner with 95% CIs; red marks
  the gaps that survive Holm correction.

`--noise 0.4` restricts everything to one noise level.

Note the difference between the two views: a *significant* gap and a *large* gap
are not the same thing. With 3000 paired cells a 0.017 difference can be
overwhelmingly significant while the head-to-head win rate is only 0.54 — the
better algorithm wins slightly more often, not always. Report both.

## 4. `analyze_traces.py` — how each algorithm searches

The results files say how good the recommendation was. The traces say what the
algorithm actually *did*.

`trace_summary_<ts>.csv`, one row per (algorithm, noise level) plus a pooled row:

| column | meaning |
|---|---|
| `churn_mean`, `churn_late_mean` | roles changed per round, overall and over the last 20 rounds |
| `churn_late_frac_of_team` | the same as a fraction of team size, so 3-role and 9-role settings are comparable |
| `hamming_mean`, `hamming_late_mean` | how many roles the team it actually *played* got wrong |
| `pct_rounds_played_optimum` | % of late rounds spent playing the optimal team outright |
| `pct_rounds_blocked_by_limit` | % of rounds where it asked for more changes than the switching limit allowed. Always 0 under `--switch-limit none` |
| `requested_changes_mean` | what it asked for, before any limit was applied |

Figures: `churn_over_time.png`, `hamming_over_time.png`,
`churn_vs_performance.png`, and `blocked_by_limit.png` (written only when a
switching limit was in force).

**Caching.** Each trace file's summary is written to `<output_dir>/.trace_cache`
as soon as it is computed, keyed by the file's name, size and modification time.
Re-running picks up where it stopped, which matters when the scan outlives a
shell's time limit. `--no_cache` forces a full recompute; deleting the directory
does the same. A phase and a noise filter are part of the key, so
`--phase post1` is cached separately from `--phase all`.

**Read `churn_vs_performance.png` carefully.** High late churn does not mean an
algorithm is failing. Some keep *playing* exploratory teams while the team they
*recommend* is already settled — that is deliberate exploration, not
instability. Near-zero late churn is the more worrying signal: the search has
frozen, and if it froze on the wrong team it can no longer escape.

## 5. `aggregate_perturbation.py` and `graph_perturbation.py` (non-stationary only)

These predate the four above and answer a different question: not "how good did
it get" but "how fast did it come back".

* `aggregate_perturbation.py` → `perturbation_summary_all_algorithms_<ts>.csv`
  and `.md`: per algorithm and noise level, the pre-swap `baseline`, the
  `recovery_threshold_rounds_{1,2}` (rounds to climb back to a fraction of
  baseline), `time_to_baseline_rounds_{1,2}`, `max_dip_{1,2}` and
  `recovered_pct_{1,2}`.
* `graph_perturbation.py` → `perturbation_<algo>.png` (per-round curve with the
  swap rounds marked, one panel per noise level) and `recovery_summary.png`
  (median recovery time per algorithm, grouped by noise and by which swap).

They complement the four general scripts rather than replacing them: recovery
*speed* from these, recovery *quality* from `aggregate_results.py --phase`.

**One caveat.** Recovery metrics are relative to each algorithm's own baseline,
so an algorithm that never learned anything "recovers" instantly and perfectly.
`random` scores 100% recovery in about 2 rounds. Always read a recovery number
next to the absolute performance it recovered *to*.

---

## Reading the numbers responsibly

* **`random` is the floor.** Its score is roughly `1 / mean(arms per role)`.
  Anything near that line has not learned.
* **Noise is multiplicative**: a measurement's standard deviation is
  `noise_level × true_quality`, so better teams are measured more noisily.
  Absolute scores are not comparable across noise levels — compare algorithms
  *within* a level, or use the robustness table.
* **Pooled rows mix noise levels.** The `noise_level = -1` row averages all six.
  Fine as a headline, but the ranking changes as noise rises; the per-noise grid
  is where the real story is.
* **`rounds_to_90pct` is on the mean curve**, not per run, and is blank when the
  mean curve never reaches 0.9. With noise pooled in, most algorithms never do.

## Not covered yet

* **Switching-limit comparison.** Everything here already handles several
  `--switch-limit` conditions side by side — the `condition` column and the
  `[flat]` / `[parabolic]` labels exist for exactly that — but there is nothing
  to compare until a limited sweep has been run. Re-run into a separate results
  directory, copy both sets of files into one directory, and every script treats
  them as separate conditions automatically.
* **Per-algorithm diagnostics.** The traces carry `diag_*` columns whose meaning
  differs per algorithm (posterior spread, pool size, observation counts).
  Plotting those is a per-algorithm question a generic script cannot do well.
* **Cost.** Wall-clock per algorithm sits in each `_manifest.json` but is not
  pulled into any table.
