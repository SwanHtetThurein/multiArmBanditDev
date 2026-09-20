# Running this benchmark (non-stationary)

Everything here is plain Python. There is no build step and no configuration
file to edit.

> This file is the quickstart. Two companion references go deeper:
> **`INPUTS.md`** documents every flag and setting, and **`OUTPUTS.md`**
> documents every output file and every column.

## 1. Dependencies

```bash
pip install numpy scipy pyarrow tqdm
```

`numpy` and `pyarrow` are required — results and traces are Parquet files.
`scipy` is used by one algorithm (`combo`). `tqdm` draws the progress bar;
without it the sweep still runs, just silently.

For analysis, `pandas` reads Parquet directly. `duckdb` is optional but very
convenient: it runs SQL straight against the files, with no import step.

## 2. Check the setup

```bash
python3 verify_setup.py
```

Takes under a minute and confirms the six guarantees below. Run it once after
copying the folder to a new machine. Exit code 0 means everything passed.

## 3. Run

```bash
# one algorithm, using all cores but one
python3 run_perturbation_experiment.py --algorithm bocs

# choose the core count yourself
python3 run_perturbation_experiment.py --algorithm bocs --n-cores 12

# every algorithm, in turn
python3 run_perturbation_experiment.py --all --n-cores 12

# under a switching limit (cap on role changes per round)
python3 run_perturbation_experiment.py --all --n-cores 12 --switch-limit flat --max-changes 2

# the whole suite, then aggregate and plot
N_CORES=12 bash run_full_sweep.sh
```

`run_experiment.py` is also here: it runs the same 26 algorithms on the
*stationary* version of this protocol, for comparison.

## 4. Stopping and resuming

**Press Ctrl-C whenever you like.** Progress is written to disk as each run
finishes. To carry on, run *the exact same command again* — it reports how
many runs were already done and continues from there.

This survives a hard crash, a power cut, or a `kill -9`, not just Ctrl-C. A
run that was half-written when the process died is detected and redone.

`--no-resume` throws away saved progress and starts that sweep over.

With `--all`, resuming also skips whole algorithms that already finished.

## 5. What is guaranteed

| | how it is guaranteed |
|---|---|
| Every algorithm faces the same 500 problems | They are sampled once into `bandit_settings_n500_seed20240501.json` and loaded from that file. The file carries a SHA-256 digest of its own contents, printed at the start of every run. |
| The file has not been tampered with | `verify_setup.py` re-derives the 500 settings from the recorded seed and compares digests. |
| Every algorithm faces the same reward noise | Each run's RNG is seeded from `(master seed, setting, noise level)` — never from the algorithm — so the noise sequence is a property of the problem, not of who is solving it. Comparisons are paired. |
| Results do not depend on the core count | Same reason: seeding is independent of how work was scheduled. 1 core and 16 cores produce byte-identical CSVs. |
| Results do not depend on interruptions | Also the same reason. A crashed-and-resumed run is byte-identical to an uninterrupted one. |
| Runs are not silently lost | The final CSV is only written once every expected run is present. |

## 6. The sampling protocol

500 settings, each drawn independently:

1. `n_bandits` uniform from {3, 6, 9}
2. each bandit's arm count uniform from 2–5
3. every bandit is `ongoing`
4. plus one `initial_bias` and one `optimal_arm`

One run = one (setting, noise level) pair, over **300 rounds**, with the
hidden optimum swapped after rounds 101 and 201. With 6 noise levels that is
**500 × 6 = 3000 runs per algorithm**. There are no repetitions — 500
independent settings give more variety than repeating a smaller set would.

Example setting: `<6, (1,4,3,5,3,2), ongoing>` in the original note had a `1`
in it; arm counts here are 2–5 as the written rule says. A 1-arm dimension has
no choice to make, so it would inflate every algorithm's score without any
learning happening.

## 7. Output

Five files per algorithm, in `Unconstrained Perturbation Results/`, all named
from the run's configuration (no timestamp, so a resumed run finds its own
progress). Writing `<stem>` for
`<algo>_limit-<mode>_perturbation_rounds300_settings500_perturbs2_t101_t201_seed42`:

```
<stem>_trace.parquet     full detail, one row per round
<stem>.parquet           results projection
<stem>_summary.parquet   per-run recovery metrics, four per perturbation
<stem>_manifest.json     provenance
```

The trace and results here carry three extra columns: `phase` (`pre`, `post1`,
`post2`), `baseline` (that run's pre-perturbation performance), and
`optimum_version` (0 before the first swap, 1 after it, 2 after the second) —
so you can tell which answer was in force at any round.

**The trace** is one row per round of every run — the raw record of the
simulation:

| column | meaning |
|---|---|
| `run_id`, `setting_id`, `n_bandits`, `noise_level`, `noise_pct`, `round` | which run, which round |
| `requested_team` | the team the algorithm **asked** to play |
| `played_team` | the team it was **allowed** to play |
| `n_requested_changes`, `n_played_changes` | how many role changes it wanted vs got |
| `blocked_dims` | exactly which requested changes the constraint refused |
| `budget`, `allowance` | the cap in force that round (null when unconstrained) |
| `reward` | the noisy signal it received |
| `predict_best`, `n_correct`, `performance` | its recommendation and its score |
| `hamming_to_optimum`, `cum_changes` | distance from the answer, churn so far |
| `diag_*` | that algorithm's own internals — see below |

**The results file** is a six-column projection of the trace. It holds nothing
new; it exists because reading six columns out of a twenty-column file on every
plot is wasteful.

**The manifest** records how the run was produced: seeds, settings digest,
switch-limit mode, algorithm constants, core count, wall time, library
versions. This is what lets you attribute a file months later.

A `_parts/` directory appears while a sweep runs — that is the checkpoint, one
Parquet file per finished run. It is deleted automatically on completion.

### Per-algorithm diagnostics

At `--trace full` (the default) every algorithm also records its own internal
state each round, as `diag_*` columns. A few examples:

| algorithm | records |
|---|---|
| `dreamteam` | `n_condemned` — arms killed by the +1000 beta penalty, the most direct measure of what that modification does |
| `casmopolitan` | `trust_radius`, `succ_count`, `fail_count` — the trust region growing and collapsing |
| `cocabo` | `played_from_exp3`, `mean_exp3_entropy` — which mechanism chose, and how committed the agents are |
| `sa` | `temperature`, `incumbent_score` |
| GP arms | `rho`, `eta`, `sf2` — the fitted kernel hyperparameters |
| `bayesgap` | `best_gap` — the quantity it is minimising |

Full model state (a posterior weight vector, a covariance matrix) is
deliberately **not** recorded: for the BOCS-family surrogates that is ~470
numbers per round, over half a gigabyte per algorithm for one field.

### Trace levels

`--trace full` (default), `basic` (drops the `diag_*` columns), `off` (keeps
only the results columns and writes no trace file). **All three produce
identical results** — the level changes what is kept, never what is run.

### Sizes, measured

| | per algorithm | all 26 |
|---|---|---|
| trace (300 rounds) | ~27 MB | ~0.69 GB |
| results + summary | a few MB | negligible |

### Reading the files

```python
from tracing import read
df = read("Global results/bocs_..._trace.parquet",
          columns=["round", "performance", "n_played_changes"],
          noise_pct=40)
```

Filter on `noise_pct` (an integer, 0–100) rather than `noise_level`. Both hold
the same information; `noise_level` is a float64 and the standard levels do
compare exactly, but float equality is fragile in general and the integer
column cannot fail that way.

With DuckDB, no import step at all:

```sql
SELECT round, avg(performance) FROM 'Global results/*_trace.parquet'
WHERE noise_pct = 40 GROUP BY round ORDER BY round;
```

## 8. Disk space

Peak usage during a sweep is roughly 2× the final size, because the checkpoint
parts exist alongside the compacted file for a moment. Budget about **2 GB**
per switch-limit condition.

## 9. Reading the results in this folder

**Dimensions vary, and that changes what a perturbation means.** Flipping one
dimension is a 1/3 shock on a 3-dimension setting and a 1/9 shock on a
9-dimension one. Group by `n_bandits` before comparing dips or recovery times;
the recovery metrics already use each run's own dimension count.

**Do not use `--switch-limit parabolic` for recovery comparisons.** Its
allowance is one arc sized to the whole run and knows nothing about the
perturbations. At the swaps it is nearly equal (1.79 and 1.77), but the
recovery windows sit on opposite sides of its peak: rounds 102–201 carry a mean
allowance of 1.93, rounds 202–300 only 1.02, decaying to zero. Recovery from
the second swap gets about half the budget of the first, which confounds
exactly the comparison this experiment is for. Use `flat`.

## 10. Runtime

300 rounds per run, divided by the number of cores. The
heavy algorithms (`bocs`, `sts`, `smac`) dominate; the cheap ones finish in
minutes. Start with one algorithm to calibrate before launching `--all`.

Two notes on cores:

- The default is every core but one, so the machine stays usable.
- Each worker is pinned to a single BLAS thread. Without that, workers fight
  each other for threads and *more* cores makes the sweep slower.
