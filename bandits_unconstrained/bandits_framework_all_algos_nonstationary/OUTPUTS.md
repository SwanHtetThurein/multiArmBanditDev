# OUTPUTS — every file, every column (non-stationary)

A complete reference for what a sweep produces and what each column means.
Written for someone who has never seen this benchmark before.

Contents: [the files](#the-files) · [opening them](#opening-them) ·
[the trace](#the-trace-file) · [the results](#the-results-file) ·
[the manifest](#the-manifest) · [diagnostics](#per-algorithm-diagnostics) ·
[the settings file](#the-settings-file) · [worked examples](#worked-examples)

---

## The files

One completed algorithm produces four files in
`Unconstrained Perturbation Results/`. Writing `<stem>` for
`bocs_limit-flat2_perturbation_rounds300_settings500_perturbs2_t101_t201_seed42`:

```
<stem>_trace.parquet     the full record, one row per round
<stem>.parquet           the scores
<stem>_summary.parquet   per-run recovery metrics
<stem>_manifest.json     how it was run
```

Running the aggregator and grapher adds:

```
perturbation_summary_all_algorithms_<timestamp>.csv   cross-algorithm table
perturbation_summary_all_algorithms_<timestamp>.md    the same, as markdown
Graphs/perturbation_<algo>[_limit-<mode>].png         per-algorithm curves
Graphs/recovery_summary.png                           cross-algorithm summary
```

The filename encodes the configuration, so conditions never overwrite one
another:

| part | meaning |
|---|---|
| `bocs` | the algorithm |
| `limit-flat2` | switching limit: `flat` mode, cap 2. `limit-none` means no cap |
| `perturbation_rounds300` | rounds per run |
| `settings500` | how many of the sampled problems were used |
| `perturbs2` | how many times the answer was swapped |
| `t101_t201` | the rounds after which it was swapped |
| `seed42` | the master seed |

There is no timestamp, deliberately — a resumed run has to find its own
progress again.

A `_parts/` directory appears **while a sweep is running**. That is the
checkpoint: one small Parquet file per finished run. It is deleted
automatically when the algorithm completes. If you see one left behind, that
sweep did not finish.

### Sizes

| | per algorithm | all 26 |
|---|---|---|
| trace, 300 rounds | ~27 MB | ~0.69 GB |
| results + summary | a few MB | negligible |

Budget about twice the final size while a sweep runs, since the parts and the
compacted file briefly coexist.

---

## Opening them

These are **Parquet** files, not CSV. Parquet stores data by column, so reading
two columns out of twenty touches only those two columns' bytes — about 29×
faster than CSV on a file this shape, at a fifth of the size.

```python
import pandas as pd
df = pd.read_parquet("Unconstrained Perturbation Results/bocs_..._trace.parquet")
```

Or with the helper in this folder, which also filters without loading
everything:

```python
from tracing import read
df = read("Unconstrained Perturbation Results/bocs_..._trace.parquet",
          columns=["round", "performance", "n_played_changes"],
          noise_pct=40)
```

Or SQL, with no import step at all (`pip install duckdb`):

```sql
SELECT round, avg(performance)
FROM 'Unconstrained Perturbation Results/*_trace.parquet'
WHERE noise_pct = 40
GROUP BY round ORDER BY round;
```

> **`noise_pct` is the safe column to filter on.** Both columns hold the same
> information, but `noise_level` is a float. It is stored as float64 and the
> standard levels compare exactly, so `noise_level == 0.4` does work — but
> float equality is fragile in general: a level that arrives via arithmetic
> rather than as a literal, or a file written by a tool that narrowed the
> column to float32, will match nothing and report zero rows rather than an
> error. `noise_pct` is an integer (0–100) and has no such failure mode.

To convert to CSV for a tool that needs it:

```python
pd.read_parquet("....parquet").to_csv("out.csv", index=False)
```

---

## The trace file

**One row per round of every run.** This is the raw record of the simulation —
everything that happened, in order. 500 settings × 6 noise levels × 300 rounds
= 900,000 rows.

### Identity — which run, which round

| column | type | meaning |
|---|---|---|
| `run_id` | int32 | run number within this sweep, 0-based. One run = one (setting, noise) pair. Unique within the file. |
| `setting_id` | int32 | which of the 500 sampled problems, 0–499. Look it up in the settings JSON to see the team size, arm counts and hidden answer. |
| `n_bandits` | int8 | how many roles this setting has: 3, 6 or 9. **Varies between settings** — see the warning below. |
| `noise_level` | float64 | the noise level, 0.0–1.0 |
| `noise_pct` | int16 | the same as an integer, 0–100. Filter on this one. |
| `round` | int32 | round number within the run, 1-based |

### The decision — what it wanted vs what it got

| column | type | meaning |
|---|---|---|
| `requested_team` | list of int8 | the team the algorithm **asked** to play. One arm index per role, length `n_bandits`. |
| `played_team` | list of int8 | the team it was **allowed** to play. Identical to `requested_team` when no cap is in force, or when the cap did not bite. |
| `n_requested_changes` | int8 | how many roles it wanted to change from last round |
| `n_played_changes` | int8 | how many it actually changed |
| `blocked_dims` | list of int8 | the indices of the roles whose requested change was **refused**. Empty when nothing was blocked. |
| `budget` | float32 | the real-valued allowance for this round. Null when no limit is in force. Constant under `flat`; a parabola under `parabolic`. |
| `allowance` | int16 | the integer cap actually applied, after randomized rounding. Null when unconstrained. `n_played_changes` never exceeds it. |

`n_requested_changes > n_played_changes` marks a round where the constraint
bit. `blocked_dims` says exactly which roles it stopped.

### The outcome

| column | type | meaning |
|---|---|---|
| `reward` | float64 | the noisy score for `played_team`, in [0, 1]. This is the **only** feedback the algorithm receives. |
| `predict_best` | list of int8 | its current best guess at the hidden answer — which may differ from what it played |
| `n_correct` | int8 | how many roles of `predict_best` match the hidden answer |
| `performance` | float64 | `n_correct / n_bandits`. **This is the metric the benchmark scores.** |
| `hamming_to_optimum` | int8 | how many roles of `played_team` are wrong. Distance travelled, as opposed to distance recommended. |
| `cum_changes` | int32 | total role changes so far in this run — cumulative churn |

Note that `predict_best` and `played_team` are separate. An algorithm may play
an exploratory team while recommending a different one, and most do.

> **`performance` is a fraction, so it compares across settings — but read it
> carefully.** Getting 2 of 3 roles right scores 0.667, the same as 6 of 9, and
> they are not equally hard. Group by `n_bandits` before drawing conclusions.

### The perturbation columns

Three columns exist here and not in the stationary folder, because the hidden
answer moves mid-run:

| column | type | meaning |
|---|---|---|
| `phase` | string | `pre` (before the first swap), `post1`, `post2`, … |
| `baseline` | float64 | this run's mean performance over the baseline window (default rounds 80–100), i.e. the level it had reached just before the first swap. Constant within a run. |
| `optimum_version` | int8 | 0 before any swap, 1 after the first, 2 after the second. Tells you which answer was in force. |

### `diag_*`

At `--trace full` (the default), each algorithm's own internal state, one
column per quantity. Which columns exist depends on the algorithm — see
[below](#per-algorithm-diagnostics).

---

## The results file

A projection of the trace holding only what you plot. It contains nothing the
trace does not; it exists because reading six columns out of a twenty-column
file on every plot is wasteful.

```
run_id, setting_id, n_bandits, noise_level, round, performance
```

Plus `phase` and `baseline`. All columns mean exactly what they mean in the
trace.

Written even at `--trace off`.

---

## The summary file

`<stem>_summary.parquet` — **one row per run**, with recovery metrics for each
perturbation event. The suffix `_1` refers to the first swap, `_2` to the
second.

| column | type | meaning |
|---|---|---|
| `algorithm` | string | the algorithm, with its switch-limit condition, e.g. `bocs [flat2]` |
| `setting_id`, `n_bandits`, `noise_level`, `run_id` | | as in the trace |
| `baseline` | float64 | performance reached just before the first swap |
| `recovery_threshold_rounds_i` | float64 | rounds after swap *i* until performance reached the achievable ceiling. **NaN** if it never did. |
| `time_to_baseline_rounds_i` | float64 | rounds until it regained the full pre-swap `baseline`. NaN if never. |
| `max_dip_i` | float64 | the largest `1 − performance` in the 100 rounds after the swap |
| `recovered_within_horizon_i` | bool | whether the threshold was reached at all |

**The ceiling is not 1.0.** After *k* swaps each flipping *p* dimensions, the
best attainable is `(n_bandits − k·p) / n_bandits` of the original baseline —
for one flip on 9 roles that is 8/9 ≈ 0.889. Each run's own `n_bandits` is used,
because dimensions vary between settings.

> **`recovery_threshold_rounds` degenerates for good algorithms.** An algorithm
> sitting at `baseline = 1.0` lands on 8/9 the instant one role is flipped, and
> is scored as having "recovered" in 1 round. The better the algorithm, the
> less this metric says. `time_to_baseline_rounds` and `max_dip` carry the real
> information. The threshold metric is kept for comparability with the earlier
> design, not because it is the right summary.

---

## The manifest

A small JSON file recording how the run was produced — so a result found months
later can be attributed without guesswork.

| key | meaning |
|---|---|
| `algorithm` | which algorithm |
| `algorithm_constants` | its tunable class constants at run time (e.g. `N_INIT`, `ALPHA`) |
| `total_rounds`, `noise_levels`, `n_settings` | the sweep's shape |
| `master_seed` | `--seed`: controls the noise |
| `settings_seed`, `settings_file` | which problem set |
| `settings_digest` | SHA-256 of the 500 problems. **Two result sets with the same digest provably faced the same problems.** |
| `switch_limit`, `max_changes` | the constraint in force |
| `trace_level` | `off` / `basic` / `full` |
| `n_runs`, `n_trace_rows` | size of the output |
| `n_cores`, `wall_seconds` | what it cost |
| `written_at`, `host`, `platform`, `python`, `numpy`, `pyarrow` | provenance |

This folder's manifest also records `perturbation_rounds`, `p_perturb_dims`
and `baseline_window` — the perturbation schedule that produced the run.

---

## Per-algorithm diagnostics

At `--trace full`, every algorithm records its own internals each round as
`diag_*` columns. Common ones: `n_obs` (observations so far), `best_y` (best
reward seen), `mean_y`.

The interesting ones are algorithm-specific:

- **`dreamteam`** — `n_condemned`, `frac_condemned`: arms whose beta has grown
  past 1000. Under the published Beta update this essentially cannot happen, so
  it should stay at zero. It is recorded because the modified variant this
  package used to carry drove it up sharply.
- **`casmopolitan`** — `trust_radius`, `succ_count`, `fail_count`: the trust
  region growing, shrinking and restarting. A run where the radius keeps
  collapsing is one where the model never earned the right to be trusted.
- **`cocabo`** — `played_from_exp3` (which mechanism chose this round),
  `mean_exp3_entropy` (how committed the per-dimension agents have become).
- **`sa`** — `temperature`, `incumbent_score`. The temperature schedule is why
  SA stops exploring late.
- **`bayesgap`** — `best_gap`: its own estimate of how much better the true
  best arm could be than the one it would recommend. The natural convergence
  signal for that arm.
- **`bocs_hs`** — `sigma2`, `tau2`, `frac_near_zero`: how sparse the horseshoe
  prior actually made the weights, which is the thing that arm exists to
  measure against plain ridge shrinkage in `bocs`.
- **GP arms** — `rho` (how far similarity reaches), `eta` (fitted noise),
  `sf2`. Both moving is the model changing its mind.

Full table:

| algorithm | `diag_*` columns |
|---|---|
| `bayesgap` | `best_gap`, `best_y`, `beta`, `n_features`, `n_obs`, `pool_size` |
| `bocs` | `best_y`, `last_y`, `mean_y`, `n_features`, `n_obs` |
| `bocs_hs` | `best_y`, `beta_absmax`, `beta_absmean`, `frac_near_zero`, `n_features`, `n_obs`, `sigma2`, `tau2` |
| `bootnn` | `best_y`, `in_dim`, `n_models`, `n_obs`, `trained` |
| `casmopolitan` | `best_y`, `eta`, `fail_count`, `n_obs`, `rho`, `sf2`, `succ_count`, `trust_radius` |
| `cocabo` | `best_y`, `eta`, `mean_exp3_entropy`, `mean_gamma`, `min_exp3_entropy`, `n_obs`, `played_from_exp3`, `sf2` |
| `combo` | `best_y`, `fitted_n`, `log_sf2`, `log_sn2`, `mean_log_beta`, `n_obs`, `rounds_since_fit`, `std_log_beta`, `ymean` |
| `combo_slice` | `best_y`, `n_hyper_samples`, `n_obs`, `rounds_since_refit`, `theta_mean`, `theta_std` |
| `cts` | `max_posterior`, `mean_posterior`, `min_posterior`, `n_base_arms`, `t`, `total_evidence` |
| `cucb` | `max_count`, `max_radius`, `mean_count`, `min_count`, `n_base_arms`, `t` |
| `dreamteam` | `frac_condemned`, `max_posterior`, `mean_posterior`, `min_posterior`, `n_arms`, `n_condemned` |
| `glm_fpl` | `best_y`, `fitted`, `mean_y`, `n_dims`, `n_dual_coefs`, `n_obs`, `perturb_scale`, `ridge` |
| `gp_nei`, `gp_onehot`, `gp_ts`, `gp_ucb` | `best_y`, `eta`, `n_obs`, `rho`, `rounds_since_tune`, `sf2`, `ymean` |
| `kg` | `best_y`, `mean_y`, `n_features`, `n_obs`, `pool_size`, `space_size` |
| `linucb` | `alpha`, `best_y`, `mean_y`, `n_features`, `n_obs` |
| `neurallinear` | `best_y`, `in_dim`, `mean_y`, `n_features`, `n_obs`, `trained` |
| `ols` | `best_neighbor_score`, `incumbent_score`, `phase`, `queue_remaining` |
| `purexp` | `best_y`, `last_y`, `mean_y`, `n_features`, `n_obs` |
| `random` | _(none — the baseline holds no state)_ |
| `regevo` | `pop_best`, `pop_mean`, `pop_worst`, `population_size` |
| `sa` | `incumbent_score`, `round`, `temperature` |
| `smac` | `best_y`, `fitted_n`, `mean_y`, `n_obs`, `n_trees` |
| `sts` | `best_y`, `epsilon`, `mean_y`, `n_features`, `n_obs` |

Model state proper — a posterior weight vector, a covariance matrix — is
deliberately not recorded. For the BOCS family that is ~470 numbers per round,
over half a gigabyte per algorithm for one field.

---

## The settings file

`bandit_settings_n500_seed20240501.json` — the 500 problems every algorithm
faced. **Keep it with your results**; it is what makes the trace
interpretable, since `setting_id` is a pointer into it.

```json
{
  "version": 1,
  "seed": 20240501,
  "n_settings": 500,
  "protocol": { "bandit_choices": [3, 6, 9], "arm_range": [2, 5], ... },
  "digest": "e3dcdeae8226...",
  "settings": [
    {
      "setting_id": 0,
      "n_bandits": 6,
      "arm_counts": [4, 3, 5, 2, 3, 2],
      "bandit_types": ["ongoing", "ongoing", ...],
      "initial_bias": [1, 0, 4, 1, 2, 0],
      "optimal_arm":  [2, 0, 1, 1, 2, 1]
    },
    ...
  ]
}
```

`arm_counts[d]` is how many candidates role `d` has, so a valid arm index for
that role is `0 .. arm_counts[d] - 1`. `optimal_arm` is the hidden answer the
algorithm is trying to find; it never sees it.

---

## Worked examples

**Learning curve at one noise level**

```python
from tracing import read
df = read("Unconstrained Perturbation Results/bocs_..._trace.parquet",
          columns=["round", "performance"], noise_pct=40)
df.groupby("round").performance.mean().plot()
```

**How often did the switching limit bite?**

```python
df = read("...limit-flat2..._trace.parquet",
          columns=["round", "n_requested_changes", "n_played_changes"])
blocked = df.n_requested_changes > df.n_played_changes
print(f"{blocked.mean():.1%} of rounds had a change refused")
```

**Which roles got blocked most often**

```python
from collections import Counter
df = read("...limit-flat2..._trace.parquet", columns=["blocked_dims"])
print(Counter(d for row in df.blocked_dims for d in row))
```

**Compare two algorithms fairly** — same problems, same noise, so this is a
paired comparison:

```python
a = read("bocs_...parquet",   columns=["run_id","round","performance"])
b = read("linucb_...parquet", columns=["run_id","round","performance"])
final = lambda d: d[d.round == d.round.max()].set_index("run_id").performance
diff = final(a) - final(b)
print(diff.mean(), diff.std())
```

**Split by team size, which you should do before any absolute claim**

```python
df = read("..._trace.parquet", columns=["n_bandits", "round", "performance"])
df[df.round == 100].groupby("n_bandits").performance.mean()
```

**Recovery curve around a swap**

```python
df = read("..._trace.parquet", columns=["round", "performance", "phase"],
          noise_pct=40)
curve = df.groupby("round").performance.mean()
curve.loc[90:140].plot()      # the dip at 102 and what follows
```

**Did the switching limit slow recovery?** Compare the same algorithm with and
without a cap, restricted to the rounds after a swap:

```python
free = read("..._limit-none_trace.parquet",  columns=["round","performance"])
cap  = read("..._limit-flat2_trace.parquet", columns=["round","performance"])
post = lambda d: d[(d.round > 101) & (d.round <= 201)].groupby("round").performance.mean()
(post(free) - post(cap)).plot()
```

**Confirm two result sets faced the same problems**

```python
import json
d1 = json.load(open("..._manifest.json"))["settings_digest"]
d2 = json.load(open("..._manifest.json"))["settings_digest"]
assert d1 == d2
```
