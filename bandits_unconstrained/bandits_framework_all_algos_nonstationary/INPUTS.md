# INPUTS — every setting, and what it does (non-stationary)

A complete reference for everything you can pass to this benchmark. If you have
never run it before, read `SETUP.md` first for the five-minute version; this
file is the exhaustive one.

Contents: [the commands](#the-commands) · [experiment settings](#experiment-settings) ·
[the switching limit](#the-switching-limit) · [recording detail](#recording-detail) ·
[execution](#execution) · [the settings file](#the-settings-file) ·
[what interacts with what](#what-interacts-with-what)

---

## The commands

| command | what it does |
|---|---|
| `python3 run_perturbation_experiment.py` | **the main entry point here.** Runs the sweep with the hidden answer swapped mid-run. |
| `bash run_full_sweep.sh` | all 26 algorithms, then aggregate and plot. Configured by environment variables, not flags — see below. |
| `python3 aggregate_perturbation.py` | recovery metrics across algorithms, from finished results |
| `python3 graph_perturbation.py` | per-algorithm curves plus a cross-algorithm summary plot |
| `python3 run_experiment.py` | the *stationary* runner, same 26 algorithms, for comparison |
| `python3 verify_setup.py` | self-check: dependencies, determinism, resume. Takes under a minute. Run it after copying the folder to a new machine. |
| `python3 sampling.py` | prints the shared settings file's digest and composition. `--verify` also re-derives it from its seed to prove it was not edited. |

`run_perturbation_experiment.py` needs either `--algorithm NAME` or `--all`.
With neither it prints the list of algorithms and exits.

Every flag below applies to `run_perturbation_experiment.py`. The stationary
`run_experiment.py` in this folder takes the same flags minus the perturbation
ones.

---

## Experiment settings

### `--algorithm NAME`

Which algorithm to run. One of the 26 registered names:

```
bayesgap  bocs  bocs_hs  bootnn  casmopolitan  cocabo  combo  combo_slice
cts  cucb  dreamteam  glm_fpl  gp_nei  gp_onehot  gp_ts  gp_ucb  kg  linucb
neurallinear  ols  purexp  random  regevo  sa  smac  sts
```

`algorithms/README.md` explains what each one is and why it is in the suite.

### `--all`

Run every registered algorithm, one after another, instead of picking one.
Each gets its own output files. If you interrupt partway through, rerunning the
same command resumes — both within the algorithm that was running and across
the ones not yet started.

### `--rounds N` — default `300`

How many rounds each run lasts. One round = the algorithm picks a team, gets a
noisy score, and updates.

Raising this does not just cost time linearly: several algorithms refit a model
over all observations so far, so their cost grows faster than the round count.

### `--noise A B C ...` — default `0.0 0.2 0.4 0.6 0.8 1.0`

The noise levels to sweep. Every setting is played once at every level, so the
number of runs is `n_settings × len(noise)`.

Noise is **multiplicative**: the standard deviation of the reward is
`noise_level × reward`. A perfect team at `noise=1.0` is measured with standard
deviation 1.0; a completely wrong team is measured exactly. So the better the
team, the noisier its evaluation — the opposite of the usual assumption, and
the source of much of the interesting behaviour here.

`--noise 0.0` gives a noise-free run, which is useful for sanity checks.

### `--seed N` — default `42`

The master seed for per-run randomness. Each run's streams are derived from
`(seed, setting_id, noise_level)`, and deliberately **not** from the algorithm
name or from execution order. Three consequences:

- every algorithm faces the identical reward noise on a given problem, so
  comparisons between them are paired rather than carrying the noise's variance
- results do not depend on `--n-cores`
- results do not depend on whether the run was interrupted and resumed

Changing it gives a fresh set of noise draws over the same problems.

**This is not the same as `--settings-seed`.** The master seed changes the
noise; the settings seed changes the problems themselves.

---

## Perturbations — what makes this folder different

The hidden answer **moves mid-run**, and the algorithm is never told and never
reset. What is measured is how fast it notices and recovers.

### `--perturbation_rounds A B ...` — default `101 201`

The rounds *after* which the hidden answer is swapped. The new answer takes
effect on the following round. Must be ascending and inside the run.

Each setting's swaps are derived from its setting id alone, so every algorithm
— and every switch-limit condition — faces the identical swaps at the identical
rounds.

### `--p_perturb_dims N` — default `1`

How many roles have their correct answer changed at each swap. Each flip is
applied to the *previous* answer, so a later swap may revert an earlier role or
hit a different one.

Clamped per setting to that setting's dimension count, so a value larger than 3
still works on 3-role settings.

> **Dimensions vary, and that changes what a perturbation means.** Flipping one
> role is a 1/3 shock on a 3-role setting and a 1/9 shock on a 9-role one, so
> the size of the dip depends on the setting as much as on the algorithm.
> `n_bandits` is in every output row for exactly this reason — group by it.

### `--baseline-window LO HI` — default `80 100`

The inclusive round range used to measure each run's pre-perturbation
performance, which every recovery metric is relative to.

It **must end at or before the first perturbation**, or the baseline would be
measured partly after the answer moved. Passing a window that violates this is
rejected with an error rather than silently producing wrong numbers — so if you
change `--perturbation_rounds`, change this too.

---

## The switching limit

A cap on how many roles may change between one round and the next. It is
enforced by `environment.py` at a single point in the run loop and applies to
every algorithm identically — no algorithm knows it exists.

### `--switch-limit MODE` — default `none`

| mode | allowance at round `t` |
|---|---|
| `none` | unbounded — algorithms may change any number of roles |
| `flat` | `K`, the same every round |
| `parabolic` | `y(t) = K · (1 − ((t − T/2)/(T/2))²)` — zero at both ends, peaking at `K` at the midpoint |

`parabolic` is the budget the published DreamTeam algorithm used to apply to
itself, lifted out of the algorithm and renamed. The intent is that a team
settles in at the start, experiments most freely in the middle, and converges
by the end.

### `--max-changes K` — default `2.0`

The cap for `flat`; the mid-run peak for `parabolic`. Ignored when the mode is
`none`.

It is an **absolute number of dimensions, not a fraction**. Settings in this
benchmark have 3, 6 or 9 dimensions, so `K=2` is a much weaker constraint on a
3-dimension setting (2 of 3 may move) than on a 9-dimension one (2 of 9). That
is deliberate — it matches DreamTeam's original fixed budget — but say so when
reporting results that pool settings of different sizes.

### Things worth knowing

**Fractional budgets are rounded randomly.** `y(t)` is real-valued but the
number of roles that change is an integer, so `floor(y)` changes are always
allowed plus one more with probability `y − floor(y)`. That makes
`E[allowance] = y(t)` exactly, which is what keeps the hard cap comparable to
the soft, expectation-based budget DreamTeam used to enforce internally.

**When the cap bites, a random subset of the requested changes survives.** The
rest revert to the currently held arm. Random selection keeps enforcement
inside one class — no algorithm needs a scoring hook — and favours no method
over another.

**It limits speed, not reach.** A run starts about 6 of 9 roles away from the
optimum, so even `flat 2` could reach it in three rounds out of a hundred. What
is really restricted is the ability to probe a *distant* team in order to learn
from it.

**Do not use `parabolic` for recovery comparisons in this folder.** Its
allowance is a single arc sized to the whole run and knows nothing about the
perturbations. At the swaps themselves it is nearly equal (1.79 at round 101,
1.77 at round 201), but the recovery windows sit on opposite sides of its peak:
rounds 102–201 carry a mean allowance of 1.93 (193 changes in total), rounds
202–300 only 1.02 (101 changes), decaying to zero. Recovery from the second
swap therefore gets about half the churn budget of the first, which confounds
exactly the between-event comparison this experiment exists to make. Use `flat`,
whose allowance is identical before and after every swap.

**Model-based algorithms pay a warm-up cost.** Every surrogate method opens
with 10 uniformly random teams, which sit ~6 roles away. Under a cap those
rounds get truncated, so part of any measured drop is a crippled warm-up rather
than the acquisition rule. Worth stating when you report constrained results.

---

## Recording detail

### `--trace LEVEL` — default `full`

How much per-round detail to keep. See `OUTPUTS.md` for the columns.

| level | what is written |
|---|---|
| `off` | only the columns the results table needs. No trace file. |
| `basic` | the full per-round record: both teams, the blocked changes, the budget, the reward, the recommendation |
| `full` | also calls each algorithm's `diagnostics()` every round, adding `diag_*` columns |

**All three produce identical results.** The level changes what is kept, never
what is run — that is verified by `verify_setup.py`.

`off` does not mean "record nothing": the per-round score is always kept,
because the results table is a projection of these rows.

Cost at `full`: about 9 MB per algorithm for a 100-round sweep, and a few
percent of runtime.

There is deliberately no level that dumps a model's full internal state every
round. For the BOCS-family surrogates that is ~470 numbers per round — over
half a gigabyte per algorithm for a single field, almost none of which anyone
reads. Each algorithm summarises instead.

---

## Execution

### `--n-cores N` — default: every core but one

Worker processes. `0` or a negative number means the default. Values above the
machine's core count are clamped.

Each worker is pinned to a single BLAS thread. Without that, every worker
spawns its own thread pool, they fight each other, and *raising* `--n-cores`
makes the sweep slower.

Results are identical regardless of this setting.

### `--no-resume`

Throw away saved progress for this configuration and start it over. Without it,
rerunning the same command continues from where it stopped.

Resume works by keeping one small Parquet file per finished run in a `_parts/`
directory. A file either exists complete or does not exist at all, so a crash
mid-write costs you that one run and nothing else. This survives `kill -9` and
power loss, not just Ctrl-C.

### `--output-dir DIR` — default `Unconstrained Perturbation Results`

Where results go, relative to the folder. Mostly useful for keeping an
experimental run apart from your real results.

---

## run_full_sweep.sh — environment variables

The sweep script takes no flags; configure it with environment variables:

| variable | default | meaning |
|---|---|---|
| `SEED` | `42` | master seed, passed as `--seed` |
| `SWITCH_LIMIT` | `none` | `none` / `flat` / `parabolic` |
| `MAX_CHANGES` | `2` | the cap or the peak |
| `N_CORES` | `0` | worker processes; `0` = all cores but one |
| `N_SETTINGS` | `500` | how many problems |
| `ALGOS` | all 26 | a space-separated subset, e.g. `ALGOS="dreamteam bocs kg"` |

```bash
N_CORES=12 bash run_full_sweep.sh
SWITCH_LIMIT=flat MAX_CHANGES=2 N_CORES=12 bash run_full_sweep.sh
ALGOS="dreamteam bocs" bash run_full_sweep.sh
```

The list is ordered cheapest-first, so stopping partway still leaves a usable
spread of algorithms. Stopping is safe: everything resumes.

## Aggregation and plotting

Both read finished results from a directory and take no experiment flags.

| flag | applies to | meaning |
|---|---|---|
| `--results_dir DIR` | both | where the per-round result files are |
| `--output_dir DIR` | grapher | where to write the PNGs |
| `--total_rounds N` | aggregator | must match the run |
| `--n_bandits N` | grapher | a fallback only; per-row `n_bandits` is used when present |
| `--p_perturb_dims N` | both | **must match the run**, or the recovery ceiling is computed wrongly |

Results are keyed by `(algorithm, switch limit)`, so `bocs` and `bocs [flat2]`
appear as separate rows and separate plot series rather than one shadowing the
other.

---

## The settings file

The 500 problems every algorithm faces. Sampled once, written to JSON, and
loaded by every run — so "they all faced the same problems" is checkable
rather than assumed.

### `--n-settings N` — default `500`

How many settings to sample. A different count is a different benchmark and
gets its own file and its own output filenames. Lower values are for quick
tests, not for results.

### `--settings-seed N` — default `20240501`

The seed the settings are drawn from. **Changing this changes the benchmark
itself** — different team sizes, different arm counts, different hidden
answers. Only change it if you deliberately want a fresh problem set, and
report it if you do.

Both folders produce a byte-identical settings file at the same seed, so
stationary and non-stationary results are directly comparable.

### `--settings-path FILE`

Use a specific settings JSON instead of the default for the seed and count.
Useful for pointing two folders at one shared file, or for re-running an old
experiment against its original problems.

### What a setting contains

Each of the 500 is drawn independently:

1. `n_bandits` — uniform from **{3, 6, 9}** (the number of roles)
2. `arm_counts` — each role's candidate count, uniform from **2–5**
3. `bandit_types` — always `ongoing` for every dimension
4. `initial_bias` — the team the algorithm starts from
5. `optimal_arm` — the hidden answer

There is no early/late split. The published DreamTeam algorithm defines a
temporal schedule over dimension types, but an `ongoing` dimension has discount
`d = 1`, which makes the schedule the identity — so it is not applied, and no
registered algorithm reads `bandit_types`.

### Checking it

```bash
python3 sampling.py --verify
```

Prints the file's SHA-256 digest and re-derives the 500 settings from the
recorded seed to confirm the file on disk matches. Every run prints the same
digest at startup, and it is recorded in every manifest — so two result sets
carrying the same digest provably faced the same problems.

---

## What interacts with what

**`--seed` vs `--settings-seed`.** The first changes the noise; the second
changes the problems. Most of the time you want to vary only the first.

**Filenames encode the configuration.** `--algorithm`, `--switch-limit`,
`--max-changes`, `--rounds`, `--n-settings` and `--seed` all appear in the
output filename, so different configurations never overwrite one another, and
a resumed run finds its own progress. `--trace` and `--n-cores` do not appear,
because they do not change the results.

**Changing a setting mid-sweep.** If you resume with a different
`--n-settings` or `--noise`, the saved progress no longer matches: parts from
the old configuration are ignored and dropped rather than counted, so the run
completes correctly rather than hanging at "almost done".

**`--rounds` and `parabolic`.** The parabola is sized to the whole run, so
changing `--rounds` changes the shape of the budget, not just its length.

---

## Quick recipes

```bash
# one algorithm, defaults
python3 run_perturbation_experiment.py --algorithm bocs

# the whole suite on 12 cores, then aggregate and plot
N_CORES=12 bash run_full_sweep.sh

# the constrained condition (use flat, not parabolic, for recovery work)
N_CORES=12 SWITCH_LIMIT=flat MAX_CHANGES=2 bash run_full_sweep.sh

# a fast smoke test, with the baseline window moved to match
python3 run_perturbation_experiment.py --algorithm bocs --rounds 60 \
    --perturbation_rounds 21 41 --baseline-window 15 21 \
    --n-settings 10 --noise 0.4

# a harder shock: three roles change at each swap
python3 run_perturbation_experiment.py --all --p_perturb_dims 3
```
