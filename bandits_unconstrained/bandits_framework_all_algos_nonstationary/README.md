# Non-stationary benchmark — all 26 algorithms

The full algorithm suite from `bandit_framework_all_algos`, run under the
**perturbation** experiment design taken from `bandit_framework-6/unconstrained_perturbation/`.

**New to this folder? Read `SETUP.md`** — dependencies, how to run, how to
stop and resume, and what is guaranteed. `python3 verify_setup.py` checks all
of it in under a minute.

Then, as reference: **`REQUIREMENTS.md`** for dependencies and hardware,
**`INPUTS.md`** for every flag and setting, and **`OUTPUTS.md`** for every
output file and what each column means.

The difference from the stationary folder is one thing only: **the hidden
optimal team changes partway through each run**, and the algorithm is never
told and never reset. What this measures is not "can you find the answer" but
"how fast do you notice the answer moved, and how fast do you recover".

---

## The design

| | |
|---|---|
| Test cases | **500 sampled settings** from a shared file — see below |
| Rounds per run | **300** (vs. 100 in the stationary folder) |
| Perturbations | after round **101** and again after round **201** |
| Each perturbation | flips the optimal arm in **1** dimension of that setting |
| Algorithm state | **never reset** — it carries its beliefs across both swaps |
| Chaining | each flip is applied to the *previous* optimum, so the second swap may hit the same dimension (reverting it) or a different one |
| Sweep | 500 settings × 6 noise levels = **3000 runs** per algorithm |

## The sampling protocol

500 settings are sampled once into `bandit_settings_n500_seed20240501.json`
and loaded by every algorithm. Each is drawn independently:

1. `n_bandits` uniform from **{3, 6, 9}**
2. each bandit's arm count uniform from **2–5**
3. every bandit is **`ongoing`**
4. plus one `initial_bias` and one `optimal_arm`

One run = one (setting, noise level) pair; there are no repetitions. The file
carries a SHA-256 digest of its contents, printed at the start of every run, so
"every algorithm faced the same problems" is checkable rather than assumed. The
file is byte-identical to the stationary folder's at the same seed, so results
from the two folders are directly comparable.

Each setting's perturbation chain is derived from its setting id alone, so
every algorithm — and every switching-limit condition — faces the identical
swaps at the identical rounds.

**Dimensions now vary, and that changes what a perturbation means.** Flipping
one dimension is a 1/3 shock on a 3-dimension setting and a 1/9 shock on a
9-dimension one, so the size of the dip depends on the setting as much as on
the algorithm. `n_bandits` is written into every row for exactly this reason:
group by it before comparing dips or recovery times. The recovery metrics use
each run's own dimension count for their ceiling.

## Parallel execution and resuming

`--n-cores` sets the worker count (default: every core but one), with a tqdm
progress bar. Press **Ctrl-C at any time**; rerun the same command to resume.
Progress is checkpointed per run and survives a hard kill or a power cut.

Each run's RNG is seeded from `(master seed, setting, noise)` — never from the
algorithm or the scheduling order — so **1 core, 16 cores, and an
interrupted-then-resumed run all produce byte-identical output**, and every
algorithm faces the identical reward noise on a given problem.

A run's performance drops at each swap by construction — one dimension's
best guess is now wrong — and the interesting question is the shape of the
curve afterwards.

## Running it

```bash
# one algorithm
python3 run_perturbation_experiment.py --algorithm kg --n-cores 12

# every algorithm, in turn
python3 run_perturbation_experiment.py --all --n-cores 12

# override the design
python3 run_perturbation_experiment.py --algorithm bocs \
    --rounds 300 --perturbation_rounds 101 201 --p_perturb_dims 1

# the whole suite, then aggregate + plot
N_CORES=12 bash run_full_sweep.sh
ALGOS="dreamteam bocs kg" bash run_full_sweep.sh    # a subset
SEED=123 bash run_full_sweep.sh

# under a switching limit (see below)
python3 run_perturbation_experiment.py --algorithm kg --switch-limit flat --max-changes 2
SWITCH_LIMIT=flat MAX_CHANGES=2 N_CORES=12 bash run_full_sweep.sh
```

**Runtime warning.** 3000 runs of 300 rounds each, across 26 algorithms. Even
in parallel this is a long job — `bocs`, `sts` and `smac` dominate it. Run one
algorithm first to calibrate, then launch the sweep with as many cores as you
can spare. `run_full_sweep.sh` is ordered cheapest-first so a partial run still
leaves a usable spread, and stopping it is safe: everything resumes.

The folder also keeps a stationary runner (`run_experiment.py`) so both
regimes can be run from the same package with the same 26 algorithms.

## Switching limits

How many roles may change from one round to the next. Enforced by
`environment.py`, at one point in each run loop, and applied to **every**
algorithm identically — no algorithm knows it exists. Identical to the
stationary folder's implementation; `algorithms/README.md` §1.9 has the full
account.

| `--switch-limit` | behaviour |
| --- | --- |
| `none` *(default)* | no cap; an algorithm may change any number of roles each round |
| `flat` | at most `--max-changes` roles change in any round, start to finish |
| `parabolic` | `y(t) = K · (1 − ((t − T/2)/(T/2))²)` — zero at both ends, peaking at `K` mid-run |

`--max-changes` (default `2`) is the cap for `flat` and the mid-run peak for
`parabolic`. The `parabolic` shape is the budget the published DreamTeam
algorithm used to apply to itself, lifted out of the algorithm and renamed;
both DreamTeam arms now run unconstrained unless a limit is selected here.

Because `y(t)` is real-valued and the count of changed roles is an integer, the
allowance is rounded randomly — `floor(y)` changes always, plus one more with
probability `y − floor(y)` — so `E[allowance(t)] = y(t)` exactly. When more
changes are requested than allowed, a uniformly random subset survives.

### `parabolic` and perturbations do not mix well — read this

The parabolic allowance is a single settle-explore-converge arc sized to the
**whole run**, and it knows nothing about the perturbation schedule.

At the moment of each swap the allowance is almost the same — `y(101) = 1.79`,
`y(201) = 1.77` — so the problem is not the swap itself. It is the **recovery
window that follows**, because the two sit on opposite sides of the peak:

| recovery window | mean `y(t)` | total allowance | min |
|---|---|---|---|
| after swap 1 (rounds 102–201) | 1.93 | 193 changes | 1.77 |
| after swap 2 (rounds 202–300) | 1.02 | 101 changes | 0.00 |

An algorithm recovering from the second perturbation gets roughly **half the
total churn budget** of one recovering from the first, and the tail of that
window allows essentially no movement at all. A late perturbation is therefore
penalised relative to an early one purely by where it falls on the curve, which
confounds any comparison of recovery times *between* the two events — the
headline metric of this experiment.

**Use `flat` for recovery work.** Its allowance is identical before and after
every swap, so differences between the events are attributable to the
algorithms rather than to the budget. Reach for `parabolic` only when the
interaction between a tapering budget and a moving optimum is itself the
object of study.

A perturbation-aware schedule — one that resets the arc at each swap — is the
obvious follow-up, and would be a small addition to `SwitchLimiter`. It is
deliberately not implemented: it is not a mode either framework's source
material defines, and inventing one would need justifying rather than
assuming.

### Results are tagged by condition

Filenames carry a `limit-<mode>` segment, e.g.
`bocs_limit-flat2_perturbation_rounds300_dims9_tests9_perturbs2_t101_t201_<ts>.csv`,
so conditions never overwrite one another. `aggregate_perturbation.py` and
`graph_perturbation.py` key results by **(algorithm, switching limit)** and
report them as separate rows and series — `bocs` and `bocs [flat2]`. Files
written before the switching limit existed still parse and are treated as
`none`.

## Output

Two CSVs per algorithm, into `Unconstrained Perturbation Results/`.

Output is **Parquet**. Writing `<stem>` for
`<algo>_limit-<mode>_perturbation_rounds300_settings500_perturbs2_t101_t201_seed42`:

```
<stem>_trace.parquet     one row per round — the raw record of the simulation
<stem>.parquet           results projection
<stem>_summary.parquet   per-run recovery metrics
<stem>_manifest.json     seeds, settings digest, switch limit, versions
```

The **trace** holds, for every round: the team the algorithm asked for, the
team it was allowed to play, which changes the switching limit blocked, the
budget in force, the reward, its recommendation, and that algorithm's internal
state as `diag_*` columns. Plus three columns specific to this folder —
`phase`, `baseline`, and `optimum_version` (0 before the first swap, 1 after
it, 2 after the second), so you can tell which answer was in force at any
round.

The **results** file projects out
`run_id, setting_id, n_bandits, noise_level, round, performance, phase, baseline`.

`--trace` sets the detail level: `full` (default), `basic`, `off`. All three
produce identical results. Trace size at `full`: ~27 MB per algorithm,
~0.69 GB for all 26.

Filenames carry no timestamp, so a resumed run finds its own checkpoint; the
`_parts/` directory beside them is that checkpoint, removed on completion.

`phase` is `pre` (rounds 1–101), `post1` (102–201), `post2` (202–300).
`baseline` is that run's mean performance over rounds 80–100, i.e. the level
it had reached just before the first swap.

The **summary** file has four columns per perturbation event:

| column | meaning |
|---|---|
| `recovery_threshold_rounds_i` | rounds until performance reaches `baseline × (9-i)/9` |
| `time_to_baseline_rounds_i` | rounds until performance reaches the full pre-perturbation `baseline` |
| `max_dip_i` | largest `1 - performance` in the 100 rounds after the swap |
| `recovered_within_300_i` | whether the threshold was ever reached |

Then:

```bash
python3 aggregate_perturbation.py --results_dir "Unconstrained Perturbation Results" --p_perturb_dims 1
python3 graph_perturbation.py --results_dir "Unconstrained Perturbation Results" \
    --output_dir Graphs --n_bandits 9 --p_perturb_dims 1
```

producing `perturbation_<algo>[_limit-<mode>].png` per condition plus a
cross-algorithm `recovery_summary.png`. These two scripts need `pandas` and `matplotlib`;
everything else in the folder is numpy-only.

## Two things to watch when interpreting results

**`recovery_threshold_rounds` degenerates for good algorithms.** The threshold
after one flip is `baseline × 8/9`, and one flipped dimension costs exactly
`1/9` of performance. So an algorithm sitting at `baseline = 1.0` before the
swap lands on `8/9` immediately after it and is scored as having "recovered"
in **1 round** — while an algorithm that was at 0.7 faces a genuine bar. The
better the algorithm, the less this metric says. `time_to_baseline_rounds` and
`max_dip` are the ones that carry real information; the threshold metric is
inherited from the framework-6 design and is kept for comparability, not
because it is the right summary. This is worth stating explicitly in any
writeup that quotes it.

**The baseline window defaults to rounds 80–100.** It is tied to the default
first perturbation at 101, and it is now on the CLI as `--baseline-window LO HI`.
Moving `--perturbation_rounds` without moving it is rejected with an error
rather than silently measuring the baseline from the wrong stretch of the run.

## What this regime should expose

The stationary benchmark rewards converging fast and staying put. This one
punishes exactly that, and the algorithms here fail in different ways:

- **Methods with a hard-to-revise posterior.** An arm whose posterior cannot
  be walked back is survivable when the answer never moves and potentially
  fatal when it does. DreamTeam's published Beta update has no such trap — each
  observation moves the posterior by exactly one unit of evidence — but methods
  that accumulate evidence without discounting still harden over 300 rounds.
  (This package previously carried a modified DreamTeam whose `+1000` beta
  penalty made a condemned arm practically unrecoverable; that variant has been
  removed, and this regime is where it would have suffered most.)
- **Methods that shrink their own exploration on a schedule.** `sa`'s
  temperature is at its floor by round 202 — right when the second swap demands
  fresh exploration. It is *designed* to stop exploring late, which is a
  liability here rather than a virtue.

  DreamTeam used to compound this with two schedules of its own: sigmoid
  early/late dimension types, and a global switching budget. Neither is active
  any more — every dimension here is `ongoing`, and the budget moved into the
  environment as the `parabolic` switching limit. Note that selecting
  `--switch-limit parabolic` reintroduces exactly this failure mode, and does
  so for **all 26 algorithms at once**; that is the confound described under
  "Switching limits" above, and the reason `flat` is the right default for
  recovery work.
- **Methods that accumulate stale data.** Every surrogate here fits all
  observations equally, with no forgetting, discounting or change detection.
  The newest arms are no exception: `casmopolitan`'s trust region restarts on
  *stagnation*, not on *change*, and `sts` satisfices against a model that is
  itself stale after a swap.
  After a swap, the pre-swap data actively misleads the model, and the more
  data it has the more inertia it carries. None of the 26 algorithms has any
  non-stationarity mechanism — which is a finding worth stating plainly, and
  the obvious opening for a sliding-window or discounted variant as a
  follow-up arm.
- **Methods that cannot move fast enough once they do notice.** New with the
  switching limit: under a cap, recognising that the optimum moved and being
  *able* to chase it come apart. An algorithm that detects the change
  immediately still needs enough allowance to walk there. Running the
  perturbation sweep at `--switch-limit flat` with a few values of
  `--max-changes` turns recovery time into a function of the churn budget,
  which is a result neither the stationary folder nor the unconstrained
  perturbation run can produce.

## Contents

```
run_perturbation_experiment.py   the non-stationary driver (main entry point)
run_full_sweep.sh                all 26 algorithms, then aggregate + plot
aggregate_perturbation.py        recovery metrics across algorithms
graph_perturbation.py            per-algorithm curves + recovery_summary.png
run_experiment.py                stationary runner, same 26 algorithms
experiment.py                    problem/test generation, sweep loop, CSV
environment.py                   hidden optimal arms, noisy reward, switching limit
sampling.py                      the 500-setting protocol + digest
make_9dim_settings.py            optional: a settings file with a fixed team size
tracing.py                       per-round Parquet trace: schema, writer, reader
parallel.py                      worker pool, tqdm, checkpoint/resume
verify_setup.py                  self-check: deps, settings, determinism, resume
SETUP.md                         how to run it (start here)
REQUIREMENTS.md                  dependencies, versions, hardware
INPUTS.md                        every flag and setting, explained
OUTPUTS.md                       every output file and column, explained
algorithms/                      all 26 plug-ins (identical to the stationary folder)
```

`environment.py` and the entire `algorithms/` folder are byte-identical to
`bandit_framework_all_algos`. Only the driver differs — the non-stationarity
lives in `run_single_with_perturbations`, which reassigns
`environment.optimal_arm` at the scheduled rounds. Nothing was changed inside
any algorithm, so results here and in the stationary folder are directly
comparable.
