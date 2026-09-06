# Non-stationary benchmark — all 27 algorithms

The full algorithm suite from `bandit_framework_all_algos`, run under the
**perturbation** experiment design taken from `bandit_framework-6/unconstrained_perturbation/`.

The difference from the stationary folder is one thing only: **the hidden
optimal team changes partway through each run**, and the algorithm is never
told and never reset. What this measures is not "can you find the answer" but
"how fast do you notice the answer moved, and how fast do you recover".

---

## The design

| | |
|---|---|
| Rounds per run | **300** (vs. 100 in the stationary folder) |
| Perturbations | after round **101** and again after round **201** |
| Each perturbation | flips the optimal arm in **1 of 9** dimensions |
| Algorithm state | **never reset** — it carries its beliefs across both swaps |
| Chaining | each flip is applied to the *previous* optimum, so the second swap may hit the same dimension (reverting it) or a different one |
| Sweep | 6 noise levels × 20 repetitions × 9 tests = **1080 runs** per algorithm |

The perturbation events are drawn from a separate seeded RNG
(`seed*9973 + 1`), so with the same `--seed` every algorithm faces the
identical problem, the identical tests **and** the identical perturbation
events. Comparisons across algorithms are like-for-like.

A run's performance drops at each swap by construction — one dimension's
best guess is now wrong — and the interesting question is the shape of the
curve afterwards.

## Running it

```bash
# one algorithm
python3 run_perturbation_experiment.py --algorithm kg --seed 42

# override the design
python3 run_perturbation_experiment.py --algorithm bocs \
    --rounds 300 --perturbation_rounds 101 201 --p_perturb_dims 1 --seed 42

# the whole suite, then aggregate + plot
bash run_full_sweep.sh
ALGOS="dreamteam bocs kg" bash run_full_sweep.sh    # a subset
SEED=123 bash run_full_sweep.sh
```

**Runtime warning.** This is 300 rounds against 1080 runs — three times the
stationary sweep — across 27 algorithms. The stationary sweep took roughly
three hours in total; this one is substantially more than three times that,
because several methods scale worse than linearly in the number of
observations. `bocs` and `smac` are the two to watch. Run a subset first, or
run it overnight. `run_full_sweep.sh` is ordered cheapest-first so a partial
run still leaves you a usable spread.

The folder also keeps a stationary runner (`run_experiment.py`) so both
regimes can be run from the same package with the same 27 algorithms.

## Output

Two CSVs per algorithm, into `Unconstrained Perturbation Results/`.

**Per-round** — `<algo>_perturbation_rounds300_dims9_tests9_perturbs2_t101_t201_<ts>.csv`

```
rowid, runID, noise_level, round, performance, phase, baseline
```

`phase` is `pre` (rounds 1–101), `post1` (102–201), `post2` (202–300).
`baseline` is that run's mean performance over rounds 80–100, i.e. the level
it had reached just before the first swap.

**Per-run summary** — `<algo>_perturbation_summary_..._<ts>.csv`, with four
columns per perturbation event:

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

producing `<algo>_perturbation.png` per algorithm plus a cross-algorithm
`recovery_summary.png`. These two scripts need `pandas` and `matplotlib`;
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

**The baseline window is fixed at rounds 80–100.** It is tied to the default
first perturbation at 101 and is *not* exposed on the CLI. If you move
`--perturbation_rounds`, set `baseline_window` on `PerturbationSettings` to
match, or the baseline will be measured from the wrong stretch of the run.

## What this regime should expose

The stationary benchmark rewards converging fast and staying put. This one
punishes exactly that, and the algorithms here fail in different ways:

- **Methods with a hard-to-revise posterior.** `dreamteam`'s `+1000` beta
  penalty makes a condemned arm practically unrecoverable — which is survivable
  when the answer never moves and potentially fatal when it does. This is the
  regime where the `dreamteam` vs `dreamteam_orig` comparison should matter
  most, since the published Beta update has no such trap.
- **Methods that shrink their own exploration on a schedule.** DreamTeam's
  sigmoid schedules and parabolic switching budget are keyed to
  `total_rounds`, so at round 202 they are near their end-of-run frozen state
  — right when the second swap demands fresh exploration. `sa`'s temperature
  is at its floor for the same reason. Both are *designed* to stop exploring
  late, which is a liability here rather than a virtue.
- **Methods that accumulate stale data.** Every surrogate here fits all
  observations equally, with no forgetting, discounting or change detection.
  The newest arms are no exception: `casmopolitan`'s trust region restarts on
  *stagnation*, not on *change*, and `sts` satisfices against a model that is
  itself stale after a swap.
  After a swap, the pre-swap data actively misleads the model, and the more
  data it has the more inertia it carries. None of the 27 algorithms has any
  non-stationarity mechanism — which is a finding worth stating plainly, and
  the obvious opening for a sliding-window or discounted variant as a
  follow-up arm.

## Contents

```
run_perturbation_experiment.py   the non-stationary driver (main entry point)
run_full_sweep.sh                all 27 algorithms, then aggregate + plot
aggregate_perturbation.py        recovery metrics across algorithms
graph_perturbation.py            per-algorithm curves + recovery_summary.png
run_experiment.py                stationary runner, same 27 algorithms
experiment.py                    problem/test generation, sweep loop, CSV
environment.py                   hidden optimal arms + noisy reward
algorithms/                      all 27 plug-ins (identical to the stationary folder)
```

`environment.py` and the entire `algorithms/` folder are byte-identical to
`bandit_framework_all_algos`. Only the driver differs — the non-stationarity
lives in `run_single_with_perturbations`, which reassigns
`environment.optimal_arm` at the scheduled rounds. Nothing was changed inside
any algorithm, so results here and in the stationary folder are directly
comparable.
