# Multi-Dimensional Bandit Experiment Framework

Modularized version of `new_global_Dynamic_Parameters.py`. The recommendation
algorithm is a swappable plug-in, the constraint on how fast a team may change
is a swappable property of the environment (see **Switching limits**), and the
test cases come from a shared 500-setting sampling protocol.

**New to this folder? Read `SETUP.md`** — dependencies, how to run, how to
stop and resume, and what is guaranteed. `python3 verify_setup.py` checks all
of it in under a minute.

Then, as reference: **`INPUTS.md`** for every flag and setting, and
**`OUTPUTS.md`** for every output file and what each column means.

## The sampling protocol

500 settings are sampled once into `bandit_settings_n500_seed20240501.json`
and loaded by every algorithm. Each setting is drawn independently:

1. `n_bandits` uniform from **{3, 6, 9}**
2. each bandit's arm count uniform from **2–5**
3. every bandit is **`ongoing`**
4. plus one `initial_bias` and one `optimal_arm`

One run = one (setting, noise level) pair, so the sweep is **500 × 6 = 3000
runs per algorithm**. No repetitions — 500 independent settings give more
variety than repeating a smaller set.

Because the settings live in a file rather than being regenerated from a seed
in each process, "every algorithm faced the same problems" is checkable rather
than assumed: the file carries a SHA-256 digest of its contents, printed at
the start of every run.

This replaces the previous design (one random problem, 9 tests, 20 repetitions),
which also had a trap — without `--seed`, each algorithm silently got a
*different* problem.

Every dimension is `ongoing` — there is no early/late split. The published
DreamTeam algorithm defines a temporal schedule over dimension types, but an
`ongoing` dimension gets a discount of d = 1, which makes its renormalization
the identity, so the schedule is simply not applied. No registered algorithm
reads `bandit_types`.

## Parallel execution and resuming

```bash
python3 run_experiment.py --algorithm bocs --n-cores 12
python3 run_experiment.py --all --n-cores 12
```

`--n-cores` sets the worker count (default: every core but one). A tqdm
progress bar tracks runs.

Press **Ctrl-C at any time**; rerun the same command to resume. Progress is
checkpointed per run and survives a hard kill or a power cut — a run that was
half-written when the process died is detected and redone. `--no-resume`
starts over.

Because each run's RNG is seeded from `(master seed, setting, noise)` and never
from the algorithm or the scheduling order, **1 core, 16 cores, and an
interrupted-then-resumed run all produce byte-identical output**. The same
seeding gives every algorithm the identical reward noise on a given problem, so
comparisons between algorithms are paired rather than carrying the noise's
variance.

## Layout

```
bandit_framework/
├── run_experiment.py        # entry point (interactive or CLI flags)
├── SETUP.md                 # how to run it (start here)
├── INPUTS.md                # every flag and setting, explained
├── OUTPUTS.md               # every output file and column, explained
├── verify_setup.py          # self-check: deps, settings, determinism, resume
├── sampling.py              # the 500-setting protocol + digest
├── tracing.py               # per-round Parquet trace: schema, writer, reader
├── parallel.py              # worker pool, tqdm, checkpoint/resume
├── experiment.py            # harness: sweep loop, CSV
├── environment.py           # hidden optimal arms, noisy reward, switching limit
└── algorithms/
    ├── __init__.py               # registry (name -> class)
    ├── base.py                   # RecommendationAlgorithm interface + ProblemConfig
    ├── dreamteam.py              # DreamTeam exactly as published (CHI 2018)
    ├── random_baseline.py        # minimal example plug-in / performance floor
    ├── simulated_annealing.py    # SA baseline (BOCS & COMBO papers)
    ├── local_search.py           # oblivious local search (BOCS paper)
    ├── regularized_evolution.py  # aging evolution (COMBO paper)
    ├── bocs.py                   # sparse Bayesian linear model + Thompson sampling
    ├── combo.py                  # GP w/ graph-Cartesian-product diffusion kernel + EI
    ├── smac.py                   # random-forest surrogate + EI
    ├── gp_onehot.py              # textbook GP-EI on one-hot ("naive GP" ablation)
    ├── neural_linear.py          # Bayesian linear posterior on an MLP's last layer
    ├── bootstrapped_nn.py        # bootstrap ensemble of MLPs
    ├── lin_ucb.py                # bocs surrogate w/ UCB instead of Thompson sampling
    ├── knowledge_gradient.py     # bocs surrogate w/ the Knowledge Gradient
    ├── pure_exploration.py       # bocs surrogate w/ no acquisition at all
    ├── satisficing_ts.py         # bocs surrogate w/ satisficing Thompson sampling
    ├── gp_acquisitions.py        # noisy-EI / IGP-UCB / GP-TS on the one-hot GP
    ├── casmopolitan.py           # trust-region categorical BO
    ├── glm_fpl.py                # logistic GLM explored by perturbing rewards
    ├── bayes_gap.py              # fixed-budget best-arm identification
    ├── cucb.py                   # combinatorial UCB
    ├── combinatorial_ts.py       # combinatorial Thompson sampling
```

## Available algorithms

Select with `--algorithm <name>`; the name is also the CSV filename prefix.

| name | family | what it does |
| --- | --- | --- |
| `random` | reference | uniform random every round; performance floor |
| `dreamteam` | reference | DreamTeam as published (Zhou, Valentine & Bernstein, CHI 2018): Beta(1,1) posteriors, Thompson sampling, the standard `alpha += r, beta += 1-r` update, probabilistic selection |
| `sa` | model-free search | simulated annealing, geometric cooling |
| `ols` | model-free search | oblivious local search: sweep neighbourhood, move, restart |
| `regevo` | model-free search | regularized (aging) evolution over a population of teams |
| `bocs` | surrogate BO | sparse Bayesian linear model with pairwise terms, Thompson sampling |
| `bocs_hs` | surrogate BO | the same, with the paper's real horseshoe prior (Gibbs-sampled) |
| `combo` | surrogate BO | GP with a diffusion kernel on the graph Cartesian product, EI |
| `combo_slice` | surrogate BO | the same, with the paper's real slice-sampled hyperparameters |
| `smac` | surrogate BO | random-forest surrogate, EI, interleaved random configs |
| `gp_onehot` | surrogate BO | textbook GP-EI on a one-hot encoding (the "naive GP" ablation) |
| `cocabo` | surrogate BO | per-dimension EXP3 agents + GP with a categorical overlap kernel |
| `neurallinear` | neural | Bayesian linear posterior on an MLP's last hidden layer |
| `bootnn` | neural | bootstrap ensemble of MLPs, greedy w.r.t. one sampled member |
| `linucb` | acquisition | the `bocs` surrogate with a UCB bonus instead of Thompson sampling |
| `kg` | acquisition | the `bocs` surrogate with the Knowledge Gradient (decision-theoretic) |
| `purexp` | acquisition | the `bocs` surrogate with *no* acquisition -- uniform allocation |
| `bayesgap` | best-arm ID | gap-based allocation aimed at the final recommendation |
| `cucb` | combinatorial bandit | optimistic per-base-arm estimates, exact separable oracle |
| `cts` | combinatorial bandit | Thompson sampling per base arm, exact separable oracle |
| `casmopolitan` | surrogate BO | trust-region BO: a Hamming ball that grows, shrinks and restarts |
| `glm_fpl` | GLM | logistic-link GLM explored by perturbing the observed rewards |
| `sts` | acquisition | the `bocs` surrogate with satisficing Thompson sampling |
| `gp_nei` | acquisition | the `gp_onehot` surrogate with Noisy Expected Improvement |
| `gp_ucb` | acquisition | the `gp_onehot` surrogate with IGP-UCB |
| `gp_ts` | acquisition | the `gp_onehot` surrogate with GP Thompson sampling |

### The global constraint has moved out of DreamTeam

DreamTeam used to cap its own total off-current probability mass with the
parabolic budget above, which made it the only switch-constrained arm in a
suite where every competitor could jump anywhere each round — a fairness
problem the old README listed under "not held constant".

That internal budget has been removed. The curve now lives in `environment.py`
as the `parabolic` mode and applies to everyone or to no one. So with
`--switch-limit none` DreamTeam moves more freely than the published algorithm
did, and `--switch-limit parabolic --max-changes 2` is the closer analogue of
the original — though not an identical one, since the old budget was a soft cap
on the *expected* number of changes while the environment enforces a hard cap
whose expectation matches it.

**Note on what `dreamteam` now is.** This package used to carry two DreamTeam
arms: the published algorithm and a modified descendant of it that replaced the
Beta update with a threshold-and-penalty rule (`alpha += r` above a 0.1
threshold, `beta += 1000` below it). The modified variant has been removed
entirely. `dreamteam` is the published algorithm.

### Two comparisons the registry is built around

**Acquisition rules on one fixed surrogate -- twice over.** `bocs`, `linucb`, `kg`, `purexp` and
`sts` all use the identical second-order Bayesian linear model with identical priors and the
identical local-search optimizer, differing *only* in how they pick the next team: Thompson
sampling, upper confidence bound, knowledge gradient, nothing at all, and satisficing Thompson
sampling. Separately, `gp_onehot`, `gp_nei`, `gp_ucb` and `gp_ts` do the same on a Gaussian
process -- EI, noisy EI, IGP-UCB and GP Thompson sampling on one shared kernel and fit. Running
the acquisition comparison on two different surrogates separates "this rule is better" from
"this rule happens to suit a linear model".

**Faithful-vs-simplified surrogates.** `bocs_hs` and `combo_slice` implement the inference their
papers actually specify (horseshoe prior via the Makalic-Schmidt Gibbs sampler; hyperparameter
marginalization via Murray-Adams slice sampling) where `bocs` and `combo` substitute cheaper
point estimates. Running each pair turns a documented fairness caveat into a measured quantity
instead of a disclaimer.

`dreamteam.py` also exports a standalone `DreamTeam` class implementing the
paper's fixed five-dimension system (405 structures) with its own
`initialize()` / `get_current_structure()` / `step(reward)` interface, for
reproducing the published behaviour directly outside the harness. That class
keeps the paper's early/late temporal schedule, because its five named
dimensions genuinely have different types; it is not registered with the
harness and never runs during a sweep.

No registered algorithm consumes `ProblemConfig.bandit_types`: every dimension
in this benchmark is `ongoing`, so there is no side information for any arm to
exploit. Every algorithm treats all dimensions identically, which matches how
the baselines in the source papers work.

Each plug-in's module docstring cites the paper it comes from and lists what
was simplified relative to that paper.

## Running

```bash
python3 run_experiment.py --algorithm bocs
python3 run_experiment.py --algorithm bocs --n-cores 12
python3 run_experiment.py --all --n-cores 12
python3 run_experiment.py --algorithm bocs --switch-limit flat --max-changes 2
python3 run_experiment.py --algorithm bocs --noise 0.0 0.5 1.0
```

`--bandits`, `--tests` and `--runs` are gone: dimensions come from the settings
file and there are no repetitions.

Output is **Parquet**, in `Global results/` — three files per algorithm:

```
<algo>_limit-<mode>_rounds100_settings500_seed42_trace.parquet
<algo>_limit-<mode>_rounds100_settings500_seed42.parquet
<algo>_limit-<mode>_rounds100_settings500_seed42_manifest.json
```

The **trace** is the raw record of the simulation: one row per round, holding
the team the algorithm asked for, the team it was allowed to play, exactly
which changes the switching limit blocked, the budget in force, the reward, its
recommendation, and that algorithm's own internal state as `diag_*` columns.
It is the source of truth.

The **results** file is a six-column projection of the trace —
`run_id, setting_id, n_bandits, noise_level, round, performance` — and exists
only because that is what you plot.

The **manifest** records seeds, the settings digest, the switch-limit
configuration, the algorithm's constants, core count and wall time, so a file
found later can be attributed without guesswork.

`setting_id` and `n_bandits` are per row because **dimensions vary between
settings**. `performance` is a fraction, so it compares across settings — but
group by `n_bandits` before reading anything absolute into it, since 2 of 3 is
a different achievement from 6 of 9.

Filenames carry the switching-limit mode, so conditions never overwrite one
another, and no timestamp, so a resumed run finds its own checkpoint. A
`_parts/` directory is the checkpoint while a sweep runs; it is removed on
completion.

**Why Parquet.** Measured on 300k realistic rows: 5 MB against 24 MB for the
same data as CSV, and reading three columns is ~29× faster because it only
touches those columns' bytes. Full suite at trace level `full`: ~9 MB per
algorithm, ~0.25 GB for all 26.

`--trace` sets the detail level — `full` (default), `basic` (no `diag_*`
columns), `off` (results columns only, no trace file). All three produce
identical results; the level changes what is kept, never what is run.

`tracing.read()` loads a file as a DataFrame; see SETUP.md. Prefer the integer
`noise_pct` over the float `noise_level` when filtering — both hold the same
information, and the integer one cannot be tripped up by float equality.

## Switching limits

How many roles may change from one round to the next. The limit is enforced by
`environment.py`, in one place in the harness, and applies to **every**
algorithm identically — no algorithm knows it exists.

| `--switch-limit` | behaviour |
| --- | --- |
| `none` *(default)* | no cap; an algorithm may change any number of roles each round |
| `flat` | at most `--max-changes` roles change in any round, start to finish |
| `parabolic` | the allowance follows `y(t) = K · (1 − ((t − T/2)/(T/2))²)` — zero at both ends, peaking at `K` mid-run |

`--max-changes` (default `2`) is the cap for `flat` and the mid-run peak for
`parabolic`. It is an **absolute** number of dimensions, not a fraction, so at
K=2 it constrains a 9-dimension setting far more than a 3-dimension one. That
is deliberate — it matches DreamTeam's original fixed budget — but it is worth
stating when reporting results across settings of different sizes.

**Why `parabolic` has that shape.** It is the budget the published DreamTeam
algorithm used to apply to itself, lifted out of the algorithm and renamed so
it is no longer tied to one arm. The intent is that a team settles in at the
start, experiments most freely in the middle, and converges by the end.

**Randomized rounding.** `y(t)` is real-valued but the number of roles that
actually change is an integer. Truncating would systematically under-spend the
budget, so `floor(y)` changes are always allowed plus one more with probability
`y − floor(y)`. That makes `E[allowance(t)] = y(t)` exactly (verified to within
0.004 over 40k draws), so the hard per-round cap stays directly comparable to
the soft, expectation-based budget DreamTeam used to enforce internally.

**Which changes survive.** When an algorithm requests more changes than the
allowance permits, a uniformly random subset of the requested changes is kept
and the rest revert to the currently held arm. Random selection keeps
enforcement entirely inside `environment.py` — no algorithm needs a scoring
hook, and the truncation rule favours nobody.

**How restrictive is it, really.** The cap limits *velocity*, not *reach*. At
the default settings a run starts about 6 of 9 roles away from the optimum, so
even `flat 2` could reach it in three rounds out of a hundred. Summed over a
100-round run the parabolic budget allows ~133 changes and `flat 2` allows 200.
What the cap actually restricts is an algorithm's ability to probe a *distant*
team in order to learn from it — which is why the model-based arms feel it most
(their `N_INIT = 10` random warm-up teams sit ~6 roles away and get truncated),
while `sa` and `ols`, which move in single steps, are barely affected.

## Adding a new recommendation algorithm

1. Create `algorithms/my_algo.py`:

```python
from .base import RecommendationAlgorithm

class MyAlgorithm(RecommendationAlgorithm):
    name = "myalgo"

    def choose(self, round_num):
        # return one arm index per bandit, e.g. using self.config.arm_counts,
        # self.config.bandit_types, self.initial_bias, self.total_rounds
        ...

    def update(self, arms_chosen, reward):
        # incorporate the observed reward (float in [0, 1])
        ...

    def predict_best(self):
        # return your current best-guess arm per bandit (used for scoring)
        ...

    # Optional. Only needed if your algorithm tracks "the team I am currently
    # holding", or if its update is only valid for actions it drew itself.
    def notify_played(self, arms_played, arms_requested=None):
        ...
```

2. Register it in `algorithms/__init__.py`:

```python
from .my_algo import MyAlgorithm
ALGORITHMS[MyAlgorithm.name] = MyAlgorithm
```

3. Run it: `python run_experiment.py --bandits 9 --rounds 100 --algorithm myalgo`

The contract: the algorithm never sees the optimal arms or the noise level —
only the rewards it receives via `update()`. Scoring compares
`predict_best()` against the hidden optimum each round.

**Under a switching limit**, the team returned by `choose()` is not necessarily
the team that gets played — some requested changes may be reverted. This is
handled for you: `update()` is always called with the team that was *actually*
played, and every algorithm in this package reads its `arms_chosen` argument
rather than whatever it stashed during `choose()`, so surrogates, posteriors and
design matrices all train on reality automatically. `notify_played()` is called
just before `update()` with both teams; the default implementation keeps the
conventional `last_choice` attribute in sync. Override it only if you track an
incumbent of your own (as `dreamteam` does) or if your update
is invalid for actions you did not draw yourself (as `cocabo`'s importance-
weighted EXP3 step is).

## Design notes

- **Environment vs. algorithm separation**: `TeamRewardEnvironment` owns the
  optimal-arm vector and reward noise, so algorithms physically cannot peek
  at the answer. Alternative reward structures (per-dimension credit,
  relative thresholds, etc.) can be added there without touching algorithms.
- **The switching limit is a property of the world, not of an algorithm.**
  `SwitchLimiter` lives in `environment.py` and is enforced at a single point
  in `experiment.py::run_single`. That is what makes it a controlled
  experimental factor: every one of the 26 arms is subject to the identical
  rule, enforced identically, and adding a new algorithm gets it for free.
- **Fresh state per run**: the harness constructs a new algorithm instance
  per run, which replaces the original `clone_bandits()` deep-copy dance.
- **Known quirks preserved in `dreamteam.py`** (documented as class
  constants so they're easy to experiment with): the fixed `0.1` reward
  threshold and the `+1000` beta penalty. The third quirk — the global budget
  peak `m=2` — is no longer a DreamTeam constant; it is now
  `--max-changes` on the environment's `parabolic` mode.
