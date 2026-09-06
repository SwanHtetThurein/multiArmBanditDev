# Multi-Dimensional Bandit Experiment Framework

Modularized version of `new_global_Dynamic_Parameters.py`. The testing
structure (random problem generation, 9 tests x 20 runs x 6 noise levels,
per-round performance CSV) is unchanged; the recommendation algorithm is now
a swappable plug-in.

## Layout

```
bandit_framework/
├── run_experiment.py        # entry point (interactive or CLI flags)
├── experiment.py            # harness: problem/test generation, sweep loop, CSV
├── environment.py           # hidden optimal arms + noisy reward function
└── algorithms/
    ├── __init__.py               # registry (name -> class)
    ├── base.py                   # RecommendationAlgorithm interface + ProblemConfig
    ├── dreamteam.py              # the original algorithm, logic preserved exactly
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
    └── dreamteam_original.py     # DreamTeam exactly as published (CHI 2018)
```

## Available algorithms

Select with `--algorithm <name>`; the name is also the CSV filename prefix.

| name | family | what it does |
| --- | --- | --- |
| `random` | reference | uniform random every round; performance floor |
| `dreamteam` | reference | this project's modified algorithm: per-bandit Thompson sampling + type-scheduled stickiness + global switch budget, with the 0.1-threshold / +1000-beta update |
| `dreamteam_orig` | reference | DreamTeam exactly as published (Zhou, Valentine & Bernstein, CHI 2018): standard `alpha += r, beta += 1-r` update and equal-share global constraint |
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

`dreamteam_original.py` also exports a standalone `DreamTeam` class implementing the
paper's fixed five-dimension system (405 structures) with its own
`initialize()` / `get_current_structure()` / `step(reward)` interface, for
reproducing the published behaviour directly outside the harness. Its module
docstring enumerates every difference from `dreamteam.py`.

Only `dreamteam` and `dreamteam_orig` consume `ProblemConfig.bandit_types` (the early/late/ongoing
split). Every other algorithm treats all dimensions identically, which matches
how the baselines in the source papers work -- worth stating explicitly when
comparing, since `dreamteam` is using side information the others ignore.

Each plug-in's module docstring cites the paper it comes from and lists what
was simplified relative to that paper.

## Running

```bash
# Interactive prompts (like the original script)
python run_experiment.py

# Non-interactive
python run_experiment.py --bandits 9 --rounds 100
python run_experiment.py --bandits 12 --rounds 100 --seed 42
python run_experiment.py --bandits 9 --rounds 100 --algorithm random
python run_experiment.py --bandits 9 --rounds 50 --runs 5 --noise 0.0 0.5 1.0
```

Output CSV has the same schema as before —
`[rowid, runID, noise_level, round, performance]` — saved to
`Global results/` with the algorithm name in the filename, e.g.
`dreamteam_results_rounds100_dims9_tests9_<timestamp>.csv`.

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

## Design notes

- **Environment vs. algorithm separation**: `TeamRewardEnvironment` owns the
  optimal-arm vector and reward noise, so algorithms physically cannot peek
  at the answer. Alternative reward structures (per-dimension credit,
  relative thresholds, etc.) can be added there without touching algorithms.
- **Fresh state per run**: the harness constructs a new algorithm instance
  per run, which replaces the original `clone_bandits()` deep-copy dance.
- **Known quirks preserved in `dreamteam.py`** (documented as class
  constants so they're easy to experiment with): the fixed `0.1` reward
  threshold, the `+1000` beta penalty, and the global budget peak `m=2`.
