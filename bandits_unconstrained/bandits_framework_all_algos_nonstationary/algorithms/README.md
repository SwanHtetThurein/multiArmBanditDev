> **Note — this copy lives in the non-stationary folder.**
> The per-algorithm descriptions below are exact, and the algorithm files here
> are byte-identical to the stationary folder's. But **Part 1 (the experiment)
> describes the stationary setup**: 100 rounds with a hidden optimum that never
> moves. In this folder the run is 300 rounds and the optimum is swapped after
> rounds 101 and 201. See `../README.md` in this folder for the non-stationary
> design, and read Part 1 below as background on the shared problem, the
> environment and the algorithm contract — all of which still apply.

# The Algorithm Suite

A guide to the benchmark: what the experiment is, why each algorithm is in it,
and how each one works.

This document covers the 27 algorithms registered in `__init__.py`. The parent
directory's `README.md` covers how to *run* things; this one covers what is
being run and why.

---

# Part 1 — The experiment

## 1.1 The problem

Assemble a project team where several roles must be filled at once. Each role
has a few candidates. Exactly one candidate per role is secretly the right one.
After fielding a complete team you receive a single overall score — and nothing
else. You are never told which specific roles were right or wrong. Find the
fully correct team within 100 attempts.

Formally this is a **multi-dimensional multi-armed bandit**, or equivalently a
**noisy combinatorial black-box optimization problem**:

| Property | Value |
|---|---|
| Dimensions (bandits) | 9 by default; must be divisible by 3 |
| Arms per dimension | drawn uniformly from 2–5 |
| Search space | ~10^5 distinct teams at the default settings |
| Observation | one scalar reward in [0, 1] per round |
| Budget | 100 rounds per run |
| Per-round context | none — the only thing that varies is the team you pick |
| Constraints | none (unconstrained variant) |

Each dimension also carries a **temporal type** — `early`, `late`, or
`ongoing`, in equal thirds — describing when in a project that aspect of team
structure would naturally be settled. This is metadata: only `dreamteam` and
`dreamteam_orig` consume it (see §1.7).

## 1.2 The environment (`environment.py`)

The environment owns the hidden answer and the noise. Algorithms cannot reach
either — that separation is enforced structurally, not by convention.

**Reward.** For a chosen team, let `p` be the fraction of dimensions matching
the hidden optimum:

```
p    = 1 - (mismatches / n_bandits)
std  = noise_level * p
r    = clip(Normal(p, std), 0, 1)
```

**Scoring.** Each round the harness asks the algorithm for its current
best guess and scores it against the hidden optimum:

```
performance = (number of dimensions where predict_best() is correct) / n_bandits
```

## 1.3 What makes this hard

Three properties, and most of the interesting behaviour in this benchmark traces
back to one of them.

**Credit assignment.** The reward is a single joint number. When a team scores
7/9, the learner is not told *which* seven were right. A wrong arm sitting on an
otherwise excellent team still receives strong positive reinforcement, and a
correct arm on a poor team gets punished. Any algorithm that maintains
independent per-dimension beliefs is fighting this directly.

**The noise is multiplicative.** `std = noise * p` means the *better* the team,
the noisier its evaluation. A perfect team at `noise=1.0` has a standard
deviation of 1.0; a completely wrong team is measured with no noise at all.
This inverts the usual assumption: precisely the teams you care about
distinguishing are the ones you can least trust a single measurement of. It also
violates the homoscedastic-Gaussian likelihood that several surrogates here
assume, and it makes the standard noise-free Expected Improvement formula
over-trust the incumbent.

**The budget is tiny relative to the space.** 100 rounds against ~10^5 teams
means you will observe roughly 0.1% of the space. No algorithm can succeed by
sampling; every one of them must succeed by *generalizing* — by having a model
that predicts unobserved teams from observed ones. The methods here differ
mainly in what that model assumes.

## 1.4 The harness (`experiment.py`)

**Problem generation.** Random arm counts per dimension (2–5), and an equal
`early`/`late`/`ongoing` split, shuffled.

**Test generation.** Each *test* is a random `(initial_bias, optimal_arm)` pair.
`initial_bias` is the team the algorithm starts from — the structure the team is
already using when the study begins. `optimal_arm` is the hidden answer.

**The sweep.** For every noise level, for every repetition, for every test:
construct a **fresh algorithm instance** and a fresh environment, and run for
`total_rounds`.

```
6 noise levels  x  20 repetitions  x  9 tests  =  1080 runs
1080 runs       x  100 rounds               = 108,000 rounds per algorithm
```

Default noise levels: `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`.

**Output.** One CSV per algorithm in `Global results/`, schema
`[rowid, runID, noise_level, round, performance]`, named
`<algorithm>_results_rounds<R>_dims<D>_tests<T>_<timestamp>.csv`.

Because every algorithm faces the identical generated problem and the identical
test scenarios under the same seed, differences between two CSVs are
attributable to the algorithms and not to luck in problem generation.

## 1.5 The contract (`base.py`)

Every algorithm implements three methods. This is the entire interface.

```python
choose(round_num) -> List[int]     # one arm index per dimension
update(arms_chosen, reward) -> None
predict_best()    -> List[int]     # current best guess, scored by the harness
```

Constructed with a `ProblemConfig` (arm counts and temporal types), the
`initial_bias` vector, and the total number of rounds. **Algorithms never see
the optimal arm or the noise level** — only the rewards they are handed.

Note that `choose()` and `predict_best()` are separate. An algorithm may play an
exploratory team while recommending a different one. Every algorithm here uses
that freedom except `random`, which recommends whatever it last played.

## 1.6 What is actually being measured

The harness scores `predict_best()` every round. That is a **simple-regret**
objective — the quality of the recommendation you would make if forced to stop
now — and *not* cumulative regret, which measures reward earned along the way.

The distinction matters more than it first appears. Bubeck, Munos & Stoltz
(2009) proved these two objectives trade off against each other: an algorithm
working hard to earn reward while learning ends up with a worse final answer
than one that explores freely and only commits at the end. Most of the
algorithms in this suite (Thompson sampling, Expected Improvement, UCB) are
built to optimize reward earned, not the final recommendation. Two are not:
`purexp` isolates the allocation question, and `bayesgap` is the only arm
designed for this objective from the start.

## 1.7 Fairness: what is and is not held constant

Stated plainly, because a reviewer will ask.

**Held constant.** The generated problem, the test scenarios, the harness, the
scoring rule, the round budget, a fresh instance per run, and — for the four
algorithms sharing the BOCS surrogate — the model, priors and discrete
optimizer.

**Not held constant, and worth disclosing:**

- **Side information.** `dreamteam` and `dreamteam_orig` consume
  `bandit_types` (the early/late/ongoing split). No other algorithm does. That
  matches how the baselines in the source papers work, but it means DreamTeam
  has access to structure the others ignore.
- **Switching freedom.** DreamTeam is switch-constrained by design; every other
  algorithm may jump anywhere in the space each round. A fair account should
  report switch counts alongside performance.
- **Tuning.** DreamTeam's parameters were fixed in advance; the competitors use
  documented defaults from their papers. None received a serious
  hyperparameter search.
- **Reward structure.** The reward here is *linear and separable* — each
  dimension contributes independently to `p`. That is a structural gift to
  models with first-order features, and it means algorithms built to capture
  interactions (BOCS's pairwise terms, the neural methods) are paying for
  expressiveness the environment does not reward. An interaction-reward
  environment is the natural follow-up experiment.

## 1.8 Implementation conventions

Shared across the suite so that comparisons are about algorithms, not
engineering:

- **Round 1 always plays `initial_bias`**, so every algorithm starts from the
  same team.
- **Model-based methods use `N_INIT = 10`** random rounds before their
  surrogate takes over.
- **Discrete maximization is multi-start greedy local search** over
  single-arm changes (`N_RESTARTS = 4`), everywhere it is needed. This is
  deliberate: COMBO's own appendix (§3.1) tested adding simulated annealing to
  its local search and found "the optimum of all 3 methods is hardly better
  than the optimum discovered solely by BFLS", so holding the optimizer fixed
  costs nothing and removes a confound.
- **Everything is numpy-only** except `combo.py`, which uses `scipy.optimize`.
  No deep-learning framework; the MLPs are hand-written with manual backprop.
- **Each module's docstring cites its paper** and lists what was simplified
  relative to it.

---

# Part 2 — Map of the suite

## 2.1 All 27 algorithms

| name | family | models interactions? | one-line idea |
|---|---|---|---|
| `random` | floor | no | uniform random every round |
| `dreamteam` | reference | no | this project's modified DreamTeam |
| `dreamteam_orig` | reference | no | DreamTeam exactly as published (CHI 2018) |
| `sa` | model-free | n/a | simulated annealing with geometric cooling |
| `ols` | model-free | n/a | greedy neighbourhood sweep, then random restart |
| `regevo` | model-free | n/a | aging evolution over a population of teams |
| `bocs` | surrogate BO | yes, explicit pairwise | sparse Bayesian linear model + Thompson sampling |
| `bocs_hs` | surrogate BO | yes, explicit pairwise | the same, with the real horseshoe prior (Gibbs) |
| `combo` | surrogate BO | yes, via kernel | GP with a graph-diffusion kernel + EI |
| `combo_slice` | surrogate BO | yes, via kernel | the same, with real slice-sampled hyperparameters |
| `smac` | surrogate BO | yes, via tree splits | random-forest surrogate + EI |
| `gp_onehot` | surrogate BO | weakly | textbook GP-EI on a one-hot encoding |
| `cocabo` | surrogate BO | yes, via kernel | per-dimension EXP3 agents + overlap-kernel GP |
| `neurallinear` | neural | yes, learned | Bayesian linear layer on a trained MLP |
| `bootnn` | neural | yes, learned | bootstrap ensemble of MLPs |
| `linucb` | acquisition | yes, explicit pairwise | the BOCS surrogate with a UCB bonus |
| `kg` | acquisition | yes, explicit pairwise | the BOCS surrogate with Knowledge Gradient |
| `purexp` | acquisition | yes, explicit pairwise | the BOCS surrogate with *no* acquisition |
| `bayesgap` | best-arm ID | yes, explicit pairwise | gap-based allocation for the final recommendation |
| `cucb` | combinatorial bandit | no | optimistic per-base-arm estimates |
| `cts` | combinatorial bandit | no | Thompson sampling per base arm |
| `casmopolitan` | surrogate BO | yes, via kernel | trust-region BO over a Hamming ball that grows, shrinks and restarts |
| `glm_fpl` | GLM | yes, explicit pairwise | logistic-link GLM explored by perturbing observed rewards |
| `sts` | acquisition | yes, explicit pairwise | satisficing Thompson sampling — good enough, fewer role changes |
| `gp_nei` | acquisition | weakly | Noisy EI: the incumbent's uncertainty integrated out |
| `gp_ucb` | acquisition | weakly | IGP-UCB on the shared one-hot GP |
| `gp_ts` | acquisition | weakly | GP Thompson sampling on the shared one-hot GP |

## 2.2 The three controlled comparisons

The suite is not 21 unrelated methods. It is built around three comparisons
where exactly one thing changes.

**(a) Acquisition rule, surrogate held fixed — on a LINEAR surrogate.**
`bocs`, `linucb`, `kg`, `purexp` and `sts` share the identical second-order
Bayesian linear model, identical priors, and the identical local-search
optimizer. They differ *only* in how they choose the next team:

| arm | acquisition | question it answers |
|---|---|---|
| `bocs` | Thompson sampling | posterior sampling |
| `linucb` | upper confidence bound | optimism |
| `kg` | knowledge gradient | one-step-optimal decision theory |
| `sts` | satisficing Thompson sampling | is chasing the exact optimum worth it at 100 rounds? |
| `purexp` | none — uniform allocation | is adaptive allocation worth anything? |

Any difference among these five is attributable to the acquisition rule and
nothing else. `purexp` against `random` isolates the value of the *model*;
`purexp` against the others isolates the value of *adaptive allocation*.

**(a′) The same grid on a GP surrogate.** `gp_onehot`, `gp_nei`, `gp_ucb` and
`gp_ts` share one kernel, one hyperparameter fit and one optimizer:

| arm | acquisition |
|---|---|
| `gp_onehot` | Expected Improvement (noise-free formula) |
| `gp_nei` | Noisy EI — incumbent uncertainty integrated out |
| `gp_ucb` | IGP-UCB |
| `gp_ts` | GP Thompson sampling |

Having the acquisition comparison on *two* surrogates rather than one is what
separates "this acquisition rule is better" from "this acquisition rule
happens to suit a linear model" — a claim the suite could not make before.

**(b) Faithful vs. simplified inference.** Two pairs, where the second member
implements the inference its paper actually specifies:

| simplified | faithful | what changes |
|---|---|---|
| `bocs` (grouped ridge) | `bocs_hs` (horseshoe, Gibbs) | the sparsity prior |
| `combo` (MAP hyperparameters) | `combo_slice` (slice sampling) | hyperparameter marginalization |

These exist so that two documented fairness caveats become measured quantities
instead of disclaimers.

**(c) Published vs. modified DreamTeam.** `dreamteam_orig` is the CHI 2018
algorithm; `dreamteam` is this project's descendant. Running them head to head
measures what the modifications did — principally the replacement of the
standard Beta update with a threshold-and-penalty rule.

## 2.3 How to read the families

- **Model-free** (`sa`, `ols`, `regevo`) build no model of the reward at all.
  They are the honest floor for model-based methods: whatever a surrogate buys
  must show up as an improvement over guided local search, not merely over
  random.
- **Surrogate BO** methods fit a regression model over the whole team space and
  use it to decide where to look. They differ in the model's inductive bias —
  which teams the model believes are similar to which.
- **Neural** methods learn that similarity structure instead of assuming it.
- **Combinatorial bandits** keep per-dimension estimates with an exact
  separable oracle — the principled version of what DreamTeam does by hand.
- **The GLM arm** (`glm_fpl`) is the only one whose *likelihood* matches this
  environment. Everything else assumes homoscedastic Gaussian noise; the
  environment's noise is bounded and coupled to the mean.

---

# Part 3 — The algorithms

Each entry: where it comes from, **why it is in the benchmark**, and **how it
works** in detail.

---

## `random` — Random baseline
`random_baseline.py` · performance floor

### Why it's in the benchmark
It defines zero. Every other result is only meaningful relative to it, and it
catches harness bugs: if a real algorithm cannot beat uniform random on a
separable reward, something is broken. It is also the minimal template for
writing a new plug-in.

Its expected score is the average of `1/n_d` across dimensions — roughly 0.3 at
the default arm counts, because a random guess in a 2-to-5-arm dimension is
right that often by chance.

### How it works
Each round it draws a uniformly random arm in every dimension. `update()` does
nothing — it has no state to update. `predict_best()` returns the team it just
played, so it is scored on a fresh random guess every round.

That last detail is deliberate and distinguishes it from `purexp`, which also
samples uniformly but recommends a *model-based* answer. The pair brackets the
value of the surrogate: `random` is uniform allocation with no model, `purexp`
is uniform allocation with a good one.

---

## `dreamteam` — DreamTeam (this project's modified version)
`dreamteam.py` · reference

### Why it's in the benchmark
This is the algorithm the project is about. Everything else exists to give it a
context. It is also the only method here designed under a constraint the others
ignore: a real team cannot be reorganized arbitrarily every round, so DreamTeam
limits *when* and *how many* dimensions may change at once. That makes a raw
performance comparison slightly unfair in its disfavour — the competitors are
unconstrained — which is why switch-count accounting belongs in any writeup.

### How it works
Per-dimension Thompson sampling, wrapped in two layers of temporal constraint.

**1. Beliefs.** Each dimension keeps a Beta(α, β) posterior per arm, initialized
Beta(1, 1).

**2. Thompson sampling.** Each round, for every dimension, draw one sample from
each arm's Beta and normalize the draws into a probability vector.

**3. Per-dimension stickiness schedule.** A discount `δ` shrinks the
probability of moving off the currently held arm, on a schedule set by the
dimension's temporal type (`half = total_rounds / 2`):

```
early    δ = 1 / (1 + e^(round - half))     free early, frozen late
late     δ = 1 / (1 + e^(half - round))     frozen early, free late
ongoing  δ = 1                              never restricted
```

The renormalization moves the discounted mass onto the current arm:
`p_i ← p_i · δ` for `i ≠ current`, and the current arm absorbs the remainder.

**4. Global switching budget.** Team adaptability is modelled as a downward
parabola peaking mid-run:

```
y = 2 · (1 - ((round - half) / half)²)
```

Let `z` be the total probability mass sitting off-current across all
dimensions — the expected number of simultaneous changes. If `z > y`, every
dimension is scaled **proportionally** by `y/z`, which drives the expected
number of changes to exactly `y`.

**5. Selection.** Sample (not argmax) one arm per dimension from the final
distributions. Round 1 always plays `initial_bias`.

**6. Update — the modified part.**

```
if reward > 0.1:  alpha[arm] += reward
else:             beta[arm]  += 1000
```

**7. Recommendation.** Per-dimension argmax of `α/(α+β)`.

### Caveats
Step 6 is not a Bayesian update. A reward at or below 0.1 adds 1000 to β, which
collapses that arm's posterior mean to essentially zero in one observation and
makes it practically unrecoverable. Failure is also never credited
proportionally: rewards of 0.11 and 0.99 both only increase α. Under this
environment's reward-scaled noise, a good team can draw a low reward by chance
and have several of its *correct* arms permanently condemned. Compare against
`dreamteam_orig` to measure the effect.

---

## `dreamteam_orig` — DreamTeam as published
`dreamteam_original.py` · reference
**Source:** Zhou, Valentine & Bernstein, CHI 2018 (DOI 10.1145/3173574.3173682)

### Why it's in the benchmark
Without it, the paper cannot say what its own modifications did. `dreamteam` is
a descendant of the published algorithm, not the algorithm itself, and the two
differ in ways that plausibly matter a great deal under this noise model. This
arm turns "we modified it" into a measured quantity.

### How it works
Identical to `dreamteam` in structure — Beta posteriors, Thompson sampling,
the same sigmoid schedules, the same parabolic budget, probabilistic
selection — with two substantive differences.

**The posterior update is the standard one:**

```
alpha[arm] += r
beta[arm]  += (1 - r)
```

Every observation moves the posterior by exactly one unit of evidence, split
between success and failure in proportion to the reward.

**The global constraint shares the cut equally rather than proportionally.**
Where `dreamteam` scales all dimensions by `y/z`, the published rule computes
a per-dimension discount:

```
d_global = clamp(1 - excess / (z_d · D), 0, 1)      where excess = z - y
```

Applied to dimension `d`, this subtracts `excess/D` from its off-current mass —
an equal *absolute* share from every dimension, rather than an equal
proportional one. A dimension that only mildly wants to move can be silenced
entirely, while proportional scaling would have preserved the relative ordering.

### A verified property worth knowing
Because `d_global` is clamped at 0, the published budget is a **soft** cap.
When a dimension's `z_d` is smaller than `excess/D` its discount clamps and it
surrenders only `z_d` instead of its full share; the shortfall is not
redistributed, so the realized total sits *above* `y`. Measured over 1500
trials: the overshoot occurred in exactly the trials where the clamp fired, and
was exactly zero otherwise. `dreamteam`'s proportional rule hits `y` exactly
every time. Do not attribute a `dreamteam` vs `dreamteam_orig` difference
solely to the posterior update without accounting for this.

### Also in this file
A standalone `DreamTeam` class reproducing the paper's fixed five-dimension
system — Hierarchy, Interaction Patterns, Norms of Engagement, Decision-Making
Norms, Feedback Norms; 3×3×3×5×3 = 405 structures — with its own
`initialize()` / `get_current_structure()` / `step(reward)` interface. Both
classes share one engine, so the math cannot drift between them. The harness
plug-in generalizes to any number of dimensions.

---

## `sa` — Simulated Annealing
`simulated_annealing.py` · model-free
**Source:** baseline in both BOCS (arXiv:1806.08838) and COMBO (arXiv:1902.00448)

### Why it's in the benchmark
It is the standard non-Bayesian reference point in both combinatorial-BO papers
this project follows, and it is the honest bar for the surrogate methods.
Beating `random` proves very little; beating guided local search is the claim
that matters. It costs essentially nothing to run.

### How it works
No model of the reward whatsoever. It holds a single incumbent team and a
temperature that cools geometrically from `T_START = 0.25` to `T_END = 0.01`
across the run.

Each round it proposes a neighbour — one randomly chosen dimension switched to
a different arm — and plays it. On observing the reward it applies the
Metropolis criterion:

```
accept if  r >= incumbent_reward
otherwise accept with probability  exp((r - incumbent_reward) / T_t)
```

Early rounds run hot and accept almost anything, which is exploration; late
rounds run cold and accept only improvements, which is exploitation. Round 1
plays `initial_bias` and its reward seeds the incumbent.

`predict_best()` returns the team with the best **average** observed reward,
not the best single draw — a much more stable estimate under multiplicative
noise, and the convention shared by all three model-free arms.

### Caveats
Classical SA assumes near-noiseless evaluations. Here the accept/reject test
compares two noisy draws, so the incumbent can be displaced by luck. That is
standard SA behaviour on a stochastic objective and is kept as-is rather than
silently de-noised — but it is a genuine reason to expect SA to underperform
here relative to its reputation on deterministic problems.

---

## `ols` — Oblivious Local Search
`local_search.py` · model-free
**Source:** baseline in BOCS (arXiv:1806.08838)

### Why it's in the benchmark
The simplest structured search there is, and the natural companion to `sa`:
same neighbourhood, no randomness in the acceptance rule. Together they bracket
what pure local search achieves before any modelling. Its "oblivious" restart —
deliberately carrying no memory of the landscape between restarts — is exactly
what makes it a clean lower reference.

### How it works
Hold an incumbent, evaluate every team at Hamming distance 1 from it, move to
the best if it beats the incumbent, repeat. When no neighbour improves, the
incumbent is a local optimum and the search restarts from a fresh random team.

The paper's version assumes it can evaluate a whole neighbourhood at will. Here
each evaluation costs one round, so the sweep is spread out: **one neighbour per
round, in shuffled order**, with the accept-or-restart decision taken when the
sweep completes. With 9 dimensions of 2–5 arms a sweep costs roughly 20 rounds,
so a 100-round run completes a handful of hill-climbing moves — which is itself
an informative result about how expensive exhaustive neighbourhood search is
under a tight budget.

Round 1 plays `initial_bias`, whose reward becomes the incumbent's score.
`predict_best()` returns the best-average-reward team observed.

---

## `regevo` — Regularized (aging) Evolution
`regularized_evolution.py` · model-free
**Source:** baseline in COMBO; method from Real et al., AAAI 2019 (arXiv:1802.01548)

### Why it's in the benchmark
COMBO's strong non-Bayesian competitor, and the one model-free method whose
mechanics fit this problem naturally: a team *is* a categorical genotype, and
single-dimension mutation *is* the obvious move operator. It also completes the
model-free family — annealing, hill-climbing, and population search.

### How it works
Maintain a fixed-size population of 16 evaluated teams. Each round:

1. Sample a tournament of 5 members uniformly from the population.
2. Take the best-scoring one as the parent.
3. Mutate it: change one randomly chosen dimension to a different arm.
4. Evaluate the child and append it to the population.
5. **Evict the oldest member — not the worst.**

Step 5 is the "regularization", and it is the reason this method belongs in a
*noisy* benchmark. Good solutions survive only by being re-discovered, so the
search cannot lock permanently onto a team that happened to draw a lucky
high reward once. Plain non-aging evolution is badly biased toward exactly that
failure mode under multiplicative noise.

Round 1 plays `initial_bias`; rounds 2–16 fill the population with random teams.
`predict_best()` returns the best-average-reward team observed.

---

## `bocs` — Bayesian Optimization of Combinatorial Structures
`bocs.py` · surrogate BO
**Source:** Baptista & Poloczek, ICML 2018 (arXiv:1806.08838)

### Why it's in the benchmark
The canonical method for exactly this problem shape, and the one whose modelling
assumption is most directly interpretable in team terms. Its second-order
features are precisely the *"this candidate only works alongside that one"*
interactions that per-dimension learners like DreamTeam structurally cannot
represent. It is also half of the BOCS-vs-COMBO rivalry that gives the suite
its spine.

### How it works
**Features.** A team is mapped to a sparse binary feature vector:

```
phi(team) = [ intercept ]
          + [ 1[dim_d = arm_a]                    for every (d, a) ]      first-order
          + [ 1[dim_d = a AND dim_e = b]          for every d<e, a, b ]   second-order
```

At 9 dimensions with 2–5 arms this is roughly 470 features, of which exactly 46
are non-zero for any given team.

**Surrogate.** A conjugate Normal-Inverse-Gamma Bayesian linear regression on
those features. After every observation the exact posterior over all weights
*and* the noise variance is available in closed form — no approximation, no
gradient steps. Sufficient statistics are rank-1 updated per round.

**Sparsity.** The paper uses a horseshoe prior; this implementation substitutes
a grouped ridge — prior variance 1.0 on first-order weights, 0.25 on pairwise
weights, so interactions are shrunk harder than main effects. Cheap, conjugate,
and adequate at ~470 features. See `bocs_hs` for the faithful version.

**Acquisition — Thompson sampling.** Draw `σ² ~ InvGamma(a_n, b_n)`, then
`w ~ N(m, σ²·Prec⁻¹)`, giving one plausible model of the world. Maximize that
sampled surrogate over the team space by multi-start greedy local search
(4 restarts, one seeded from the best observed team). Early on the posterior is
wide, so draws vary and the algorithm explores; as data accumulates the draws
concentrate and it exploits.

**Recommendation.** Local search on the posterior *mean* (no sampling).

Round 1 plays `initial_bias`; rounds 2–10 are random initialization.

### Caveats
BOCS's appendix E is explicit that substituting a point estimate for the
posterior sample "often results in purely exploitative behavior that fails to
find the global optimum" — the Thompson step is load-bearing, not decorative.
Also note that this environment's reward is separable, which means BOCS's
pairwise terms model structure that is not actually present; it pays for
expressiveness the environment does not reward.

---

## `bocs_hs` — BOCS with the real horseshoe prior
`bocs_horseshoe.py` · surrogate BO
**Sources:** Baptista & Poloczek 2018; sampler from Makalic & Schmidt, IEEE SPL
2016; fast draw from Bhattacharya, Chakraborty & Mallick, Biometrika 2016;
prior from Carvalho, Polson & Scott, Biometrika 2010

### Why it's in the benchmark
`bocs.py` substitutes grouped ridge for the paper's horseshoe prior — a
simplification listed as a fairness caveat in the comparison report. BOCS's own
appendix F.3 shows the horseshoe's advantage over standard Bayesian linear
regression is largest **specifically when N is small**, which is exactly the
100-round regime. This arm converts that caveat from a disclaimer into a
measurement. Kept separate from `bocs.py` rather than replacing it, so existing
results stay valid and the simplification becomes an explicit experimental
factor.

### How it works
Same features, same Thompson-sampling acquisition, same local search. What
changes is the prior and how the posterior is obtained.

**The hierarchy** (Makalic & Schmidt's auxiliary-variable form, which makes
every conditional an inverse-gamma and so needs no Metropolis step):

```
beta_j | lambda_j, tau, sigma^2 ~ N(0, lambda_j^2 tau^2 sigma^2)
lambda_j^2 | nu_j              ~ InvGamma(1/2, 1/nu_j)
nu_j                           ~ InvGamma(1/2, 1)
tau^2 | xi                     ~ InvGamma(1/2, 1/xi)
xi                             ~ InvGamma(1/2, 1)
sigma^2                        ~ InvGamma(a0, b0)
```

The heavy tail on `lambda_j` lets a genuinely important interaction escape
shrinkage entirely, while the mass near zero crushes the hundreds of irrelevant
pairwise terms. Grouped ridge, applying one shrinkage scale to every pairwise
weight, cannot do both.

**The expensive step, done cheaply.** With ~470 features but at most 100
observations, the textbook O(p³) Cholesky for the β draw is wasteful.
Bhattacharya et al.'s algorithm samples the same Gaussian in O(n²p) via an
n×n solve — not an approximation, the same distribution computed in the
cheaper direction.

**Persistent chain.** The Gibbs chain is burned in once (40 sweeps) when enough
data exists, then advanced 4 sweeps per round as observations arrive.
Consecutive rounds' posteriors differ by one data point, so the previous state
is an excellent warm start. This is what makes genuine MCMC affordable inside a
1080-run sweep; the paper re-runs its sampler from scratch each iteration.

**Recommendation.** Local search on a running average of the post-burn-in β
draws (Rao-Blackwellized), not a single noisy draw.

---

## `combo` — Combinatorial BO via the Graph Cartesian Product
`combo.py` · surrogate BO
**Source:** Oh, Tomczak, Gavves & Welling, NeurIPS 2019 (arXiv:1902.00448)

### Why it's in the benchmark
The direct rival to BOCS — it cites and beats it — and the other half of the
suite's central rivalry. Where BOCS hand-designs which interactions can exist,
COMBO lets a kernel decide which teams are similar. It is the strongest
argument in the literature that the *right notion of distance* between
combinatorial objects matters more than an explicit feature expansion.

### How it works
**The space as a graph.** Build one sub-graph per dimension — for a categorical
variable with `n` values, the complete graph `K_n` — and take the **graph
Cartesian product** across dimensions. The result is a graph whose vertices are
all possible teams and whose shortest-path distance is exactly Hamming
distance.

**The kernel.** A diffusion kernel on that graph. Because the complete graph's
Laplacian eigenstructure is analytic, the per-dimension kernel has a closed
form, and the Cartesian product means the full kernel factorizes as a product
over dimensions:

```
k(x, x') = sigma_f^2 * PROD_d  k_d(x_d, x'_d)

k_d(a, a) = (1 + (n_d - 1) e^(-n_d beta_d)) / n_d
k_d(a, b) = (1 - e^(-n_d beta_d)) / n_d           for a != b
```

No graph Fourier transform is needed — the eigendecomposition is available in
closed form. Each dimension gets its own `beta_d` (ARD), so the model can learn
that some roles matter more than others.

**Fitting.** Hyperparameters (`beta_d`, signal variance, noise variance) are
fit by maximizing the marginal likelihood with weak log-normal priors, using
L-BFGS-B, refit every 10 rounds for speed.

**Acquisition.** Expected Improvement, maximized by multi-start greedy local
search. `predict_best()` local-searches the posterior mean.

### Caveats
The paper does *not* MAP-fit its hyperparameters — it slice-samples them with
horseshoe priors (appendix §2.3), because with ten hyperparameters and 20–100
observations a point estimate is badly overconfident. See `combo_slice`. Also,
a GP this flexible needs a fair number of samples to sharpen, and it must learn
the noise level from scratch — both are structural disadvantages at a 100-round
budget.

---

## `combo_slice` — COMBO with slice-sampled hyperparameters
`combo_slice.py` · surrogate BO
**Sources:** Oh et al. 2019; Murray & Adams, NeurIPS 2010 (arXiv:1006.0868);
Neal, Annals of Statistics 2003

### Why it's in the benchmark
The second of the two faithful-inference arms. COMBO's appendix §2.3 specifies
100 burn-in iterations then 10 hyperparameter samples per BO round, with
horseshoe priors — not the MAP fit `combo.py` uses. That is not a detail at
this budget: a point estimate commits early to one story about which dimensions
matter, its posterior variance collapses, and Expected Improvement stops
exploring. Averaging the acquisition over hyperparameter samples is what keeps
it honest.

### How it works
Identical kernel to `combo`, so the two arms differ only in inference.

**What is sampled.** `log beta_d` for each dimension (half-Cauchy prior, the
horseshoe-style heavy tail the paper uses), `log` signal variance (log-normal),
and `log` noise variance (half-Cauchy).

**The sampler.** Univariate slice sampling with stepping-out and shrinkage,
applied to each hyperparameter in turn against the log marginal likelihood plus
log prior. Slice sampling needs no gradients, no step size, and no accept/reject
tuning — which is precisely why it is the right tool for a hyperparameter
posterior with no closed form.

**Marginalized acquisition.** Three hyperparameter samples are retained, each
with its own cached Cholesky and posterior weights. Expected Improvement is
computed under each and **averaged**. A team only scores highly if it looks
promising under several plausible stories about the kernel.

**Speed adaptations** (documented deviations): burn-in is 20 sweeps once rather
than 100 restarted every round, the chain persists across rounds and is
advanced 2 sweeps every 10 rounds, and 3 samples are kept rather than 10.
Roughly an order of magnitude cheaper, which is what makes genuine MCMC
affordable across 1080 runs.

---

## `smac` — Sequential Model-based Algorithm Configuration
`smac.py` · surrogate BO
**Source:** Hutter, Hoos & Leyton-Brown, LION 2011; baseline in both BOCS and COMBO

### Why it's in the benchmark
The standard "tree surrogate instead of a GP" comparison, and a structurally
different uncertainty model from everything else here. Random forests handle
unordered categorical variables **natively** — no kernel, no one-hot encoding,
no continuous relaxation — and they capture interactions between dimensions
without an explicit second-order expansion. It is also the method in the whole
suite with the most mature real-world track record for exactly this kind of
categorical configuration problem.

### How it works
**Surrogate.** An ensemble of 10 randomized regression trees fit directly on
the raw categorical team vector. At each node: consider a random subset of
dimensions (`mtry = D/3`), draw one random value-subset split per candidate
dimension, and keep the split with the best variance reduction. Depth is capped
at 8, with a minimum of 4 samples to attempt a split.

**Uncertainty.** Frequentist, not Bayesian — the mean and standard deviation of
predictions **across trees**:

```
mu(team) = mean over trees
sd(team) = std-dev over trees
```

Regions the trees disagree about are the uncertain ones.

**Acquisition.** Expected Improvement computed from `(mu, sd)` exactly as for a
GP, maximized by multi-start greedy local search.

**Interleaved random configurations.** Every second post-initialization round
plays a uniformly random team regardless of the model. This is a real SMAC
feature, not a simplification: it guarantees continued exploration even if the
surrogate is badly wrong.

The forest is refit every 5 new observations rather than every round, for speed.

### Caveats
SMAC's original machinery for noisy objectives — intensification, racing,
capped runs across problem instances — has nothing to act on here, since every
team is evaluated exactly once per round. Tree-ensemble variance is also a
cruder uncertainty estimate than a GP posterior, so exploration is less
principled at small N.

---

## `gp_onehot` — Textbook GP-EI on a one-hot encoding
`gp_onehot.py` · surrogate BO
**Source:** the "GP-BO with one-hot encoding" baseline in BOCS (arXiv:1806.08838)

### Why it's in the benchmark
It answers the question a reviewer will ask first: *how much does the
combinatorial-specific machinery actually buy over textbook Bayesian
optimization?* This is the obvious thing anyone would try before building a
diffusion kernel or an interaction feature map, and BOCS includes it precisely
to show it is not enough. Without this arm, `bocs` and `combo` are being
compared only to each other and to model-free search — there is no measurement
of what *specialization* is worth.

### How it works
A standard squared-exponential GP applied to the one-hot encoded team, with no
diffusion kernel and no interaction expansion.

**The kernel, and why there is no one-hot matrix in the code.** For one-hot
encoded categorical vectors the squared distance between two teams is exactly
twice their Hamming distance:

```
||onehot(x) - onehot(x')||^2 = 2 * hamming(x, x')
```

so the SE kernel collapses to a closed form:

```
k(x, x') = sigma_f^2 * exp( -hamming(x, x') / rho )
```

This is an exact algebraic reduction, not an approximation — it just avoids
materializing the one-hot matrix and the pairwise distance computation.

**Fitting.** Only two free hyperparameters (`rho`, and the noise-to-signal
ratio `eta`), because the signal variance has a closed-form maximizer at each
grid point — the profile-likelihood trick:

```
sigma_f^2 = yc' (R + eta I)^-1 yc / n
```

So the fit is a deterministic grid search over 8 lengthscales × 6 noise ratios
on the concentrated log marginal likelihood, re-tuned every 10 rounds. A grid
cannot fail to converge, which matters inside a 1080-run sweep.

**Acquisition.** Expected Improvement, maximized by multi-start greedy local
search — the same optimizer as `combo` and `smac`, so acquisition-optimizer
differences do not confound the kernel comparison.

### Caveats
This is deliberately the *unspecialized* GP, and its known pathology is the
point: an isotropic kernel treats all dimensions as equally important and all
non-matches as equally distant. The principled fix (Garrido-Merchán &
Hernández-Lobato, arXiv:1805.03463) applies the categorical transformation
inside the covariance function; it is not implemented here.

---

## `cocabo` — Continuous and Categorical Bayesian Optimisation
`cocabo.py` · surrogate BO
**Source:** Ru, Alvi, Nguyen, Osborne & Roberts, ICML 2020 (arXiv:1906.08878)

### Why it's in the benchmark
Its design is unusually well matched to this problem. Rather than forcing
categorical variables into a continuous relaxation, CoCaBO puts an
**independent bandit agent on each categorical dimension** and lets a GP handle
the rest — which is very nearly a description of "one agent per role, each
choosing among that role's candidates". It is also the only post-2019
categorical-BO method in the suite, and its kernel encodes a genuinely
different similarity assumption from everything else here.

### How it works
Two mechanisms running together.

**1. Multi-agent EXP3.** One EXP3 agent per dimension, each keeping weights over
its own arms and playing from the mixture

```
p_a = (1 - gamma) * w_a / sum(w) + gamma / n
```

On observing the shared team reward `r`, the agent that played arm `a` applies
the importance-weighted update `w_a *= exp(gamma * (r / p_a) / n)`. The agents
never communicate — they coordinate only through the reward they all receive.
That decomposition is what keeps the cost linear rather than exponential in the
number of dimensions. EXP3 rather than a stochastic bandit is the right choice
because each agent's reward is *non-stationary*: it depends on what the other
agents are doing.

**2. The categorical overlap kernel.**

```
k(h, h') = (sigma_f^2 / D) * sum_d 1[h_d = h'_d]
```

Similarity is **linear** in the number of shared choices. This is a materially
different modelling assumption from anything else in the suite: `gp_onehot`
decays exponentially in Hamming distance and `combo` uses a diffusion kernel,
both of which make distant teams nearly uncorrelated, whereas the overlap
kernel keeps a floor of correlation and pools information far more
aggressively across the space. At 100 rounds over 10^5 teams, that pooling is
close to the whole game.

### An honest note on the pure-categorical case
CoCaBO's published algorithm splits each round: EXP3 picks the categorical
values, then the GP optimizes the *continuous* variables conditioned on them.
This problem has no continuous variables, so taken literally CoCaBO would
degenerate to plain multi-agent EXP3 and the kernel would do nothing.

This implementation instead follows CoCaBO-B, the paper's batch variant, where
EXP3 fixes part of the batch and the GP proposes the rest: **half the rounds
play the EXP3 draw directly**, and the rest play the GP's EI maximizer over
candidates local-searched from the EXP3 draw and from the incumbent. Crucially
the EXP3 agents are updated **only** on rounds where the action really came
from their own distribution — otherwise the `r / p_a` importance weighting
would be biased. Both halves of the method stay live and the bandit estimator
stays unbiased.

---

## `neurallinear` — Bayesian linear layer on a learned representation
`neural_linear.py` · neural
**Source:** Riquelme, Tucker & Snoek, ICLR 2018 (arXiv:1802.09127)

### Why it's in the benchmark
The winner of a 15-method bake-off on how to get usable uncertainty out of a
neural network for sequential decisions, and the representative of "let the
model learn its own features". Where BOCS hand-designs the interaction features
and COMBO assumes a kernel, NeuralLinear learns the similarity structure from
data. Its headline finding — that *exact-but-simple* beats
*approximate-but-complex* in online settings — is a claim this benchmark can
test directly, and at 100 rounds it should hold roughly ten times more strongly
than in the paper's own experiments.

### How it works
Deliberately decoupled into two stages.

**1. Representation (learned, slow).** A 2-hidden-layer MLP of sizes (48, 24),
implemented directly in numpy with manual backprop and Adam — no deep-learning
framework at this scale. Input is the concatenated one-hot encoding of the
team. Trained on all observed `(team, reward)` pairs for 300 full-batch epochs,
every 10 rounds.

**2. Uncertainty (exact, fast).** The network is *not* trusted for exploration.
Its last hidden layer is taken as a learned feature map `phi(x) = [1, A2]`, and
an **exact** conjugate Normal-Inverse-Gamma Bayesian linear regression is fit on
top of those features. Thompson sampling on that linear layer drives
exploration — the same math as BOCS, applied to learned rather than
hand-designed features.

**The key bookkeeping detail.** Whenever the network is retrained, the
representation changes, so the last layer's sufficient statistics are
**recomputed from scratch** on the new representation of *all* past data.
Between retrainings they are updated incrementally. Getting this wrong is the
most common way to break NeuralLinear, which is why the paper calls it out.

Acquisition maximizes the sampled linear model by multi-start greedy local
search. `predict_best()` uses the posterior mean.

---

## `bootnn` — Bootstrapped neural-network ensemble
`bootstrapped_nn.py` · neural
**Sources:** baseline in Riquelme et al. 2018, NeuralUCB and NeuralTS; method
from Osband, Blundell, Pritzel & Van Roy, NIPS 2016 (arXiv:1602.04621)

### Why it's in the benchmark
It appears as a baseline in **three** of the five source papers, which makes it
the field's standard cheap alternative to an analytic posterior. More usefully,
it fails differently from NeuralLinear, and pairing them isolates *where*
neural uncertainty comes from:

- **NeuralLinear** gets uncertainty from an exact Bayesian posterior on the
  last hidden layer, treating the representation as fixed and known. It
  underestimates uncertainty when the representation is wrong.
- **bootnn** gets uncertainty from variation across *independently trained
  networks*, so it includes disagreement about the representation itself. It
  underestimates uncertainty when all members collapse to the same fit.

### How it works
Five small MLPs of sizes (32, 16), each trained on a bootstrap resample of the
same data, retrained every 10 rounds for 150 epochs.

**Bootstrap weights are fixed when a data point arrives.** Each observation is
assigned a Poisson(1) weight vector over the five members at the moment it is
observed — the online bootstrap — so member `k` trains on a stable weighted
resample rather than a freshly redrawn one at each retraining. Training
minimizes weighted MSE.

**Acquisition — bootstrap Thompson sampling.** Pick one ensemble member
uniformly at random and act greedily with respect to *its* prediction,
maximized by multi-start local search. Acting greedily under one randomly
chosen posterior sample is exactly the Thompson-sampling step; the ensemble
stands in for the posterior.

`predict_best()` maximizes the **ensemble mean** prediction — no exploration
bonus.

---

## `linucb` — UCB acquisition on the BOCS surrogate
`lin_ucb.py` · acquisition
**Source:** Li, Chu, Langford & Schapire, WWW 2010 (arXiv:1003.5956); baseline
in NeuralUCB and NeuralTS

### Why it's in the benchmark
It is a **controlled experiment, not a new model**. `bocs` already fits this
exact surrogate and explores by Thompson sampling. This arm keeps the surrogate
byte-for-byte identical — same features, same priors, same optimizer — and
swaps only the acquisition rule for the optimism-based one. Running the two
side by side isolates a single factor: *optimism in the face of uncertainty
versus posterior sampling*. That is a far cleaner comparison than adding an
algorithm that differs in several ways at once.

### How it works
Score each candidate team by

```
score(team) = mu(team) + ALPHA * sd(team)          ALPHA = 1.0
```

where `mu` and `sd` come from the same second-order Bayesian linear posterior
`bocs` uses, and maximize by multi-start greedy local search.

**Two implementation details worth knowing.** The posterior covariance
`A = (X'X + prior_prec)^-1` is maintained incrementally by the
**Sherman-Morrison** identity rather than re-factorized each round. Because the
feature vector is a 0/1 indicator, `A @ phi` is a column sum and `phi' A phi` is
a sub-block sum, so both the update and the per-candidate variance are cheap; an
exact re-inversion runs every 25 observations to stop numerical drift. The
noise scale uses the posterior mean of the Inverse-Gamma,
`sigma^2 = b_n / (a_n - 1)`, so the confidence width adapts to how noisy the run
actually is rather than being a fixed constant.

`predict_best()` maximizes the posterior mean with no exploration bonus.

### Adaptation note
Textbook LinUCB assumes a per-round *context* revealed before choosing, and a
small explicit arm set. This environment has neither: the only thing that varies
is the team picked. So the feature vector is the team's own encoding, and the
"arm set" is the whole combinatorial space searched by local search — the same
adaptation `neural_linear.py` makes to the contextual-bandit methods.

---

## `kg` — Knowledge Gradient with correlated beliefs
`knowledge_gradient.py` · acquisition
**Sources:** Negoescu, Frazier & Powell, INFORMS JoC 2011; exact computation
from Frazier, Powell & Dayanik, INFORMS JoC 2009

### Why it's in the benchmark
Two reasons.

First, **the source problem is structurally almost identical to this one.** In
Negoescu et al., a candidate compound is a vector of categorical substituent
choices at several sites; you synthesize one, measure one noisy scalar, and the
budget is very small. They put a Bayesian linear model on the one-hot encoding
and choose what to measure by Knowledge Gradient. That is this problem with
different nouns.

Second, **it fills the one acquisition family the suite was missing.** Thompson
sampling, Expected Improvement and UCB all score a candidate by how good it
might turn out to be. KG scores it by something different: *how much better the
final recommendation will be after measuring it.* It is the only genuinely
decision-theoretic, one-step-optimal criterion here — and the only acquisition
that explicitly optimizes the simple-regret metric the harness actually reports
(§1.6).

### How it works
With belief `theta ~ N(m, S)`, measuring team `x` with noise variance `lambda`
updates *every* alternative's posterior mean along a single random direction:

```
mu'_i = a_i + b_i * Z,     Z ~ N(0, 1)
a_i   = phi_i' m
b_i   = phi_i' S phi_x / sqrt(lambda + phi_x' S phi_x)
```

So `KG(x) = E[max_i (a_i + b_i Z)] - max_i a_i` is the expectation of the
maximum of a set of **lines** in one scalar variable — which has an exact closed
form. Sort the lines by slope, discard those never on the upper envelope, and
sum over the envelope's breakpoints `c_i`:

```
KG(x) = sum_i (b_{i+1} - b_i) * f(-|c_i|)      f(z) = z*Phi(z) + phi(z)
```

No sampling and no approximation. The algorithm plays the team with the largest
KG.

The surrogate is deliberately the same second-order Bayesian linear model as
`bocs`, `linucb` and `purexp`, so the four isolate the acquisition rule alone.

### Caveats
KG is evaluated over a candidate **pool** of 40 rather than all ~10^5 teams —
rebuilt each round from the best observed team, the posterior-mean maximizer,
their single-arm neighbourhoods, and random teams. Scoring every team exactly
would need a 10^5-line envelope computation per candidate per round. Restricting
to a pool is standard practice for KGCB on large discrete sets, and the pool
always contains the incumbent, so the recommendation can never degrade for lack
of coverage.

---

## `purexp` — Pure exploration with a model-based recommendation
`pure_exploration.py` · acquisition (the null)
**Source:** Bubeck, Munos & Stoltz, ALT 2009

### Why it's in the benchmark
It is the control condition that separates the two halves of the problem, and
no other arm can supply the number.

The harness scores `predict_best()` — a simple-regret metric — but every other
algorithm allocates its 100 rounds so as to *earn reward*. Bubeck et al. proved
those two objectives trade off. This arm spends the entire budget on uniform
random teams and puts all of its intelligence into the recommendation, fitting
the same Bayesian linear surrogate `bocs` uses and reporting its posterior-mean
maximizer. The resulting bracket:

```
random                = uniform allocation + no model      (floor)
purexp                = uniform allocation + BOCS's model
bocs / linucb / kg    = adaptive allocation + BOCS's model
```

The gap from `random` to `purexp` is what the **surrogate** is worth. The gap
from `purexp` to the adaptive methods is what **adaptive allocation** is worth.
Neither number is recoverable from the existing arms, and if the second gap is
small, that is a substantive finding about this problem rather than a null
result.

### How it works
Round 1 plays `initial_bias`; every subsequent round plays a uniformly random
team, **regardless of what has been observed**. Observations feed the same
second-order Bayesian linear model (Sherman-Morrison updated, exact
re-inversion every 25 rounds). `predict_best()` maximizes its posterior mean by
greedy local search.

### Note on the original
Bubeck et al.'s concrete procedures (uniform sampling and UCB-E) assume finitely
many *unstructured* arms, which does not survive contact with a 10^5-team space
— with 100 rounds you cannot sample each team even once. What transfers is the
uniform-allocation principle and the separation of sampling from
recommendation, which is what is implemented.

---

## `bayesgap` — Gap-based best-arm identification
`bayes_gap.py` · fixed-budget best-arm identification
**Source:** Hoffman, Shahriari & de Freitas, AISTATS 2014 (arXiv:1303.6746)

### Why it's in the benchmark
It is the **only algorithm in the suite designed for the objective the harness
actually measures**. Everything else optimizes reward earned along the way and
is scored on its final recommendation as a side effect; BayesGap targets the
recommendation directly, under a fixed budget, with arms correlated through a
Bayesian model. If simple regret and cumulative regret really do trade off
here, this is the arm that should reveal it.

### How it works
For each candidate team `k`, form a posterior mean `mu_k` and an interval
`[L_k, U_k] = mu_k -/+ BETA * sd_k`. Define the **gap index**

```
B_k = max_{j != k} U_j  -  L_k
```

which upper-bounds how much better the true best arm could be than arm `k`.
Each round:

```
J = argmin_k B_k                 the arm we would currently recommend
j = argmax_{k != J} U_k          its most threatening challenger
play whichever of {J, j} has the larger posterior uncertainty
```

and recommend, at any point, the `J` that has achieved the smallest `B_J` so
far. Sampling is aimed squarely at resolving the ambiguity between the leader
and its closest rival, rather than at collecting reward.

`predict_best()` returns that tracked recommendation — not a fresh argmax —
because that is the quantity the algorithm's guarantee is about.

### Adaptations
- **Arms.** BayesGap assumes a fixed enumerated arm set; there are ~10^5 teams
  here. A growing candidate pool is maintained (seeded with 60 random teams plus
  `initial_bias`, capped at 400), and the posterior-mean maximizer is added each
  round if new. The pool tracks the promising region without enumerating the
  space, and always contains the incumbent.
- **Correlation.** Hoffman et al. use a GP; this uses the same second-order
  Bayesian linear model as `bocs`/`linucb`/`kg`, so this arm differs from those
  three in its *allocation rule* only.
- **Exploration width.** The paper adapts `BETA` to the budget via a hardness
  estimate; a fixed `BETA = 1.0` is used here.

---

## `cucb` — Combinatorial Upper Confidence Bound
`cucb.py` · combinatorial bandit
**Source:** Chen, Wang & Yuan, ICML 2013 (extended JMLR 2016)

### Why it's in the benchmark
This is the **formal framework for exactly this problem shape** — choose one
configuration out of a product space — and it is the family that *none* of the
five source papers (BOCS, COMBO, NeuralLinear, NeuralUCB, NeuralTS) engage with
at all. Adding it closes a genuine gap in the literature review.

It is also the principled, theoretically-analysed version of what DreamTeam
does by hand: per-dimension estimates updated from a joint reward. The
difference is that CUCB uses an optimism bonus where DreamTeam uses a
stickiness schedule and a switching budget.

### How it works
**The mapping.**

```
base arm  = one (bandit, arm) pair    -- one candidate for one role
super arm = a full team               -- one base arm per bandit
oracle    = argmax over super arms of the summed base-arm estimates
```

Because that objective is **separable across dimensions**, the oracle here is
exact and instant: independently take the best-scoring arm in each bandit.
CUCB's theory only assumes an (α, β)-approximation oracle, so an exact one is
the best case its guarantee allows.

**The algorithm.** Maintain an empirical mean and a play count for every base
arm. Each round form the optimistic estimate

```
mu_bar[d][a] = mu_hat[d][a] + sqrt( 3 * ln(t) / (2 * T[d][a]) )
```

and call the oracle on it. Round 1 plays `initial_bias`; the next `max(n_d)`
rounds sweep every base arm at least once, which CUCB requires before its
confidence radii are defined. `predict_best()` is the per-dimension argmax of
the empirical means, with no optimism bonus.

### Adaptation note — important
CUCB assumes **semi-bandit feedback**: the outcome of each played base arm is
observed individually. This environment returns only one joint scalar for the
whole team, so that reward is credited to every base arm in the played team.
This is a real departure from the paper's feedback model — and it is the same
credit-assignment shortcut DreamTeam makes, which is exactly what makes CUCB an
interesting comparison rather than merely an addition.

---

## `cts` — Combinatorial Thompson Sampling
`combinatorial_ts.py` · combinatorial bandit
**Source:** Wang & Chen, ICML 2018

### Why it's in the benchmark
The Thompson-sampling counterpart of CUCB, and — more importantly — **the
cleanest available control for DreamTeam**. DreamTeam also keeps a Beta
posterior per (bandit, arm) and also feeds it the joint reward. CTS is what that
design looks like when done by the book: an unbiased Bernoulli update instead of
the "+reward on success, +1000 to beta on failure" rule, and no stickiness
schedule or global switching budget. Any gap between `cts` and `dreamteam` is
attributable to DreamTeam's bespoke machinery.

### How it works
Keep a Beta(α, β) posterior for every base arm, starting Beta(1, 1). Each round
sample `theta[d][a] ~ Beta(alpha, beta)` independently and call the same exact
separable oracle — per-dimension argmax of the sampled values. Round 1 plays
`initial_bias`; no forced exploration sweep is needed, since the Beta(1,1) prior
handles cold starts. `predict_best()` is the per-dimension argmax of the
posterior means.

**Bounded rewards.** The Beta posterior is conjugate to Bernoulli outcomes, but
rewards here are continuous in [0, 1]. CTS uses the standard Agrawal & Goyal
device: on observing reward `r`, draw a single Bernoulli(r) and update `alpha`
on a success or `beta` on a failure. This keeps the posterior exactly conjugate
and is the construction the analysis assumes for [0,1]-valued rewards — not an
approximation.

Like CUCB, it inherits the semi-bandit feedback caveat: the joint reward is
credited to every base arm played.

---

## `casmopolitan` — Trust-region categorical Bayesian optimization
`casmopolitan.py` · surrogate BO
**Source:** Wan, Nguyen, Ha, Ru, Lu & Osborne, ICML 2021 (arXiv:2102.07188)

### Why it's in the benchmark
It is the most recent method in the suite targeting this exact problem class —
the successor to `cocabo` in a line that cites COMBO — and it makes a
structurally different bet from every other arm.

Every other surrogate here fits one global model and asks it to rank all ~10⁵
teams. Casmopolitan's claim is that this is the wrong thing to ask of a model
trained on 100 noisy points: a global fit over a combinatorial space is
stretched too thin to be trusted far from the data, and its acquisition
maximizer will happily wander into a region where the model is confidently
wrong. Under a 100-round budget that is a live hypothesis, not a refinement.

### How it works
**Trust region.** A centre (the best team observed so far) and an integer
Hamming radius `L`. Candidates are restricted to teams within `L` role-changes
of the centre. At `L = D` the restriction vanishes and the method is ordinary
global EI — which is exactly what makes it testable.

**Adaptation.** Each observation improving the incumbent counts as a success,
each that does not as a failure. Three consecutive successes double `L`
(capped at `D`); ten failures halve it. This is TuRBO's rule adapted to
discrete spaces: the ball tracks how much of the space the model is currently
earning the right to search.

**Restart.** When `L` falls below 1 the local search is judged converged: `L`
resets and the centre jumps to a fresh random team. All observations are
retained, so the GP keeps improving across restarts — only the locality resets.

**Surrogate.** The exponentiated categorical-overlap kernel, which for a flat
categorical space is the same family as `gp_onehot`'s, with the same
hyperparameter fit and the same local-search optimizer. That is deliberate:
`gp_onehot` and `casmopolitan` then differ in exactly one thing — whether the
acquisition is maximized globally or inside an adaptive trust region — so the
pair isolates the paper's actual contribution rather than confounding it with
a kernel change.

`predict_best()` maximizes the posterior mean **globally**: the trust region
governs where to sample, not what to recommend.

### Caveats
The paper picks restart locations using a global model; this picks uniformly at
random (the GP still sees all data, so nothing is forgotten — only the restart
is uninformed). The paper also handles mixed categorical/continuous spaces with
separate trust regions; this problem is purely categorical, so only the
categorical region exists here.

---

## `glm_fpl` — Follow-the-perturbed-leader over a generalized linear model
`glm_fpl.py` · GLM
**Source:** Kveton, Zaheer, Szepesvári, Li, Ghavamzadeh & Boutilier, AISTATS
2020 (arXiv:1906.08947)

### Why it's in the benchmark
Two reasons, and the first is about correctness rather than performance.

**The likelihood actually matches this environment.** Every other model in the
suite assumes homoscedastic Gaussian noise — one variance, the same everywhere.
This environment does not work that way. Its reward is bounded in [0, 1] and its
noise is multiplicative, `std = noise_level × p`, so the variance is *coupled to
the mean* and vanishes as the reward goes to zero. That is precisely the
mean–variance relationship a logistic-link GLM encodes natively, and it is a
structural mis-specification in `bocs`, `linucb`, `kg`, `purexp`, `sts` and
every GP arm. This arm measures what that mis-specification costs.

**It explores by perturbation rather than by a posterior.** No confidence width
to set, no covariance matrix, one knob. At 100 rounds — where every posterior
here is thin and every theory-prescribed exploration constant needs shrinking by
hand — a method whose exploration comes from resampling the data rather than
from a fitted posterior is a genuinely different bet, and it degrades gracefully
under model mis-specification, which the point above says is guaranteed.

### How it works
Features are the same intercept + first-order + pairwise map as `bocs`, so the
model class is comparable. Each round:

1. Perturb every observed reward: `ỹᵢ = clip(yᵢ + σ·zᵢ)`, `z ~ N(0,1)`.
2. Refit the logistic GLM to the perturbed rewards by IRLS.
3. Play the team maximizing the fitted mean — greedy. All the exploration lives
   in step 1.

`predict_best()` refits on the *unperturbed* rewards, so no exploration noise
enters the recommendation.

### Why it's cheap despite ~470 features
IRLS would normally need a P×P solve per Newton step. But there are at most 100
observations, so the fit is done in the dual by the Woodbury identity, making
every solve n×n. The Gram matrix has a closed form: two teams sharing `m` of
their `D` dimensions share the intercept, `m` first-order indicators, and every
pairwise indicator whose *both* dimensions match, so

```
(X X')ᵢⱼ = 1 + m + m(m-1)/2
```

with no feature vector ever materialized. Predictions are likewise kernelized.
The whole algorithm touches only n×n matrices. (This dual formulation is checked
against an explicit primal IRLS fit in the test suite — they agree to 1e-6.)

---

## `sts` — Satisficing Thompson Sampling
`satisficing_ts.py` · acquisition (on the `bocs` surrogate)
**Source:** Russo, Tse & Van Roy, arXiv:1704.09028, 2017

### Why it's in the benchmark
This paper is about our constraint, not merely compatible with it.

Thompson sampling is asymptotically optimal, and that is exactly the problem: it
keeps spending pulls to distinguish actions whose values are nearly identical,
because given enough time that separation pays off. With 100 rounds against
~10⁵ teams there is no "enough time" — a method insisting on identifying the
single best team never leaves the exploration phase. Russo, Tse & Van Roy
formalize this: at short horizons, targeting the exact optimum is the wrong
objective. Satisficing TS instead targets an action within ε of optimal, and
among those prefers the one cheapest to adopt.

Because the surrogate is identical to `bocs`, the pair `bocs` (plain TS) vs
`sts` (satisficing TS) isolates the satisficing rule alone.

### How it works
Each round, after the usual Thompson draw from the second-order Bayesian linear
posterior:

1. Maximize the sampled model by multi-start local search → `S*`, the sampled
   optimum's value.
2. Assemble the *reachable* candidates: the team currently in place, its
   single-role neighbourhood, and the local-search trajectory.
3. Keep those scoring at least `S* − ε` — the satisficing set.
4. Play the member requiring the **fewest role changes**. If the set is empty,
   play the sampled optimum: no good-enough nearby team exists, so there is
   nothing to satisfice with.

`predict_best()` maximizes the posterior mean, as in `bocs` — satisficing
governs what is *played*, not what is recommended.

### Why "cheapest to adopt" means what it does here
The paper's cost term is problem-specific. In team composition it has an obvious
reading: the cost of adopting a team is the number of roles you must change to
reach it. That makes this arm doubly relevant — it is a short-horizon correction
to Thompson sampling, *and* it is the only algorithm in the suite besides
DreamTeam that treats changing the team as itself costly. But it arrives there
from decision theory rather than from a hand-set schedule and switching budget,
so `sts` vs `dreamteam` is a comparison of two different answers to the same
question. (Measured: `sts` makes about 10% fewer role changes than `bocs` over
100 rounds on the same surrogate.)

---

## `gp_nei` — Noisy Expected Improvement
`gp_acquisitions.py` · acquisition (on the shared one-hot GP)
**Source:** Letham, Karrer, Ottoni & Bakshy, *Bayesian Analysis* 14(2), 2019

### Why it's in the benchmark
This is closer to a correctness fix than a new competitor.

Standard EI compares candidates against `max(observed y)`. That is correct only
when observations are noise-free. Here the noise is multiplicative, so the best
observed reward is systematically an **overestimate** — it is the maximum of a
set of noisy draws, and the noisiest draws come from exactly the good teams. EI
against that inflated incumbent under-explores. `combo`, `smac` and `gp_onehot`
all use the noise-free formula, so this arm measures what the shortcut costs.

### How it works
Draw S = 24 joint samples of the *true* function values at the observed points
from the GP posterior — each sample is one plausible account of what actually
happened. Take each sample's maximum as that scenario's incumbent, compute EI
against it, and average across scenarios. The incumbent is thereby integrated
out rather than plugged in.

### Caveats
Letham et al. re-condition the GP on each sampled set of noiseless values before
computing EI, giving a different posterior per scenario. This implementation
keeps the single noise-conditioned posterior and marginalizes only the
*incumbent* — capturing the correction this environment actually needs at 1/S of
the cost. The full version would additionally sharpen the posterior.

---

## `gp_ucb` — IGP-UCB
`gp_acquisitions.py` · acquisition (on the shared one-hot GP)
**Source:** Chowdhury & Gopalan, ICML 2017 (arXiv:1704.00445)

### Why it's in the benchmark
It supplies the optimism member of the GP acquisition grid, so the `linucb`
result on the linear surrogate can be checked against a second model class.
Chowdhury & Gopalan tightened the confidence width of the original GP-UCB
(Srinivas et al., 2010) via a self-normalized concentration bound, which is why
this is the version implemented rather than the 2010 one.

### How it works
`score(x) = μ(x) + βₜ·σ(x)`, with the finite-domain schedule

```
βₜ = sqrt( 2 · log( |D| · t² · π² / (6δ) ) )
```

where `|D|` is genuinely computable here — it is the number of teams — so the
schedule is used as written rather than replaced by a constant.

### Caveats
One documented deviation: `βₜ` is multiplied by `BETA_SCALE = 0.2`. The
theory-prescribed width is famously over-conservative; NeuralTS's own
experiments needed an exploration coefficient two to five orders of magnitude
below the theoretical value. At `|D| ~ 10⁵` the unscaled `βₜ` exceeds 6, which
would swamp a reward living in [0, 1]. The scale factor makes that adjustment
explicit and tunable rather than hidden.

---

## `gp_ts` — GP Thompson Sampling
`gp_acquisitions.py` · acquisition (on the shared one-hot GP)
**Source:** Chowdhury & Gopalan, ICML 2017 (arXiv:1704.00445)

### Why it's in the benchmark
The posterior-sampling member of the GP acquisition grid — the GP counterpart of
what `bocs` does with a linear model. With `gp_onehot` (EI), `gp_nei` (noisy EI)
and `gp_ucb` (optimism) it completes a four-way acquisition comparison on one
fixed surrogate.

### How it works
Draw one sample of the objective from the GP posterior and play its maximizer.

A GP sample is a *function*, and drawing one consistently over a 10⁵-team space
is not possible directly. Instead an exact joint sample is drawn over a
candidate pool — the incumbent, the posterior-mean maximizer, their single-arm
neighbourhoods, and random teams. Within that pool the sample is exact: a joint
draw from the posterior covariance, **not** independent marginal draws, which
would be a different and far more erratic algorithm. The pool always contains
the incumbent, so the play can never be worse than a local move away from it.

The variance inflation factor is left at 1. Agrawal & Goyal's analysis
prescribes inflating the posterior for linear TS, but Abeille & Lazaric (2017)
showed the guarantee tolerates a wide family of sampling distributions, and
every practical study in this bibliography shrinks rather than inflates it at
short horizons.

---

# Appendix

## A.1 Cost at the full sweep

Measured on one run at 9 bandits × 100 rounds and extrapolated to 1080 runs.
Indicative only — hardware and arm counts move these.

| tier | algorithms | approx. full sweep |
|---|---|---|
| free | `random`, `sa`, `ols`, `regevo`, `cucb`, `cts` | < 0.5 min |
| cheap | `dreamteam`, `dreamteam_orig`, `cocabo`, `purexp` | 0.2 – 2 min |
| moderate | `casmopolitan`, `glm_fpl`, `bayesgap`, `gp_ts`, `gp_onehot`, `gp_ucb`, `neurallinear` | 3 – 6 min |
| heavy | `kg`, `bocs_hs`, `bootnn`, `gp_nei`, `combo`, `combo_slice`, `linucb` | 8 – 18 min |
| heaviest | `bocs`, `sts`, `smac` | ~34 min each |

Running all 27 is on the order of four hours.

## A.2 Shared conventions at a glance

| convention | value | applies to |
|---|---|---|
| Round 1 | plays `initial_bias` | all |
| Random initialization | `N_INIT = 10` | all model-based |
| Discrete maximization | multi-start greedy local search, `N_RESTARTS = 4` | all model-based |
| Model-free recommendation | best *average* observed reward | `sa`, `ols`, `regevo` |
| BOCS feature map | intercept + first-order + pairwise, ~470 features | `bocs`, `bocs_hs`, `linucb`, `kg`, `purexp`, `sts`, `bayesgap`, `glm_fpl` |
| Prior scales | `G_FIRST = 1.0`, `G_SECOND = 0.25` | the same, except `glm_fpl` (ridge) |
| Noise prior | `InvGamma(A0=2.0, B0=0.05)` | the same, except `glm_fpl` (logistic likelihood) |
| One-hot GP kernel | `exp(-hamming / rho)`, profile-likelihood grid fit | `gp_onehot`, `gp_nei`, `gp_ucb`, `gp_ts`, `casmopolitan` |

## A.3 Adding a new algorithm

1. Create `algorithms/my_algo.py` with a class implementing
   `RecommendationAlgorithm` (`choose`, `update`, `predict_best`) and a unique
   `name`.
2. Import it in `__init__.py` and add it to `ALGORITHMS`.
3. Run it: `python run_experiment.py --bandits 9 --rounds 100 --algorithm myalgo`

Conventions worth following, so the new arm is comparable to the rest: play
`initial_bias` on round 1; use `N_INIT = 10` random rounds if the method is
model-based; use the shared local-search optimizer; and write a module
docstring that cites the source and lists every simplification relative to it.

## A.4 Source papers

The five the suite is built around:

- Baptista & Poloczek, *Bayesian Optimization of Combinatorial Structures*,
  ICML 2018 — arXiv:1806.08838
- Oh, Tomczak, Gavves & Welling, *Combinatorial Bayesian Optimization using the
  Graph Cartesian Product*, NeurIPS 2019 — arXiv:1902.00448
- Riquelme, Tucker & Snoek, *Deep Bayesian Bandits Showdown*, ICLR 2018 —
  arXiv:1802.09127
- Zhou, Li & Gu, *Neural Contextual Bandits with UCB-based Exploration*,
  ICML 2020 — arXiv:1911.04462
- Zhang, Zhou, Li & Gu, *Neural Thompson Sampling*, ICLR 2021 — arXiv:2010.00827

Plus the algorithm the project is about:

- Zhou, Valentine & Bernstein, *In Search of the Dream Team*, CHI 2018 —
  DOI 10.1145/3173574.3173682

Every other citation appears in the module docstring of the plug-in that
implements it.
