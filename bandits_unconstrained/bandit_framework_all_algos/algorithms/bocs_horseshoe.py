"""BOCS with the paper's actual horseshoe prior (Gibbs-sampled).

Base method: Baptista & Poloczek, "Bayesian Optimization of Combinatorial
Structures" (ICML 2018), arXiv:1806.08838.
Sampler: Makalic & Schmidt, "A Simple Sampler for the Horseshoe Estimator"
(IEEE Signal Processing Letters 23(1), 2016).
Fast p >> n sampling step: Bhattacharya, Chakraborty & Mallick, "Fast Sampling
with Gaussian Scale Mixture Priors in High-Dimensional Regression"
(Biometrika, 2016).
Prior itself: Carvalho, Polson & Scott, "The Horseshoe Estimator for Sparse
Signals" (Biometrika 97(2), 2010).

Why this file exists separately from bocs.py:
    Our `bocs.py` substitutes grouped ridge shrinkage for the paper's
    horseshoe prior -- a simplification our own comparison report lists as a
    fairness caveat. BOCS's appendix F.3 is the evidence that this matters:
    the horseshoe's advantage over standard Bayesian linear regression is
    largest *specifically when N is small*, which is exactly our 100-round
    regime.

    Rather than editing bocs.py and silently changing published results, this
    is a separate arm. Running `bocs` against `bocs_hs` turns the
    simplification into an explicit, measurable experimental factor and lets
    the paper report what the shortcut actually cost.

The horseshoe hierarchy (Makalic & Schmidt's auxiliary-variable form, which
turns every conditional into an inverse-gamma and so needs no Metropolis step):

    beta_j | lambda_j, tau, sigma^2 ~ N(0, lambda_j^2 tau^2 sigma^2)
    lambda_j^2 | nu_j              ~ InvGamma(1/2, 1/nu_j)
    nu_j                           ~ InvGamma(1/2, 1)
    tau^2 | xi                     ~ InvGamma(1/2, 1/xi)
    xi                             ~ InvGamma(1/2, 1)
    sigma^2                        ~ InvGamma(a0, b0)

The heavy tail on lambda_j lets a genuinely important interaction escape
shrinkage entirely, while the mass near zero crushes the hundreds of
irrelevant pairwise terms -- which grouped ridge, applying one shrinkage
scale to every pairwise weight, cannot do.

Implementation notes:
  - The beta update is the expensive one: p ~= 470 features but n <= 100
    observations, so the textbook O(p^3) Cholesky is wasteful. We use
    Bhattacharya et al.'s exact algorithm, which draws the same Gaussian in
    O(n^2 p) via an n x n solve. Not an approximation -- the same distribution,
    computed in the cheaper direction.
  - The Gibbs chain is *persistent across rounds*: it is burned in once when
    enough data exists, then advanced a few sweeps per round as new
    observations arrive. Since consecutive rounds' posteriors differ by one
    data point, the previous state is an excellent warm start. This is what
    makes a genuine MCMC posterior affordable inside the full experiment
    sweep; the paper re-runs its sampler from scratch each iteration.
  - Thompson sampling uses the current beta draw directly -- an exact
    posterior sample, which is the whole point of BOCS's exploration (its
    appendix E shows that substituting a point estimate collapses the
    algorithm into pure exploitation).
  - predict_best() maximizes a running posterior-mean estimate accumulated
    from the draws (Rao-Blackwellized), not a single noisy draw.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (random initialization).
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _inv_gamma(shape: float, scale) -> np.ndarray:
    """Draw from InvGamma(shape, scale) with density ~ x^-(shape+1) e^{-scale/x}."""
    g = np.random.gamma(shape, 1.0, size=np.shape(scale))
    return np.asarray(scale) / np.maximum(g, 1e-12)


class BocsHorseshoe(RecommendationAlgorithm):
    name = "bocs_hs"

    N_INIT = 10            # random-exploration rounds before the surrogate takes over
    BURN_IN_SWEEPS = 40    # Gibbs sweeps the first time the chain is started
    SWEEPS_PER_ROUND = 4   # sweeps to advance the persistent chain each round
    A0 = 2.0               # Inverse-Gamma prior on noise variance
    B0 = 0.05
    N_RESTARTS = 4         # local-search restarts for acquisition maximization

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        # -- Feature index layout (identical to bocs.py) ----------------------
        self.first_offset = [0] * self.D
        idx = 1
        for d in range(self.D):
            self.first_offset[d] = idx
            idx += self.n[d]
        self.pair_offset = {}
        for d in range(self.D):
            for e in range(d + 1, self.D):
                self.pair_offset[(d, e)] = idx
                idx += self.n[d] * self.n[e]
        self.P = idx

        # dense design matrix, grown row by row
        self._X = np.zeros((max(total_rounds, 1), self.P))
        self.X_teams: List[List[int]] = []
        self.y: List[float] = []

        # horseshoe state (persistent Gibbs chain)
        self.beta = np.zeros(self.P)
        self.lam2 = np.ones(self.P)
        self.nu = np.ones(self.P)
        self.tau2 = 1.0
        self.xi = 1.0
        self.sigma2 = 0.1
        self._started = False
        self._mean_beta = np.zeros(self.P)
        self._mean_count = 0

        self.last_choice = list(initial_bias)

    # -- Features ---------------------------------------------------------------

    def _active_indices(self, team) -> np.ndarray:
        idxs = [0]
        for d in range(self.D):
            idxs.append(self.first_offset[d] + team[d])
        for d in range(self.D):
            for e in range(d + 1, self.D):
                idxs.append(self.pair_offset[(d, e)] + team[d] * self.n[e] + team[e])
        return np.array(idxs)

    def _score_teams(self, teams, w) -> np.ndarray:
        return np.array([w[self._active_indices(t)].sum() for t in teams])

    # -- Gibbs sampler ----------------------------------------------------------

    def _sample_beta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bhattacharya et al. (2016) exact sampler for N(A^-1 X'y/s2, A^-1),
        A = X'X/s2 + D^-1, D = s2 * tau2 * diag(lam2). Costs an n x n solve."""
        n = len(y)
        sigma = np.sqrt(max(self.sigma2, 1e-12))
        d_vec = self.sigma2 * self.tau2 * self.lam2            # prior variances
        scaled = self.tau2 * self.lam2                          # = d_vec / sigma^2

        u = np.random.normal(0.0, np.sqrt(np.maximum(d_vec, 1e-300)))
        delta = np.random.normal(size=n)
        v = (X @ u) / sigma + delta                             # Phi u + delta

        # Phi D Phi' + I  =  X diag(scaled) X' + I   (the sigma factors cancel)
        M = (X * scaled) @ X.T
        M[np.diag_indices_from(M)] += 1.0
        alpha = y / sigma
        try:
            w = np.linalg.solve(M, alpha - v)
        except np.linalg.LinAlgError:
            M[np.diag_indices_from(M)] += 1e-6
            w = np.linalg.solve(M, alpha - v)
        return u + (d_vec * (X.T @ w)) / sigma

    def _gibbs_sweep(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape

        self.beta = self._sample_beta(X, y)

        resid = y - X @ self.beta
        prior_quad = float(np.sum(self.beta ** 2 / np.maximum(self.tau2 * self.lam2, 1e-300)))
        self.sigma2 = float(_inv_gamma(self.A0 + 0.5 * (n + p),
                                       self.B0 + 0.5 * (float(resid @ resid) + prior_quad)))
        self.sigma2 = max(self.sigma2, 1e-10)

        b2 = self.beta ** 2
        self.lam2 = _inv_gamma(1.0, 1.0 / self.nu
                               + b2 / (2.0 * self.tau2 * self.sigma2))
        self.lam2 = np.maximum(self.lam2, 1e-12)
        self.nu = _inv_gamma(1.0, 1.0 + 1.0 / self.lam2)

        self.tau2 = float(_inv_gamma(0.5 * (p + 1),
                                     1.0 / self.xi
                                     + float(np.sum(b2 / self.lam2)) / (2.0 * self.sigma2)))
        self.tau2 = max(self.tau2, 1e-12)
        self.xi = float(_inv_gamma(1.0, 1.0 + 1.0 / self.tau2))

    def _advance_chain(self) -> None:
        n = len(self.y)
        X = self._X[:n]
        y = np.array(self.y)
        sweeps = self.SWEEPS_PER_ROUND
        if not self._started:
            self.sigma2 = max(float(np.var(y)), 1e-3)
            sweeps = self.BURN_IN_SWEEPS
            self._started = True
        for _ in range(sweeps):
            self._gibbs_sweep(X, y)
            self._mean_beta += self.beta
            self._mean_count += 1

    # -- Discrete maximization (multi-start greedy local search) ---------------

    def _neighbors(self, team):
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return neigh

    def _local_search(self, w, start):
        current = list(start)
        current_score = float(self._score_teams([current], w)[0])
        while True:
            neigh = self._neighbors(current)
            if not neigh:
                return current, current_score
            scores = self._score_teams(neigh, w)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current, current_score

    def _maximize(self, w):
        starts = [list(self.X_teams[int(np.argmax(self.y))])]
        for _ in range(self.N_RESTARTS - 1):
            starts.append([int(np.random.randint(k)) for k in self.n])
        best_team, best_score = None, -np.inf
        for s in starts:
            team, score = self._local_search(w, s)
            if score > best_score:
                best_team, best_score = team, score
        return best_team

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            self._advance_chain()
            # Thompson sampling on an exact posterior draw
            self.last_choice = self._maximize(self.beta)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        idxs = self._active_indices(arms_chosen)
        row = len(self.y)
        if row < self._X.shape[0]:
            self._X[row, idxs] = 1.0
        else:                                   # ran past the preallocation
            extra = np.zeros((1, self.P))
            extra[0, idxs] = 1.0
            self._X = np.vstack([self._X, extra])
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

    def predict_best(self) -> List[int]:
        if not self._started or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        w = self._mean_beta / max(self._mean_count, 1)
        team, _ = self._local_search(w, list(self.X_teams[int(np.argmax(self.y))]))
        return team
