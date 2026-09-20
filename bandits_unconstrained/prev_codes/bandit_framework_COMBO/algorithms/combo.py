"""COMBO-style algorithm plug-in.

Based on: Oh, Gavves, Welling, "Combinatorial Bayesian Optimization using
the Graph Cartesian Product" (NeurIPS 2019), arXiv:1902.00448.

COMBO models the objective with a Gaussian Process whose kernel is a
diffusion kernel on the combinatorial graph formed by the graph Cartesian
product of one sub-graph per variable. For purely categorical variables
(our case: each bandit = one categorical variable), each sub-graph is a
complete graph K_n, whose Laplacian eigenstructure gives the ARD diffusion
kernel in closed form. For a variable with n categories and diffusion
parameter beta >= 0:

    k(x, x') = (1 + (n-1)*e^(-n*beta)) / n     if x == x'
    k(x, x') = (1 - e^(-n*beta))       / n     if x != x'

The full kernel is the product over variables (Cartesian product =>
kernel factorizes), scaled by a signal variance. This is exactly COMBO's
kernel for categorical spaces; no graph Fourier transform is needed
because the complete-graph eigendecomposition is analytic.

Faithful pieces:
  - ARD diffusion kernel on the product of complete graphs (per-variable beta).
  - GP surrogate with Gaussian noise.
  - Expected Improvement acquisition, maximized by greedy local search over
    single-variable changes with multi-starts (COMBO also uses multi-start
    local search for acquisition optimization).

Simplifications vs. the paper (documented for honesty in comparisons):
  - Hyperparameters (betas, signal variance, noise) are fit by maximizing
    the marginal likelihood (type-II MAP with weak log-normal priors)
    instead of slice-sampled posteriors with a Horseshoe prior.
  - Hyperparameters are re-optimized every REFIT_EVERY rounds rather than
    every round, for speed inside the large experiment sweep.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (BO random initialization).
  - predict_best() maximizes the GP posterior mean by local search; before
    enough data exists it returns the best observed team.
"""

from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from .base import RecommendationAlgorithm, ProblemConfig


class ComboAlgorithm(RecommendationAlgorithm):
    name = "combo"

    N_INIT = 10          # random-exploration rounds before the GP takes over
    REFIT_EVERY = 10     # re-optimize hyperparameters every k rounds
    N_RESTARTS = 4       # local-search restarts for acquisition maximization
    JITTER = 1e-8

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.arm_counts = np.array(config.arm_counts)

        self.X: List[List[int]] = []   # observed teams
        self.y: List[float] = []       # observed rewards

        # log-hyperparameters: per-dimension log(beta), log signal var, log noise var
        self.log_beta = np.zeros(self.D)          # beta = 1
        self.log_sf2 = np.log(0.25)
        self.log_sn2 = np.log(0.05)

        self._chol: Optional[np.ndarray] = None   # cached Cholesky of K + sn2*I
        self._alpha: Optional[np.ndarray] = None  # cached (K + sn2*I)^-1 (y - mean)
        self._ymean: float = 0.0
        self._rounds_since_fit = 0
        self._fitted_n = 0                        # dataset size the cache was built on

    # ── Kernel ────────────────────────────────────────────────────────────────

    def _per_dim_kernels(self, log_beta):
        """Closed-form diffusion kernel values on K_n per dimension.
        Returns (k_same[D], k_diff[D]), each normalized so k_same accounts
        for the complete-graph spectrum {0, n (x n-1)}."""
        beta = np.exp(log_beta)
        n = self.arm_counts
        e = np.exp(-n * beta)
        k_same = (1 + (n - 1) * e) / n
        k_diff = (1 - e) / n
        return k_same, k_diff

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray, log_beta, log_sf2) -> np.ndarray:
        """K[i,j] = sf2 * prod_d k_d(A[i,d], B[j,d]). A: (m,D), B: (p,D)."""
        k_same, k_diff = self._per_dim_kernels(log_beta)
        # match[i,j,d] = 1 if A[i,d]==B[j,d]
        match = (A[:, None, :] == B[None, :, :])
        # per-dim factor then product over dims (in log space for stability)
        factors = np.where(match, k_same[None, None, :], k_diff[None, None, :])
        K = np.exp(log_sf2) * np.prod(factors, axis=2)
        return K

    # ── GP fitting ────────────────────────────────────────────────────────────

    def _neg_log_marginal_likelihood(self, theta, X, y):
        log_beta, log_sf2, log_sn2 = theta[:self.D], theta[self.D], theta[self.D + 1]
        n = len(y)
        K = self._kernel_matrix(X, X, log_beta, log_sf2)
        K[np.diag_indices_from(K)] += np.exp(log_sn2) + self.JITTER
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        yc = y - y.mean()
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        nll = 0.5 * yc @ alpha + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)
        # weak log-normal priors keep hyperparameters in a sane range
        nll += 0.05 * (log_beta ** 2).sum() + 0.05 * log_sf2 ** 2 + 0.05 * (log_sn2 + 3) ** 2
        return nll

    def _fit(self, optimize_hypers: bool):
        X = np.array(self.X)
        y = np.array(self.y)
        if optimize_hypers and len(y) >= 5:
            theta0 = np.concatenate([self.log_beta, [self.log_sf2, self.log_sn2]])
            res = minimize(self._neg_log_marginal_likelihood, theta0, args=(X, y),
                           method="L-BFGS-B",
                           bounds=[(-4, 4)] * self.D + [(-6, 3), (-8, 1)],
                           options={"maxiter": 12})
            if np.isfinite(res.fun):
                self.log_beta = res.x[:self.D]
                self.log_sf2 = res.x[self.D]
                self.log_sn2 = res.x[self.D + 1]

        K = self._kernel_matrix(X, X, self.log_beta, self.log_sf2)
        K[np.diag_indices_from(K)] += np.exp(self.log_sn2) + self.JITTER
        self._chol = np.linalg.cholesky(K)
        self._ymean = float(y.mean())
        yc = y - self._ymean
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, yc))
        self._fitted_n = len(y)

    def _posterior(self, teams: np.ndarray):
        """Posterior mean and std for teams: (m, D) -> (mu[m], sd[m])."""
        X = np.array(self.X)
        Ks = self._kernel_matrix(teams, X, self.log_beta, self.log_sf2)   # (m, n)
        mu = self._ymean + Ks @ self._alpha
        v = np.linalg.solve(self._chol, Ks.T)                              # (n, m)
        var = np.exp(self.log_sf2) - np.einsum("ij,ij->j", v, v)
        sd = np.sqrt(np.maximum(var, 1e-12))
        return mu, sd

    # ── Acquisition: Expected Improvement + local search ─────────────────────

    @staticmethod
    def _ei(mu, sd, best):
        from scipy.stats import norm
        z = (mu - best) / sd
        return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)

    def _neighbors(self, team: List[int]) -> np.ndarray:
        """All teams differing from `team` in exactly one dimension."""
        neigh = []
        for d in range(self.D):
            for a in range(self.arm_counts[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return np.array(neigh)

    def _local_search(self, score_fn, start: List[int]) -> (List[int], float):
        current = list(start)
        current_score = float(score_fn(np.array([current]))[0])
        while True:
            neigh = self._neighbors(current)
            scores = score_fn(neigh)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current = list(neigh[j])
                current_score = float(scores[j])
            else:
                return current, current_score

    def _maximize(self, score_fn) -> List[int]:
        """Multi-start greedy local search (as in COMBO's acquisition step)."""
        starts = [list(self.X[int(np.argmax(self.y))])]
        for _ in range(self.N_RESTARTS - 1):
            starts.append([int(np.random.randint(n)) for n in self.arm_counts])
        best_team, best_score = None, -np.inf
        for s in starts:
            team, score = self._local_search(score_fn, s)
            if score > best_score:
                best_team, best_score = team, score
        return best_team

    # ── RecommendationAlgorithm interface ─────────────────────────────────────

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(n)) for n in self.arm_counts]
        else:
            self._rounds_since_fit += 1
            refit = (self._chol is None) or (self._rounds_since_fit >= self.REFIT_EVERY)
            self._fit(optimize_hypers=refit)
            if refit:
                self._rounds_since_fit = 0
            best_y = float(np.max(self.y))

            def ei_score(teams):
                mu, sd = self._posterior(teams)
                return self._ei(mu, sd, best_y)

            self.last_choice = self._maximize(ei_score)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X.append(list(arms_chosen))
        self.y.append(float(reward))

    def predict_best(self) -> List[int]:
        if self._chol is None or len(self.y) <= self.N_INIT:
            return list(self.X[int(np.argmax(self.y))]) if self.y else list(self.initial_bias)
        if self._fitted_n != len(self.y):
            self._fit(optimize_hypers=False)   # refresh stale cache after update()

        def mean_score(teams):
            mu, _ = self._posterior(teams)
            return mu

        # single-start local search from the best observed team (cheap, called
        # every round for scoring; the acquisition step keeps multi-starts)
        team, _ = self._local_search(mean_score, list(self.X[int(np.argmax(self.y))]))
        return team
