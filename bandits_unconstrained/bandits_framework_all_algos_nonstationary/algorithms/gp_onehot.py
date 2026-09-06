"""Vanilla GP-BO over a one-hot encoding -- the "naive GP" baseline.

Baseline in: Baptista & Poloczek, "Bayesian Optimization of Combinatorial
Structures" (ICML 2018), arXiv:1806.08838, where it appears as "GP-BO with
one-hot encoding / expected improvement" -- the obvious thing to try before
building a combinatorial-specific surrogate.

This is deliberately the *unspecialized* GP: a standard squared-exponential
kernel applied to the one-hot encoded team, with no diffusion kernel (COMBO)
and no interaction-feature expansion (BOCS). It exists to answer the question
a reviewer will ask: how much does the combinatorial-specific machinery
actually buy over textbook GP-BO?

Kernel note (why there is no explicit one-hot matrix here):
For one-hot encoded categorical vectors the squared distance between two
teams is exactly twice their Hamming distance,

    ||onehot(x) - onehot(x')||^2 = 2 * hamming(x, x')

so the squared-exponential kernel collapses to the closed form

    k(x, x') = sf2 * exp(-hamming(x, x') / rho)

which is what is computed below. This is an exact algebraic reduction of the
one-hot SE kernel, not an approximation of it -- it just avoids materializing
the one-hot matrix and the pairwise-distance computation.

Faithful pieces:
  - Isotropic SE kernel over the one-hot encoding, Gaussian observation noise.
  - Exact GP posterior via Cholesky.
  - Expected Improvement acquisition, maximized by multi-start greedy local
    search over single-arm changes (same optimizer as the other plug-ins, so
    acquisition-optimizer differences don't confound the comparison).

Simplifications vs. a reference implementation:
  - Hyperparameters (lengthscale rho, noise-to-signal ratio eta) are fit on a
    small grid by maximizing the *concentrated* log marginal likelihood, with
    the signal variance sf2 solved for in closed form at each grid point
    (profile likelihood). This is a deterministic, dependency-free stand-in
    for a gradient-based multi-start MLL optimizer -- there are only two free
    hyperparameters, so a grid is adequate and cannot fail to converge.
  - Hyperparameters are re-tuned every REFIT_EVERY rounds; the posterior
    itself is refreshed every round.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (BO random initialization).
  - predict_best() maximizes the GP posterior mean by local search; before
    enough data exists it returns the best observed team.
"""

import math
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z])


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


class GPOneHotAlgorithm(RecommendationAlgorithm):
    name = "gp_onehot"

    N_INIT = 10          # random-exploration rounds before the GP takes over
    REFIT_EVERY = 10     # re-tune hyperparameters every k rounds
    N_RESTARTS = 4       # local-search restarts for acquisition maximization
    JITTER = 1e-8

    #: lengthscale grid (in units of Hamming distance)
    RHO_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    #: noise-to-signal ratio grid
    ETA_GRID = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

        self.rho = 3.0
        self.eta = 0.1
        self.sf2 = 0.25

        self._chol: Optional[np.ndarray] = None   # Cholesky of (R + eta*I)
        self._alpha: Optional[np.ndarray] = None  # (R + eta*I)^-1 (y - ymean)
        self._ymean = 0.0
        self._fitted_n = 0
        self._rounds_since_tune = 0

    # -- Kernel ---------------------------------------------------------------

    @staticmethod
    def _hamming(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A[:, None, :] != B[None, :, :]).sum(axis=2).astype(float)

    # -- Fitting --------------------------------------------------------------

    def _tune_hypers(self, H: np.ndarray, yc: np.ndarray) -> None:
        """Grid search on the concentrated log marginal likelihood.

        For K = sf2 * (R + eta*I) the MLL-optimal signal variance has the
        closed form sf2 = yc' (R + eta*I)^-1 yc / n, so only (rho, eta) need
        searching.
        """
        n = len(yc)
        best = None
        for rho in self.RHO_GRID:
            R = np.exp(-H / rho)
            for eta in self.ETA_GRID:
                M = R + (eta + self.JITTER) * np.eye(n)
                try:
                    L = np.linalg.cholesky(M)
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
                quad = float(yc @ alpha)
                if quad <= 1e-12:
                    continue
                sf2 = quad / n
                nll = 0.5 * n * math.log(sf2) + float(np.log(np.diag(L)).sum()) + 0.5 * n
                if best is None or nll < best[0]:
                    best = (nll, rho, eta, sf2)
        if best is not None:
            _, self.rho, self.eta, self.sf2 = best

    def _refresh(self, tune: bool) -> None:
        X = np.array(self.X_teams)
        y = np.array(self.y)
        n = len(y)
        self._ymean = float(y.mean())
        yc = y - self._ymean

        H = self._hamming(X, X)
        if tune and n >= 5:
            self._tune_hypers(H, yc)

        M = np.exp(-H / self.rho) + (self.eta + self.JITTER) * np.eye(n)
        try:
            self._chol = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            M[np.diag_indices_from(M)] += 1e-4
            self._chol = np.linalg.cholesky(M)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, yc))
        self._fitted_n = n

    def _posterior(self, teams: np.ndarray):
        """Posterior mean and std for teams: (m, D) -> (mu[m], sd[m])."""
        X = np.array(self.X_teams)
        Hs = self._hamming(np.asarray(teams), X)
        Rs = np.exp(-Hs / self.rho)                       # (m, n) correlations
        mu = self._ymean + Rs @ self._alpha               # sf2 cancels here
        v = np.linalg.solve(self._chol, Rs.T)             # (n, m)
        var = self.sf2 * (1.0 - np.einsum("ij,ij->j", v, v))
        sd = np.sqrt(np.maximum(var, 1e-12))
        return mu, sd

    # -- Acquisition: Expected Improvement + local search ----------------------

    @staticmethod
    def _ei(mu, sd, best):
        z = (mu - best) / sd
        return (mu - best) * _norm_cdf(z) + sd * _norm_pdf(z)

    def _neighbors(self, team) -> np.ndarray:
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return np.array(neigh)

    def _local_search(self, score_fn, start: List[int]):
        current = list(start)
        current_score = float(score_fn(np.array([current]))[0])
        while True:
            neigh = self._neighbors(current)
            if not len(neigh):
                return current, current_score
            scores = score_fn(neigh)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current = [int(v) for v in neigh[j]]
                current_score = float(scores[j])
            else:
                return current, current_score

    def _maximize(self, score_fn) -> List[int]:
        starts = [list(self.X_teams[int(np.argmax(self.y))])]
        for _ in range(self.N_RESTARTS - 1):
            starts.append([int(np.random.randint(k)) for k in self.n])
        best_team, best_score = None, -np.inf
        for s in starts:
            team, score = self._local_search(score_fn, s)
            if score > best_score:
                best_team, best_score = team, score
        return best_team

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT or self._chol is None:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            best_y = float(np.max(self.y))

            def ei_score(teams):
                mu, sd = self._posterior(teams)
                return self._ei(mu, sd, best_y)

            self.last_choice = self._maximize(ei_score)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
        if len(self.y) >= self.N_INIT:
            self._rounds_since_tune += 1
            tune = (self._chol is None) or (self._rounds_since_tune >= self.REFIT_EVERY)
            if tune:
                self._rounds_since_tune = 0
            self._refresh(tune=tune)

    def predict_best(self) -> List[int]:
        if self._chol is None or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))

        def mean_score(teams):
            mu, _ = self._posterior(teams)
            return mu

        team, _ = self._local_search(mean_score, list(self.X_teams[int(np.argmax(self.y))]))
        return team
