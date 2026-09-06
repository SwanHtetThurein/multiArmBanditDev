"""Casmopolitan -- trust-region Bayesian optimization over categorical spaces.

Based on: Wan, Nguyen, Ha, Ru, Lu & Osborne, "Think Global and Act Local:
Bayesian Optimisation over High-Dimensional Categorical and Mixed Search
Spaces", ICML 2021, arXiv:2102.07188.

WHY IT IS IN THE BENCHMARK
    It is the most recent method in the suite that targets this exact problem
    class, and it is the direct successor to `cocabo` (2020) in a line that
    cites COMBO. More importantly it makes a different structural bet from
    every other arm here.

    Every surrogate method in this benchmark fits one global model and asks it
    to rank all ~10^5 teams. Casmopolitan's claim is that this is the wrong
    thing to ask of a model trained on 100 noisy points: a global fit over a
    combinatorial space is stretched too thin to be trusted far from the data,
    and its acquisition maximizer will happily wander to a region where the
    model is confidently wrong. Instead it maintains a *trust region* -- a
    Hamming ball around the current incumbent -- optimizes only inside it,
    grows the ball when it keeps finding improvements, shrinks it when it
    stops, and restarts elsewhere when the ball collapses.

    Under a 100-round budget over 10^5 teams that is a live hypothesis rather
    than a refinement, and it is testable here directly (see below).

A CONTROLLED COMPARISON, BY CONSTRUCTION
    The kernel used here is the exponentiated categorical-overlap kernel,
    which for a flat categorical space is the same family as `gp_onehot`'s:
    similarity decaying in Hamming distance, with a fitted lengthscale. The
    hyperparameter fit and the local-search optimizer are also identical.

    That is deliberate. It means `gp_onehot` and `casmopolitan` differ in
    exactly one thing -- whether the acquisition is maximized globally or
    restricted to a trust region that adapts and restarts -- so the pair
    isolates the paper's actual contribution rather than confounding it with a
    kernel change.

HOW IT WORKS
    Trust region. A centre (the best team observed so far) and an integer
    Hamming radius L. Candidates are restricted to teams within L role-changes
    of the centre; at L = D the restriction vanishes and the method is
    ordinary global EI.

    Adaptation. Each observation that improves the incumbent counts as a
    success, each that does not as a failure. SUCC_TOL consecutive successes
    double L (capped at D); FAIL_TOL failures halve it. This is the
    TuRBO-style rule Casmopolitan adapts to discrete spaces -- the ball tracks
    how much of the space the model is currently earning the right to search.

    Restart. When L falls below L_MIN the local search is judged converged: L
    resets to its initial value and the centre jumps to a fresh random team.
    All observations are retained, so the GP keeps improving across restarts;
    only the locality resets.

Simplifications vs. the paper (documented for honesty in comparisons):
  - Restart location. The paper fits a global model and uses it to choose
    where to restart; this picks a uniformly random team. The GP still sees
    all data, so nothing is forgotten -- only the restart is uninformed.
  - Mixed spaces. Casmopolitan handles categorical *and* continuous variables
    with separate trust regions. This problem is purely categorical, so only
    the categorical trust region exists here. Nothing is lost.
  - Interleaved acquisition. The paper's full version supports batch and
    multi-armed-bandit-assisted candidate generation; this uses the same
    multi-start greedy local search as every other arm in the suite, so the
    discrete optimizer is not a confound.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (BO random initialization).
  - predict_best() maximizes the GP posterior mean *globally*, not inside the
    trust region -- the trust region governs where to sample, not what to
    recommend.
"""

import math
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in np.atleast_1d(z)])


def _norm_pdf(z) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


class Casmopolitan(RecommendationAlgorithm):
    name = "casmopolitan"

    N_INIT = 10          # random-exploration rounds before the GP takes over
    REFIT_EVERY = 10     # re-tune hyperparameters every k rounds
    N_RESTARTS = 4       # local-search restarts for acquisition maximization
    JITTER = 1e-8

    SUCC_TOL = 3         # consecutive successes before the trust region grows
    FAIL_TOL = 10        # failures before it shrinks
    L_MIN = 1            # radius below which the region is judged converged

    RHO_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
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
        self._chol: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._ymean = 0.0
        self._rounds_since_tune = 0

        # trust region state
        self.L_INIT = max(1, self.D // 2)
        self.L = float(self.L_INIT)
        self.centre: List[int] = list(initial_bias)
        self._succ = 0
        self._fail = 0
        self._best_y = -np.inf
        self.n_restarts = 0

    # -- Kernel and fitting (same family and fit as gp_onehot.py) -------------

    @staticmethod
    def _hamming(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A[:, None, :] != B[None, :, :]).sum(axis=2).astype(float)

    def _tune_hypers(self, H: np.ndarray, yc: np.ndarray) -> None:
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

    def _posterior(self, teams):
        X = np.array(self.X_teams)
        Rs = np.exp(-self._hamming(np.asarray(teams), X) / self.rho)
        mu = self._ymean + Rs @ self._alpha
        v = np.linalg.solve(self._chol, Rs.T)
        var = self.sf2 * (1.0 - np.einsum("ij,ij->j", v, v))
        return mu, np.sqrt(np.maximum(var, 1e-12))

    @staticmethod
    def _ei(mu, sd, best):
        z = (mu - best) / sd
        return (mu - best) * _norm_cdf(z) + sd * _norm_pdf(z)

    # -- Trust region ----------------------------------------------------------

    @staticmethod
    def _dist(a, b) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def _in_region(self, team) -> bool:
        return self._dist(team, self.centre) <= int(round(self.L))

    def _update_region(self, reward: float) -> None:
        """TuRBO-style success/failure adaptation, then restart if collapsed."""
        if reward > self._best_y + 1e-9:
            self._succ += 1
            self._fail = 0
        else:
            self._fail += 1
            self._succ = 0
        self._best_y = max(self._best_y, reward)

        if self._succ >= self.SUCC_TOL:
            self.L = min(self.L * 2.0, float(self.D))
            self._succ = 0
        elif self._fail >= self.FAIL_TOL:
            self.L = self.L / 2.0
            self._fail = 0

        if self.L < self.L_MIN:
            # converged locally -- restart the region elsewhere, keep all data
            self.L = float(self.L_INIT)
            self.centre = [int(np.random.randint(k)) for k in self.n]
            self._succ = self._fail = 0
            self._best_y = -np.inf
            self.n_restarts += 1
        elif self.y:
            # otherwise re-centre on the incumbent
            self.centre = list(self.X_teams[int(np.argmax(self.y))])

    # -- Discrete maximization, restricted to the trust region ----------------

    def _neighbors(self, team, restrict: bool):
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    if (not restrict) or self._in_region(t):
                        neigh.append(t)
        return neigh

    def _local_search(self, score_fn, start, restrict: bool):
        current = list(start)
        current_score = float(score_fn([current])[0])
        while True:
            neigh = self._neighbors(current, restrict)
            if not neigh:
                return current, current_score
            scores = score_fn(neigh)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current, current_score

    def _random_in_region(self) -> List[int]:
        """A uniformly random team inside the Hamming ball around the centre."""
        radius = int(round(self.L))
        k = int(np.random.randint(0, radius + 1))
        team = list(self.centre)
        if k > 0:
            dims = list(np.random.permutation(self.D)[:k])
            for d in dims:
                options = [a for a in range(self.n[d]) if a != self.centre[d]]
                if options:
                    team[d] = int(np.random.choice(options))
        return team

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

            # multi-start local search, every start and every move inside the
            # trust region
            starts = [list(self.centre)]
            for _ in range(self.N_RESTARTS - 1):
                starts.append(self._random_in_region())
            best_team, best_score = None, -np.inf
            for s in starts:
                team, score = self._local_search(ei_score, s, restrict=True)
                if score > best_score:
                    best_team, best_score = team, score
            self.last_choice = list(best_team)
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
            self._update_region(float(reward))

    def predict_best(self) -> List[int]:
        if self._chol is None or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))

        def mean_score(teams):
            mu, _ = self._posterior(teams)
            return mu

        # recommendation is global: the trust region constrains sampling only
        team, _ = self._local_search(mean_score,
                                     list(self.X_teams[int(np.argmax(self.y))]),
                                     restrict=False)
        return team
