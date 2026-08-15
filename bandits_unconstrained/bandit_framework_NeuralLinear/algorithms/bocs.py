"""BOCS-style algorithm plug-in.

Based on: Baptista & Poloczek, "Bayesian Optimization of Combinatorial
Structures" (ICML 2018), arXiv:1806.08838.

BOCS models the black-box objective with a sparse Bayesian *linear* model
over low-order interaction features of the combinatorial variables, then
acquires via Thompson sampling: draw a weight vector from the posterior and
maximize the resulting surrogate (BOCS-SA uses simulated annealing; BOCS-SDP
a semidefinite relaxation).

Mapping to this framework (one categorical variable per bandit):
  Features phi(team) =
    intercept
    + first-order one-hots:  1[team_d = a]              for every (d, a)
    + second-order terms:    1[team_d = a and team_e = b] for every pair d<e
  Surrogate: f(team) ~ phi(team) . w, with a conjugate Normal-Inverse-Gamma
  Bayesian linear regression posterior over (w, noise variance).

The second-order terms are exactly the "this arm is only good with that
arm" interactions that per-dimension bandit posteriors cannot represent.

Faithful pieces:
  - Second-order sparse-ish Bayesian linear surrogate over combinatorial
    one-hot features.
  - Thompson-sampling acquisition: sample (sigma^2, w) from the exact
    conjugate posterior, maximize the sampled surrogate.
  - Discrete maximization by multi-start greedy local search over
    single-variable changes (stands in for BOCS's simulated annealing;
    same neighborhood structure, deterministic descent).

Simplifications vs. the paper (keep in mind for writeups):
  - Sparsity: the paper uses a heavy-tailed horseshoe prior on weights;
    here a Gaussian prior with a stronger shrinkage scale on second-order
    terms than first-order terms (grouped ridge) plays that role. Cheap,
    conjugate, and adequate at this problem size (~300 features).
  - Local search instead of simulated annealing / SDP relaxation.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (random initialization).
  - predict_best() maximizes the posterior-mean surrogate by local search
    from the best observed team.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class BocsAlgorithm(RecommendationAlgorithm):
    name = "bocs"

    N_INIT = 10          # random-exploration rounds before the surrogate takes over
    G_FIRST = 1.0        # prior variance scale (x sigma^2) for first-order weights
    G_SECOND = 0.25      # stronger shrinkage on pairwise weights (poor-man's sparsity)
    A0 = 2.0             # Inverse-Gamma prior on noise variance
    B0 = 0.05
    N_RESTARTS = 4       # local-search restarts for acquisition maximization
    JITTER = 1e-9

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        # ── Feature index layout ──────────────────────────────────────────────
        # [0] intercept | first-order blocks per dim | pairwise blocks per (d<e)
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

        # prior precision per weight (grouped ridge; note: scaled by 1/sigma^2
        # implicitly via the conjugate NIG formulation)
        prior_prec = np.empty(self.P)
        prior_prec[0] = 1.0 / self.G_FIRST
        for d in range(self.D):
            o = self.first_offset[d]
            prior_prec[o:o + self.n[d]] = 1.0 / self.G_FIRST
        for (d, e), o in self.pair_offset.items():
            prior_prec[o:o + self.n[d] * self.n[e]] = 1.0 / self.G_SECOND
        self.prior_prec = prior_prec

        # sufficient statistics (rank-1 updated each observation)
        self.XtX = np.zeros((self.P, self.P))
        self.Xty = np.zeros(self.P)
        self.yty = 0.0
        self.t = 0

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

    # ── Features ──────────────────────────────────────────────────────────────

    def _active_indices(self, team) -> np.ndarray:
        """Indices of the nonzero (=1) features for a team."""
        idxs = [0]
        for d in range(self.D):
            idxs.append(self.first_offset[d] + team[d])
        for d in range(self.D):
            for e in range(d + 1, self.D):
                idxs.append(self.pair_offset[(d, e)] + team[d] * self.n[e] + team[e])
        return np.array(idxs)

    def _score_teams(self, teams, w) -> np.ndarray:
        """Evaluate the linear surrogate for each team (features are sparse)."""
        return np.array([w[self._active_indices(t)].sum() for t in teams])

    # ── Posterior ─────────────────────────────────────────────────────────────

    def _posterior_factors(self):
        """Cholesky of posterior precision, posterior mean, IG(a_n, b_n)."""
        prec = self.XtX + np.diag(self.prior_prec)
        prec[np.diag_indices_from(prec)] += self.JITTER
        L = np.linalg.cholesky(prec)
        m = np.linalg.solve(L.T, np.linalg.solve(L, self.Xty))
        a_n = self.A0 + 0.5 * self.t
        b_n = self.B0 + 0.5 * max(self.yty - m @ self.Xty, 0.0)
        return L, m, a_n, b_n

    def _sample_weights(self):
        """Thompson sample: sigma^2 ~ IG(a_n, b_n), w ~ N(m, sigma^2 * Prec^-1)."""
        L, m, a_n, b_n = self._posterior_factors()
        sigma2 = b_n / np.random.gamma(a_n)
        z = np.random.normal(size=self.P)
        return m + np.sqrt(sigma2) * np.linalg.solve(L.T, z)

    # ── Discrete maximization (multi-start greedy local search) ──────────────

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

    # ── RecommendationAlgorithm interface ─────────────────────────────────────

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            w = self._sample_weights()
            self.last_choice = self._maximize(w)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        idxs = self._active_indices(arms_chosen)
        self.XtX[np.ix_(idxs, idxs)] += 1.0
        self.Xty[idxs] += reward
        self.yty += reward * reward
        self.t += 1
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

    def predict_best(self) -> List[int]:
        if self.t <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        _, m, _, _ = self._posterior_factors()
        team, _ = self._local_search(m, list(self.X_teams[int(np.argmax(self.y))]))
        return team
