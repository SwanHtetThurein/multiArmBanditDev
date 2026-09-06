"""BayesGap -- gap-based best-arm identification under a fixed budget.

Based on: Hoffman, Shahriari & de Freitas, "Exploiting correlation and budget
constraints in Bayesian multi-armed bandit optimization" (arXiv:1303.6746,
AISTATS 2014), which extends Gabillon et al.'s BayesGap/UGapE to arms whose
rewards are *correlated* through a Bayesian model.

Why this algorithm is in the testbed:
    Our harness scores predict_best() against the hidden optimum every round.
    That is a pure best-arm-identification objective under a fixed budget of
    100 rounds -- and BayesGap is the only algorithm in this comparison that
    was actually designed for it. Every other arm here (Thompson sampling,
    EI, UCB, KG) is optimizing some notion of reward earned along the way.

The algorithm:
    For each candidate team k, form a posterior mean mu_k and an interval
    [L_k, U_k] = mu_k -/+ BETA * sigma_k. Define the *gap index*

        B_k = max_{j != k} U_j - L_k

    which upper-bounds how much better the true best arm could be than arm k.
    Then each round:
        J = argmin_k B_k                  (the arm we would recommend)
        j = argmax_{k != J} U_k           (its most threatening challenger)
        pull whichever of {J, j} has the larger posterior uncertainty
    and recommend, at any point, the J that has achieved the smallest B_J so
    far. Sampling is therefore aimed squarely at resolving the ambiguity
    between the leader and its closest rival, rather than at collecting reward.

Adaptations to this framework (documented for honesty in comparisons):
  - Arms. BayesGap assumes a fixed, enumerated arm set; we have ~10^5 teams.
    We maintain a growing candidate pool seeded with random teams and the
    initial bias, and each round we add the current posterior-mean maximizer
    (found by local search) if it is new. The pool therefore tracks the
    promising region without ever enumerating the space, and the incumbent is
    always in it.
  - Correlation. Hoffman et al. use a GP to correlate arms. We use the same
    second-order Bayesian linear model as bocs.py / linucb.py / kg.py so that
    this arm differs from those three only in its *allocation rule*, not in
    its surrogate. Posterior mean and variance for a team come from that model.
  - Exploration width. The paper adapts BETA to the budget via a hardness
    estimate; we use a fixed BETA (tunable below), which is the common
    practical simplification.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams, which also seed the arm pool.
  - predict_best() returns BayesGap's own recommendation -- the arm with the
    smallest gap index seen so far -- not a fresh argmax. That is the
    quantity the algorithm's guarantee is about.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class BayesGap(RecommendationAlgorithm):
    name = "bayesgap"

    N_INIT = 10          # random rounds that also seed the arm pool
    BETA = 1.0           # confidence-interval width multiplier
    POOL_SEED = 60       # random teams placed in the pool up front
    MAX_POOL = 400       # cap on the candidate pool
    G_FIRST = 1.0        # prior variance scale for first-order weights
    G_SECOND = 0.25      # stronger shrinkage on pairwise weights (as in bocs.py)
    A0 = 2.0             # Inverse-Gamma prior on noise variance
    B0 = 0.05
    EXACT_REFRESH = 25   # re-invert exactly every k observations

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

        prior_prec = np.empty(self.P)
        prior_prec[0] = 1.0 / self.G_FIRST
        for d in range(self.D):
            o = self.first_offset[d]
            prior_prec[o:o + self.n[d]] = 1.0 / self.G_FIRST
        for (d, e), o in self.pair_offset.items():
            prior_prec[o:o + self.n[d] * self.n[e]] = 1.0 / self.G_SECOND
        self.prior_prec = prior_prec

        self.XtX = np.zeros((self.P, self.P))
        self.Xty = np.zeros(self.P)
        self.yty = 0.0
        self.t = 0
        self.A = np.diag(1.0 / prior_prec)     # (XtX + prior_prec)^-1

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

        # candidate arm pool
        self._pool: List[List[int]] = []
        self._pool_keys = set()
        self._add_arm(list(initial_bias))
        for _ in range(self.POOL_SEED):
            self._add_arm([int(np.random.randint(k)) for k in self.n])

        # BayesGap's recommendation: the arm with the smallest gap index so far
        self._recommendation: Optional[List[int]] = list(initial_bias)
        self._best_gap = np.inf

    # -- pool ------------------------------------------------------------------

    def _add_arm(self, team: List[int]) -> None:
        key = tuple(team)
        if key not in self._pool_keys and len(self._pool) < self.MAX_POOL:
            self._pool_keys.add(key)
            self._pool.append(list(team))

    # -- Features and posterior -----------------------------------------------

    def _active_indices(self, team) -> np.ndarray:
        idxs = [0]
        for d in range(self.D):
            idxs.append(self.first_offset[d] + team[d])
        for d in range(self.D):
            for e in range(d + 1, self.D):
                idxs.append(self.pair_offset[(d, e)] + team[d] * self.n[e] + team[e])
        return np.array(idxs)

    def _posterior_mean(self) -> np.ndarray:
        return self.A @ self.Xty

    def _noise_scale(self, m: np.ndarray) -> float:
        a_n = self.A0 + 0.5 * self.t
        b_n = self.B0 + 0.5 * max(self.yty - float(m @ self.Xty), 0.0)
        return b_n / max(a_n - 1.0, 1e-6)

    def _mean_scores(self, teams, m: np.ndarray) -> np.ndarray:
        return np.array([m[self._active_indices(t)].sum() for t in teams])

    def _mean_and_sd(self, teams, m: np.ndarray, lam: float):
        mu = np.empty(len(teams))
        sd = np.empty(len(teams))
        for i, t in enumerate(teams):
            idx = self._active_indices(t)
            mu[i] = m[idx].sum()
            var = lam * float(self.A[np.ix_(idx, idx)].sum())
            sd[i] = np.sqrt(max(var, 1e-12))
        return mu, sd

    # -- Discrete maximization -------------------------------------------------

    def _neighbors(self, team):
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return neigh

    def _local_search(self, score_fn, start):
        current = list(start)
        current_score = float(score_fn([current])[0])
        while True:
            neigh = self._neighbors(current)
            if not neigh:
                return current, current_score
            scores = score_fn(neigh)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current, current_score

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
            return list(self.last_choice)

        if round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
            self._add_arm(self.last_choice)
            return list(self.last_choice)

        m = self._posterior_mean()
        lam = self._noise_scale(m)

        # grow the pool toward the promising region
        best_obs = list(self.X_teams[int(np.argmax(self.y))])
        best_post, _ = self._local_search(
            lambda teams: self._mean_scores(teams, m), best_obs)
        self._add_arm(best_post)
        self._add_arm(best_obs)

        mu, sd = self._mean_and_sd(self._pool, m, lam)
        U = mu + self.BETA * sd
        L = mu - self.BETA * sd

        # B_k = max_{j != k} U_j - L_k, via the top two values of U
        order = np.argsort(U)
        top1, top2 = int(order[-1]), int(order[-2]) if len(order) > 1 else int(order[-1])
        max_other = np.where(np.arange(len(U)) == top1, U[top2], U[top1])
        B = max_other - L

        J = int(np.argmin(B))
        if B[J] < self._best_gap:
            self._best_gap = float(B[J])
            self._recommendation = list(self._pool[J])

        # most threatening challenger to J
        masked = U.copy()
        masked[J] = -np.inf
        j = int(np.argmax(masked)) if len(U) > 1 else J

        # sample whichever of the two we know least about
        pick = J if sd[J] >= sd[j] else j
        self.last_choice = list(self._pool[pick])
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        idxs = self._active_indices(arms_chosen)

        Aphi = self.A[:, idxs].sum(axis=1)
        denom = 1.0 + float(Aphi[idxs].sum())
        self.A -= np.outer(Aphi, Aphi) / denom

        self.XtX[np.ix_(idxs, idxs)] += 1.0
        self.Xty[idxs] += reward
        self.yty += reward * reward
        self.t += 1
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
        self._add_arm(list(arms_chosen))

        if self.t % self.EXACT_REFRESH == 0:
            self.A = np.linalg.inv(self.XtX + np.diag(self.prior_prec))

    def predict_best(self) -> List[int]:
        if self.t <= self.N_INIT or self._recommendation is None:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        return list(self._recommendation)
