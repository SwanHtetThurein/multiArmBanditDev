"""Pure exploration -- uniform allocation with a model-based recommendation.

Based on: Bubeck, Munos & Stoltz, "Pure Exploration in Multi-Armed Bandits
Problems" (ALT 2009), which formalizes *simple regret* -- the loss of the arm
you would recommend if forced to stop now -- and proves it trades off against
cumulative regret: the harder an algorithm works to earn reward while learning,
the worse its final recommendation tends to be.

Why this algorithm exists in the testbed:
    Our harness scores predict_best() every round against the hidden optimum.
    That is a *simple-regret* metric. But every other algorithm here allocates
    its 100 rounds so as to earn reward (Thompson sampling, EI, UCB all trade
    exploration against exploitation). Bubeck et al.'s result says that is the
    wrong objective for the metric we actually report.

    This plug-in is the control condition that separates the two halves of the
    problem. It spends the entire budget on uniform random teams -- the
    allocation Bubeck et al. show is a strong simple-regret baseline -- and
    puts all of its intelligence into the recommendation, fitting the same
    Bayesian linear surrogate BOCS uses and reporting its posterior-mean
    maximizer.

    So the comparison is exactly:
        random      = uniform allocation + no model  (performance floor)
        purexp      = uniform allocation + BOCS's model
        bocs/linucb/kg = adaptive allocation + BOCS's model
    Any gap between `purexp` and the adaptive methods is what adaptive
    allocation is worth here; any gap between `random` and `purexp` is what
    the surrogate alone is worth. Neither number is recoverable from the
    existing arms.

Note on the original algorithms: Bubeck et al.'s concrete procedures (uniform
sampling and UCB-E) assume finitely many *unstructured* arms, which does not
survive contact with a 10^5-team product space -- with 100 rounds you cannot
sample each team even once. The transferable content is the uniform-allocation
principle plus the separation of sampling from recommendation, which is what is
implemented here.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam); every subsequent
    round plays a uniformly random team, regardless of what has been observed.
  - predict_best() maximizes the posterior mean of a second-order Bayesian
    linear model by greedy local search -- the identical surrogate and
    identical optimizer used by bocs.py, linucb.py and knowledge_gradient.py,
    so the surrogate is held fixed across the comparison.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class PureExploration(RecommendationAlgorithm):
    name = "purexp"

    N_INIT = 10          # rounds before the surrogate is fitted at all
    G_FIRST = 1.0        # prior variance scale for first-order weights
    G_SECOND = 0.25      # stronger shrinkage on pairwise weights (as in bocs.py)
    JITTER = 1e-9
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
        self.t = 0
        self.A = np.diag(1.0 / prior_prec)     # (XtX + prior_prec)^-1

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

    # -- Features and posterior -----------------------------------------------

    def _active_indices(self, team) -> np.ndarray:
        idxs = [0]
        for d in range(self.D):
            idxs.append(self.first_offset[d] + team[d])
        for d in range(self.D):
            for e in range(d + 1, self.D):
                idxs.append(self.pair_offset[(d, e)] + team[d] * self.n[e] + team[e])
        return np.array(idxs)

    def _mean_scores(self, teams, m: np.ndarray) -> np.ndarray:
        return np.array([m[self._active_indices(t)].sum() for t in teams])

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
        else:
            # uniform allocation, always -- observations never steer sampling
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        idxs = self._active_indices(arms_chosen)

        # Sherman-Morrison rank-1 update of A = (XtX + prior_prec)^-1
        Aphi = self.A[:, idxs].sum(axis=1)
        denom = 1.0 + float(Aphi[idxs].sum())
        self.A -= np.outer(Aphi, Aphi) / denom

        self.XtX[np.ix_(idxs, idxs)] += 1.0
        self.Xty[idxs] += reward
        self.t += 1
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

        if self.t % self.EXACT_REFRESH == 0:
            self.A = np.linalg.inv(self.XtX + np.diag(self.prior_prec))

    def predict_best(self) -> List[int]:
        if self.t <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        m = self.A @ self.Xty
        team, _ = self._local_search(
            lambda teams: self._mean_scores(teams, m),
            list(self.X_teams[int(np.argmax(self.y))]))
        return team
