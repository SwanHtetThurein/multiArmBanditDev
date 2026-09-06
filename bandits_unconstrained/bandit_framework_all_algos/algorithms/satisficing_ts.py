"""Satisficing Thompson Sampling -- stop paying to separate near-ties.

Based on: Russo, Tse & Van Roy, "Time-Sensitive Bandit Learning and
Satisficing Thompson Sampling", arXiv:1704.09028, 2017.

WHY IT IS IN THE BENCHMARK
    This paper is about our constraint, not merely compatible with it.

    Thompson sampling is asymptotically optimal, and that is exactly the
    problem: it keeps spending pulls to distinguish actions whose values are
    nearly identical, because given enough time that separation pays off. With
    100 rounds against ~10^5 teams there is no "enough time". A method that
    insists on identifying the single best team never leaves the exploration
    phase; one that settles for a demonstrably good-enough team can spend the
    remaining budget exploiting it.

    Russo, Tse & Van Roy formalize this: when the horizon is short relative to
    the problem, targeting the exact optimum is the wrong objective. Satisficing
    TS instead targets a *satisficing action* -- one within epsilon of optimal --
    and among those prefers the one that is cheapest to adopt.

    The surrogate is deliberately identical to `bocs.py`, so `bocs` (plain
    Thompson sampling) versus `sts` (satisficing Thompson sampling) isolates
    the satisficing rule alone, in the same way `linucb`, `kg` and `purexp`
    isolate their acquisition rules.

WHAT "CHEAPEST TO ADOPT" MEANS HERE
    The paper's cost term is problem-specific. In team composition it has an
    obvious and meaningful reading: the cost of adopting a team is the number
    of roles you must change to get there. So among all teams whose sampled
    value is within EPSILON of the sampled optimum, this implementation plays
    the one at minimum Hamming distance from the team currently in place.

    That makes the arm doubly relevant to this project. It is a short-horizon
    correction to Thompson sampling, and it is the only algorithm in the suite
    other than DreamTeam that treats *changing the team* as itself costly --
    but it arrives at that from decision theory rather than from a hand-set
    schedule and switching budget. Comparing `sts` against `dreamteam` is
    therefore a comparison of two different answers to the same question.

HOW IT WORKS
    Each round, after the usual Thompson draw from the second-order Bayesian
    linear posterior:

      1. Maximize the sampled model by multi-start local search -> S*, the
         value of the sampled-optimal team.
      2. Assemble the reachable candidates: the team currently in place, its
         single-role neighbourhood, and the local-search trajectory.
      3. Keep those scoring at least S* - EPSILON -- the satisficing set.
      4. Play the member of that set requiring the fewest role changes.
         If the set is empty, play the sampled optimum (no good-enough nearby
         team exists, so there is nothing to satisfice with).

    predict_best() maximizes the posterior mean, as in `bocs` -- satisficing
    governs what is *played*, not what is recommended.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class SatisficingTS(RecommendationAlgorithm):
    name = "sts"

    N_INIT = 10          # random-exploration rounds before the surrogate takes over
    EPSILON = 0.05       # satisficing tolerance, in reward units ([0, 1] scale)
    G_FIRST = 1.0        # prior variance scale for first-order weights
    G_SECOND = 0.25      # stronger shrinkage on pairwise weights (as in bocs.py)
    A0 = 2.0             # Inverse-Gamma prior on noise variance
    B0 = 0.05
    N_RESTARTS = 4
    JITTER = 1e-9

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

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

    # -- Features and posterior (identical to bocs.py) ------------------------

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

    def _posterior_factors(self):
        prec = self.XtX + np.diag(self.prior_prec)
        prec[np.diag_indices_from(prec)] += self.JITTER
        L = np.linalg.cholesky(prec)
        m = np.linalg.solve(L.T, np.linalg.solve(L, self.Xty))
        a_n = self.A0 + 0.5 * self.t
        b_n = self.B0 + 0.5 * max(self.yty - float(m @ self.Xty), 0.0)
        return L, m, a_n, b_n

    def _sample_weights(self):
        L, m, a_n, b_n = self._posterior_factors()
        sigma2 = b_n / np.random.gamma(a_n)
        z = np.random.normal(size=self.P)
        return m + np.sqrt(sigma2) * np.linalg.solve(L.T, z)

    # -- Discrete maximization, with the visited trajectory kept --------------

    def _neighbors(self, team):
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return neigh

    def _local_search(self, w, start, trail=None):
        current = list(start)
        current_score = float(self._score_teams([current], w)[0])
        if trail is not None:
            trail.append(list(current))
        while True:
            neigh = self._neighbors(current)
            if not neigh:
                return current, current_score
            scores = self._score_teams(neigh, w)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
                if trail is not None:
                    trail.append(list(current))
            else:
                return current, current_score

    def _maximize(self, w, trail=None):
        starts = [list(self.X_teams[int(np.argmax(self.y))])]
        for _ in range(self.N_RESTARTS - 1):
            starts.append([int(np.random.randint(k)) for k in self.n])
        best_team, best_score = None, -np.inf
        for s in starts:
            team, score = self._local_search(w, s, trail)
            if score > best_score:
                best_team, best_score = team, score
        return best_team, best_score

    @staticmethod
    def _hamming(a, b) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
            return list(self.last_choice)
        if round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
            return list(self.last_choice)

        w = self._sample_weights()
        trail: List[List[int]] = []
        best_team, best_score = self._maximize(w, trail)

        # candidates that are cheap to reach from where the team stands now
        current = list(self.last_choice)
        candidates = [current] + self._neighbors(current) + trail
        seen = set()
        unique = []
        for t in candidates:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                unique.append(list(t))

        scores = self._score_teams(unique, w)
        bar = best_score - self.EPSILON
        satisficing = [(self._hamming(t, current), t)
                       for t, s in zip(unique, scores) if s >= bar]

        if satisficing:
            satisficing.sort(key=lambda pair: pair[0])
            self.last_choice = list(satisficing[0][1])
        else:
            # nothing good enough within reach -- take the sampled optimum
            self.last_choice = list(best_team)
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
