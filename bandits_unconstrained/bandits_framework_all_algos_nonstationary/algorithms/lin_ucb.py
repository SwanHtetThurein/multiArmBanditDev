"""LinUCB-style algorithm plug-in: UCB acquisition over a Bayesian linear
surrogate with second-order interaction features.

Based on: Li, Chu, Langford & Schapire, "A Contextual-Bandit Approach to
Personalized News Article Recommendation" (WWW 2010), arXiv:1003.5956.
LinUCB is a standard baseline in both neural-bandit papers this project
follows -- NeuralUCB (arXiv:1911.04462) and NeuralTS (arXiv:2010.00827).

Why it is here, and what it controls for:
    bocs.py already fits exactly this surrogate (a conjugate Bayesian linear
    model over first- and second-order one-hot features) but explores by
    *Thompson sampling* -- drawing a weight vector from the posterior. This
    plug-in keeps the surrogate byte-for-byte equivalent and swaps only the
    acquisition rule for the optimism-based one:

        score(team) = mu(team) + ALPHA * sd(team)

    Running the two side by side therefore isolates a single experimental
    factor -- optimism-in-the-face-of-uncertainty vs. posterior sampling --
    with the model class, the feature map and the discrete optimizer all
    held fixed. That is a much cleaner comparison than adding another
    algorithm that differs in several ways at once.

Adaptation note (important for the writeup):
    Textbook LinUCB assumes a per-round *context* that is revealed before the
    action is chosen, and a small explicit arm set. Our environment has no
    such context: the only thing that varies is the team we pick. So the
    feature vector is the team's own encoding, and the "arm set" is the whole
    combinatorial team space, searched by multi-start greedy local search
    instead of enumerated. This is the same adaptation neural_linear.py makes
    to the contextual-bandit methods from the Deep Bayesian Bandits Showdown.

Implementation notes:
  - The posterior covariance A = (X'X + prior_prec)^-1 is maintained
    incrementally by the Sherman-Morrison identity rather than re-factorized
    each round. Because the feature vector is a 0/1 indicator, A @ phi is a
    column sum and phi' A phi is a sub-block sum, so both the update and the
    per-candidate variance are cheap. An exact re-inversion runs every
    EXACT_REFRESH observations to stop numerical drift accumulating.
  - The observation-noise scale uses the posterior mean of the
    Inverse-Gamma, sigma^2 = b_n / (a_n - 1), so the confidence width adapts
    to how noisy the run actually is instead of being a fixed constant.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (random initialization).
  - predict_best() maximizes the posterior *mean* (no exploration bonus) by
    local search from the best observed team.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class LinUCBAlgorithm(RecommendationAlgorithm):
    name = "linucb"

    N_INIT = 10          # random-exploration rounds before the surrogate takes over
    ALPHA = 1.0          # UCB width multiplier (the classic LinUCB alpha)
    G_FIRST = 1.0        # prior variance scale for first-order weights
    G_SECOND = 0.25      # stronger shrinkage on pairwise weights (as in bocs.py)
    A0 = 2.0             # Inverse-Gamma prior on noise variance
    B0 = 0.05
    N_RESTARTS = 4       # local-search restarts for acquisition maximization
    EXACT_REFRESH = 25   # re-invert exactly every k observations

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        # -- Feature index layout (identical to bocs.py) ----------------------
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

        prior_prec = np.empty(self.P)
        prior_prec[0] = 1.0 / self.G_FIRST
        for d in range(self.D):
            o = self.first_offset[d]
            prior_prec[o:o + self.n[d]] = 1.0 / self.G_FIRST
        for (d, e), o in self.pair_offset.items():
            prior_prec[o:o + self.n[d] * self.n[e]] = 1.0 / self.G_SECOND
        self.prior_prec = prior_prec

        # sufficient statistics + maintained posterior covariance
        self.XtX = np.zeros((self.P, self.P))
        self.Xty = np.zeros(self.P)
        self.yty = 0.0
        self.t = 0
        self.A = np.diag(1.0 / prior_prec)       # (XtX + prior_prec)^-1

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

    # -- Features -------------------------------------------------------------

    def _active_indices(self, team) -> np.ndarray:
        """Indices of the nonzero (=1) features for a team."""
        idxs = [0]
        for d in range(self.D):
            idxs.append(self.first_offset[d] + team[d])
        for d in range(self.D):
            for e in range(d + 1, self.D):
                idxs.append(self.pair_offset[(d, e)] + team[d] * self.n[e] + team[e])
        return np.array(idxs)

    # -- Posterior ------------------------------------------------------------

    def _posterior_mean(self) -> np.ndarray:
        return self.A @ self.Xty

    def _noise_scale(self, m: np.ndarray) -> float:
        a_n = self.A0 + 0.5 * self.t
        b_n = self.B0 + 0.5 * max(self.yty - float(m @ self.Xty), 0.0)
        return b_n / max(a_n - 1.0, 1e-6)

    def _mean_scores(self, teams, m: np.ndarray) -> np.ndarray:
        return np.array([m[self._active_indices(t)].sum() for t in teams])

    def _ucb_scores(self, teams, m: np.ndarray, sigma2: float) -> np.ndarray:
        out = np.empty(len(teams))
        for i, t in enumerate(teams):
            idx = self._active_indices(t)
            mu = m[idx].sum()
            var = sigma2 * float(self.A[np.ix_(idx, idx)].sum())
            out[i] = mu + self.ALPHA * np.sqrt(max(var, 0.0))
        return out

    # -- Discrete maximization (multi-start greedy local search) --------------

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

    def _maximize(self, score_fn):
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
        elif round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            m = self._posterior_mean()
            sigma2 = self._noise_scale(m)
            self.last_choice = self._maximize(
                lambda teams: self._ucb_scores(teams, m, sigma2))
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        idxs = self._active_indices(arms_chosen)

        # Sherman-Morrison rank-1 update of A = (XtX + prior_prec)^-1.
        # phi is a 0/1 indicator, so A @ phi is a column sum over `idxs`.
        Aphi = self.A[:, idxs].sum(axis=1)
        denom = 1.0 + float(Aphi[idxs].sum())
        self.A -= np.outer(Aphi, Aphi) / denom

        self.XtX[np.ix_(idxs, idxs)] += 1.0
        self.Xty[idxs] += reward
        self.yty += reward * reward
        self.t += 1
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

        if self.t % self.EXACT_REFRESH == 0:      # kill accumulated drift
            self.A = np.linalg.inv(self.XtX + np.diag(self.prior_prec))

    def predict_best(self) -> List[int]:
        if self.t <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        m = self._posterior_mean()
        team, _ = self._local_search(
            lambda teams: self._mean_scores(teams, m),
            list(self.X_teams[int(np.argmax(self.y))]))
        return team
