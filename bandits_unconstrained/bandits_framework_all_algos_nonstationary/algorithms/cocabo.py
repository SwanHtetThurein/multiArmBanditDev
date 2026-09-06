"""CoCaBO -- multi-agent bandit over categories combined with a GP surrogate.

Based on: Ru, Alvi, Nguyen, Osborne & Roberts, "Bayesian Optimisation over
Multiple Continuous and Categorical Inputs" (ICML 2020), arXiv:1906.08878.

CoCaBO's design is unusually well matched to this problem: instead of forcing
categorical variables into a continuous relaxation, it puts an *independent
multi-armed bandit agent on each categorical dimension* and lets a GP handle
the rest. That is very nearly a description of our setting -- one agent per
role, each choosing among that role's candidates.

Two faithful pieces:

1. Multi-agent EXP3. One EXP3 agent per bandit dimension. Each keeps weights
   over its own arms, plays from the mixture
       p_a = (1 - gamma) * w_a / sum(w) + gamma / n
   and, on observing the shared team reward r, applies the importance-weighted
   update w_a *= exp(gamma * (r / p_a) / n) for the arm it played. The agents
   never communicate: they coordinate only through the reward they all
   receive, which is exactly CoCaBO's decomposition and is what keeps the cost
   linear rather than exponential in the number of dimensions. EXP3 is used
   rather than a stochastic bandit because the reward each agent sees is
   non-stationary -- it depends on what the *other* agents are doing.

2. The categorical overlap kernel. CoCaBO's kernel for categorical inputs is
       k(h, h') = (sigma_f^2 / D) * sum_d 1[h_d = h'_d]
   i.e. similarity is *linear* in the number of shared choices. Note this is a
   genuinely different modelling assumption from anything else in the testbed:
   gp_onehot decays exponentially in Hamming distance and combo uses a
   diffusion kernel, both of which make distant teams nearly uncorrelated,
   whereas the overlap kernel keeps a floor of correlation and so pools
   information much more aggressively across the space. At 100 rounds over
   ~10^5 teams that pooling is the whole game.

Honest note on the pure-categorical case:
    CoCaBO's published algorithm splits the round: EXP3 picks the categorical
    values, then the GP optimizes the *continuous* variables conditioned on
    them. Our problem has no continuous variables, so taken literally CoCaBO
    would degenerate to plain multi-agent EXP3 and its kernel contribution
    would do nothing. That would be a misleading thing to label "CoCaBO".

    Instead we follow the logic of CoCaBO-B, the paper's batch variant, where
    EXP3 fixes part of the batch and the GP proposes the rest: a fraction
    EXP3_FRACTION of rounds play the EXP3 draw directly, and the remaining
    rounds play the GP's Expected-Improvement maximizer over a candidate set
    built around the EXP3 draw and the incumbent. Crucially, the EXP3 agents
    are updated *only* on rounds where the action really came from their own
    distribution -- otherwise the importance weighting r / p_a would be
    biased. Both halves of the paper's method therefore stay live and the
    bandit estimator stays unbiased.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (BO random initialization).
  - predict_best() maximizes the GP posterior mean by local search.
"""

import math
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z])


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


class CoCaBO(RecommendationAlgorithm):
    name = "cocabo"

    N_INIT = 10            # random-exploration rounds before the GP takes over
    EXP3_FRACTION = 0.5    # share of rounds played straight from the EXP3 draw
    REFIT_EVERY = 10       # re-tune GP hyperparameters every k rounds
    JITTER = 1e-8
    #: noise-to-signal grid for the GP (signal variance is profiled out)
    ETA_GRID = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        # -- one EXP3 agent per dimension (log-weights for numerical safety) --
        self.log_w = [np.zeros(k) for k in self.n]
        self.gamma = [
            min(1.0, math.sqrt(k * math.log(max(k, 2))
                               / ((math.e - 1.0) * max(total_rounds, 2))))
            for k in self.n
        ]
        self._last_probs: Optional[List[np.ndarray]] = None
        self._played_from_exp3 = False

        # -- GP state ---------------------------------------------------------
        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.eta = 0.1
        self.sf2 = 0.25
        self._chol = None
        self._alpha = None
        self._ymean = 0.0
        self._rounds_since_tune = 0

        self.last_choice = list(initial_bias)

    # -- EXP3 -------------------------------------------------------------------

    def _exp3_probs(self) -> List[np.ndarray]:
        probs = []
        for d in range(self.D):
            lw = self.log_w[d] - self.log_w[d].max()
            w = np.exp(lw)
            w = w / w.sum()
            g = self.gamma[d]
            probs.append((1.0 - g) * w + g / self.n[d])
        return probs

    def _exp3_draw(self, probs) -> List[int]:
        return [int(np.random.choice(self.n[d], p=probs[d])) for d in range(self.D)]

    def _exp3_update(self, arms_chosen, reward, probs) -> None:
        r = min(max(float(reward), 0.0), 1.0)
        for d, a in enumerate(arms_chosen):
            p = max(float(probs[d][a]), 1e-12)
            self.log_w[d][a] += self.gamma[d] * (r / p) / self.n[d]
            self.log_w[d] -= self.log_w[d].max()      # keep magnitudes bounded

    # -- GP with CoCaBO's categorical overlap kernel ---------------------------

    def _overlap(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """k(h, h') / sigma_f^2 = (1/D) * number of matching dimensions."""
        return (A[:, None, :] == B[None, :, :]).sum(axis=2) / float(self.D)

    def _refresh(self, tune: bool) -> None:
        X = np.array(self.X_teams)
        y = np.array(self.y)
        n = len(y)
        self._ymean = float(y.mean())
        yc = y - self._ymean
        R = self._overlap(X, X)

        if tune and n >= 5:
            best = None
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
                nll = (0.5 * n * math.log(sf2) + float(np.log(np.diag(L)).sum())
                       + 0.5 * n)
                if best is None or nll < best[0]:
                    best = (nll, eta, sf2)
            if best is not None:
                _, self.eta, self.sf2 = best

        M = R + (self.eta + self.JITTER) * np.eye(n)
        try:
            self._chol = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            M[np.diag_indices_from(M)] += 1e-4
            self._chol = np.linalg.cholesky(M)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, yc))

    def _posterior(self, teams: np.ndarray):
        X = np.array(self.X_teams)
        Rs = self._overlap(np.asarray(teams), X)
        mu = self._ymean + Rs @ self._alpha
        v = np.linalg.solve(self._chol, Rs.T)
        var = self.sf2 * (1.0 - np.einsum("ij,ij->j", v, v))
        return mu, np.sqrt(np.maximum(var, 1e-12))

    @staticmethod
    def _ei(mu, sd, best):
        z = (mu - best) / sd
        return (mu - best) * _norm_cdf(z) + sd * _norm_pdf(z)

    # -- Candidates -------------------------------------------------------------

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
        current_score = float(score_fn(np.array([current]))[0])
        while True:
            neigh = self._neighbors(current)
            if not neigh:
                return current, current_score
            scores = score_fn(np.array(neigh))
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current, current_score

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        self._played_from_exp3 = False
        probs = self._exp3_probs()
        self._last_probs = probs

        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT or self._chol is None:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            draw = self._exp3_draw(probs)
            if np.random.random() < self.EXP3_FRACTION:
                # play the bandit's own choice -- keeps its estimator unbiased
                self.last_choice = draw
                self._played_from_exp3 = True
            else:
                # GP refines around the bandit's draw and the incumbent
                best_y = float(np.max(self.y))
                best_obs = list(self.X_teams[int(np.argmax(self.y))])

                def ei_score(teams):
                    mu, sd = self._posterior(teams)
                    return self._ei(mu, sd, best_y)

                cand_a, score_a = self._local_search(ei_score, draw)
                cand_b, score_b = self._local_search(ei_score, best_obs)
                self.last_choice = cand_a if score_a >= score_b else cand_b
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

        # EXP3 is only updated when the played action came from its own
        # distribution; otherwise the r / p importance weighting is invalid.
        if self._played_from_exp3 and self._last_probs is not None:
            self._exp3_update(arms_chosen, reward, self._last_probs)

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

        team, _ = self._local_search(mean_score,
                                     list(self.X_teams[int(np.argmax(self.y))]))
        return team
