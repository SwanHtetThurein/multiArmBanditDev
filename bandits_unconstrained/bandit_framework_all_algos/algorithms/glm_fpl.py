"""GLM-FPL -- follow-the-perturbed-leader over a generalized linear model.

Based on: Kveton, Zaheer, Szepesvari, Li, Ghavamzadeh & Boutilier,
"Randomized Exploration in Generalized Linear Bandits", AISTATS 2020,
arXiv:1906.08947.

WHY IT IS IN THE BENCHMARK -- two reasons, and the first is about correctness.

1. THE LIKELIHOOD ACTUALLY MATCHES THIS ENVIRONMENT.
   Every other model in the suite assumes homoscedastic Gaussian noise: one
   variance, the same everywhere. This environment does not work that way. Its
   reward is bounded in [0, 1] and its noise is multiplicative,

        std = noise_level * p

   so the variance is *coupled to the mean* and vanishes as the reward goes to
   zero. That is precisely the mean-variance relationship a generalized linear
   model with a logistic link encodes natively, and it is a structural
   mis-specification in `bocs`, `linucb`, `kg`, `purexp` and every GP arm.
   This arm measures what the mis-specification costs.

2. IT EXPLORES BY PERTURBATION RATHER THAN BY A POSTERIOR.
   Follow-the-perturbed-leader adds noise to the *observed rewards*, refits,
   and acts greedily on the result. There is no confidence width to set, no
   covariance matrix to maintain, and one knob (the perturbation scale). At a
   100-round budget, where every posterior in this suite is thin and every
   theory-prescribed exploration constant needs shrinking by hand, a method
   whose exploration comes from resampling the data rather than from a fitted
   posterior is a genuinely different bet -- and it degrades gracefully when
   the model is misspecified, which the point above says it will be.

HOW IT WORKS
    Features: the same intercept + first-order + pairwise map as `bocs.py`, so
    the model class is comparable.

    Each round:
      1. Perturb every observed reward:  y~_i = clip(y_i + sigma * z_i)
      2. Refit the logistic GLM to the perturbed rewards by IRLS.
      3. Play the team maximizing the fitted mean (greedy -- all the
         exploration lives in step 1).

    predict_best() refits on the *unperturbed* rewards and maximizes that.

IMPLEMENTATION NOTE -- why this is cheap despite ~470 features
    IRLS would normally need a P x P solve per Newton step, with P ~ 470. But
    there are at most 100 observations, so the fit is done in the dual by the
    Woodbury identity, turning every solve into n x n.

    The Gram matrix has a closed form. Two teams sharing `m` of their D
    dimensions share the intercept, `m` first-order indicators, and every
    pairwise indicator whose *both* dimensions match, so

        (X X')_ij = 1 + m + m(m-1)/2

    with no feature vector ever materialized. Predictions are likewise
    kernelized: eta(x) = (1/lambda) * sum_i k(x, x_i) d_i. The whole algorithm
    touches only n x n matrices.
"""

import math
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class GLMFollowPerturbedLeader(RecommendationAlgorithm):
    name = "glm_fpl"

    N_INIT = 10          # random-exploration rounds before the model takes over
    PERTURB_SCALE = 0.25 # sigma of the reward perturbation (the one exploration knob)
    RIDGE = 1.0          # lambda, the ridge penalty on the GLM weights
    IRLS_STEPS = 4       # Newton/IRLS iterations per fit
    N_RESTARTS = 4       # local-search restarts
    W_FLOOR = 1e-6       # floor on the IRLS weights mu(1-mu)
    CLIP = 1e-3          # keep perturbed targets inside (0, 1) for the logit link

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

        self._d: Optional[np.ndarray] = None      # dual coefficients of the last fit

    # -- Kernel (closed-form Gram of the second-order one-hot map) ------------

    @staticmethod
    def _matches(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A[:, None, :] == B[None, :, :]).sum(axis=2).astype(float)

    def _gram(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """<phi(a), phi(b)> = 1 + m + m(m-1)/2 for m matching dimensions."""
        m = self._matches(np.asarray(A), np.asarray(B))
        return 1.0 + m + 0.5 * m * (m - 1.0)

    # -- Dual IRLS for logistic regression with a ridge penalty ---------------

    def _fit(self, targets: np.ndarray) -> np.ndarray:
        """Return dual coefficients d with eta = (1/lambda) G d."""
        X = np.array(self.X_teams)
        G = self._gram(X, X)
        n = len(targets)
        lam = self.RIDGE

        eta = np.zeros(n)
        d = np.zeros(n)
        for _ in range(self.IRLS_STEPS):
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            w = np.maximum(mu * (1.0 - mu), self.W_FLOOR)
            z = eta + (targets - mu) / w
            s = w * z
            M = G + lam * np.diag(1.0 / w)
            try:
                c = np.linalg.solve(M, G @ s)
            except np.linalg.LinAlgError:
                M[np.diag_indices_from(M)] += 1e-6
                c = np.linalg.solve(M, G @ s)
            d = s - c
            eta = (G @ d) / lam
        return d

    def _eta(self, teams, d: np.ndarray) -> np.ndarray:
        """Linear predictor at candidate teams (monotone in the fitted mean)."""
        K = self._gram(np.asarray(teams), np.array(self.X_teams))
        return (K @ d) / self.RIDGE

    # -- Discrete maximization -------------------------------------------------

    def _neighbors(self, team):
        neigh = []
        for dd in range(self.D):
            for a in range(self.n[dd]):
                if a != team[dd]:
                    t = list(team)
                    t[dd] = a
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
            y = np.array(self.y)
            # follow the PERTURBED leader: all exploration enters here
            perturbed = y + self.PERTURB_SCALE * np.random.normal(size=len(y))
            perturbed = np.clip(perturbed, self.CLIP, 1.0 - self.CLIP)
            d = self._fit(perturbed)
            self.last_choice = self._maximize(lambda teams: self._eta(teams, d))
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

    def predict_best(self) -> List[int]:
        if len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        y = np.clip(np.array(self.y), self.CLIP, 1.0 - self.CLIP)
        d = self._fit(y)          # unperturbed: no exploration in the recommendation
        team, _ = self._local_search(lambda teams: self._eta(teams, d),
                                     list(self.X_teams[int(np.argmax(self.y))]))
        return team
