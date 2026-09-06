"""SMAC-style algorithm plug-in (random-forest surrogate + EI).

Based on: Hutter, Hoos & Leyton-Brown, "Sequential Model-Based Optimization
for General Algorithm Configuration" (LION 2011). SMAC is a baseline in both
combinatorial-BO papers this project follows -- BOCS (arXiv:1806.08838) and
COMBO (arXiv:1902.00448) -- and is the standard "tree surrogate instead of a
GP" comparison point.

Where COMBO uses a GP with a diffusion kernel and BOCS a sparse Bayesian
linear model, SMAC fits a *random forest* of regression trees directly on the
categorical variables and reads uncertainty off the spread of the trees:

    mu(team)  = mean prediction across trees
    sd(team)  = std-dev of predictions across trees   (frequentist, not Bayesian)

Expected Improvement is then computed from (mu, sd) exactly as for a GP. The
appeal for our setting is that trees handle categorical variables natively,
with no kernel and no continuous relaxation, and they capture interactions
between dimensions without an explicit second-order feature expansion.

Faithful pieces:
  - Random-forest surrogate over the raw categorical team vector, with
    randomized split selection (a random subset of dimensions considered per
    node, a random value-subset split drawn per candidate dimension, best
    variance reduction among those wins -- SMAC's randomized RF).
  - Empirical across-tree mean/variance as the predictive distribution.
  - Expected Improvement acquisition, maximized by multi-start greedy local
    search over single-arm changes.
  - Interleaved random configurations: SMAC alternates model-proposed and
    uniformly random candidates so it keeps exploring even if the surrogate
    is badly wrong. Here every INTERLEAVE_RANDOM-th round is random.

Simplifications vs. the original (documented for honesty in comparisons):
  - No log-transform of the objective and no censored/racing machinery: our
    objective is a bounded reward in [0, 1] evaluated once per round, so
    SMAC's runtime-specific handling (capped runs, intensification across
    instances) has nothing to act on.
  - The forest is refit every REFIT_EVERY observations rather than every
    round, for speed inside the large experiment sweep (same tradeoff as
    COMBO's REFIT_EVERY).

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (SMAC's initial design).
  - predict_best() maximizes the forest's mean prediction by local search;
    before enough data exists it returns the best observed team.
"""

import math
import random
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z])


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


class SmacAlgorithm(RecommendationAlgorithm):
    name = "smac"

    N_INIT = 10             # random initial design before the forest takes over
    N_TREES = 10            # forest size
    MIN_SPLIT = 4           # minimum samples in a node to attempt a split
    MAX_DEPTH = 8
    REFIT_EVERY = 5         # refit the forest every k new observations
    N_RESTARTS = 4          # local-search restarts for acquisition maximization
    INTERLEAVE_RANDOM = 2   # every k-th post-init round plays a random team

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

        self._forest: Optional[List[dict]] = None
        self._fitted_n = 0

    # -- Randomized regression trees ------------------------------------------

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if (depth >= self.MAX_DEPTH or len(y) < self.MIN_SPLIT
                or float(y.std()) < 1e-9):
            return {"leaf": True, "value": float(y.mean())}

        dims = [d for d in range(self.D) if np.unique(X[:, d]).size > 1]
        if not dims:
            return {"leaf": True, "value": float(y.mean())}

        mtry = max(1, len(dims) // 3)
        candidates = random.sample(dims, min(mtry, len(dims)))

        best = None
        best_score = None
        for d in candidates:
            values = [int(v) for v in np.unique(X[:, d])]
            random.shuffle(values)
            cut = random.randint(1, len(values) - 1)
            left_values = values[:cut]
            mask = np.isin(X[:, d], left_values)
            if mask.all() or not mask.any():
                continue
            # weighted within-node variance (lower is better)
            score = (mask.sum() * float(y[mask].var())
                     + (~mask).sum() * float(y[~mask].var()))
            if best_score is None or score < best_score:
                best_score = score
                best = (d, left_values, mask)

        if best is None:
            return {"leaf": True, "value": float(y.mean())}

        d, left_values, mask = best
        return {
            "leaf": False,
            "d": d,
            "left_values": left_values,
            "left": self._build_tree(X[mask], y[mask], depth + 1),
            "right": self._build_tree(X[~mask], y[~mask], depth + 1),
        }

    def _predict_tree(self, node: dict, X: np.ndarray, out: np.ndarray,
                      idx: np.ndarray) -> None:
        """Vectorized descent: route index sets down the tree."""
        if node["leaf"]:
            out[idx] = node["value"]
            return
        mask = np.isin(X[idx, node["d"]], node["left_values"])
        left_idx, right_idx = idx[mask], idx[~mask]
        if left_idx.size:
            self._predict_tree(node["left"], X, out, left_idx)
        if right_idx.size:
            self._predict_tree(node["right"], X, out, right_idx)

    def _fit_forest(self) -> None:
        X = np.array(self.X_teams)
        y = np.array(self.y)
        self._forest = [self._build_tree(X, y, 0) for _ in range(self.N_TREES)]
        self._fitted_n = len(y)

    def _forest_stats(self, teams: np.ndarray):
        """Across-tree mean and std for each team: (m, D) -> (mu[m], sd[m])."""
        X = np.asarray(teams)
        idx = np.arange(len(X))
        preds = np.empty((len(self._forest), len(X)))
        for t, tree in enumerate(self._forest):
            out = np.empty(len(X))
            self._predict_tree(tree, X, out, idx)
            preds[t] = out
        mu = preds.mean(axis=0)
        sd = np.sqrt(np.maximum(preds.var(axis=0), 1e-12))
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
        elif round_num <= self.N_INIT or self._forest is None:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        elif round_num % self.INTERLEAVE_RANDOM == 1:
            # SMAC interleaves uniformly random configurations with model ones
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            best_y = float(np.max(self.y))

            def ei_score(teams):
                mu, sd = self._forest_stats(teams)
                return self._ei(mu, sd, best_y)

            self.last_choice = self._maximize(ei_score)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
        if len(self.y) >= self.N_INIT and (self._forest is None
                                           or len(self.y) - self._fitted_n >= self.REFIT_EVERY):
            self._fit_forest()

    def predict_best(self) -> List[int]:
        if self._forest is None or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))

        def mean_score(teams):
            mu, _ = self._forest_stats(teams)
            return mu

        team, _ = self._local_search(mean_score, list(self.X_teams[int(np.argmax(self.y))]))
        return team
