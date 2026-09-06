"""Knowledge Gradient with correlated beliefs (KGCB).

Based on: Negoescu, Frazier & Powell, "The Knowledge-Gradient Algorithm for
Sequencing Experiments in Drug Discovery" (INFORMS Journal on Computing 23(3),
2011), with the exact one-dimensional computation from Frazier, Powell &
Dayanik, "The Knowledge-Gradient Policy for Correlated Normal Beliefs"
(INFORMS JoC 21(4), 2009).

Why this algorithm is in the testbed:
    Negoescu et al.'s problem is structurally almost identical to ours. A
    candidate compound is a vector of categorical substituent choices at
    several sites; you synthesize one, measure one noisy scalar, and you have
    a very small budget. They put a Bayesian linear model on the one-hot
    encoding of those choices and choose what to measure by Knowledge Gradient.

    It also fills the one acquisition family the testbed was missing. We had
    Thompson sampling (bocs), Expected Improvement (combo, smac, gp_onehot)
    and optimism (linucb) -- all of which score a candidate by how good it
    might turn out to be. KG scores a candidate by something different: how
    much better the *final recommendation* will be after measuring it. It is
    the only genuinely decision-theoretic, one-step-optimal criterion here, and
    it is the acquisition that explicitly optimizes the simple-regret metric
    our harness actually reports.

    The surrogate is deliberately byte-for-byte the same second-order Bayesian
    linear model used by bocs.py, linucb.py and pure_exploration.py, so the
    comparison across those four isolates the acquisition rule alone:
        bocs   = Thompson sampling
        linucb = upper confidence bound
        kg     = knowledge gradient
        purexp = no acquisition at all (uniform allocation)

How KG works here:
    With belief theta ~ N(m, S), measuring team x with noise variance lambda
    updates every alternative's posterior mean along a single random direction:

        mu'_i = a_i + b_i * Z,    Z ~ N(0, 1)
        a_i = phi_i' m
        b_i = phi_i' S phi_x / sqrt(lambda + phi_x' S phi_x)

    so KG(x) = E[max_i (a_i + b_i Z)] - max_i a_i is the expectation of the
    maximum of a set of *lines* in one scalar variable. That has an exact
    closed form: sort the lines by slope, discard the ones never on the upper
    envelope, and sum (b_{i+1} - b_i) * f(-|c_i|) over the envelope's
    breakpoints c_i, where f(z) = z*Phi(z) + phi(z). No sampling, no
    approximation -- `_kg_from_lines` below implements exactly that.

Simplifications vs. the paper (documented for honesty in comparisons):
  - KG is evaluated over a candidate *pool* rather than all ~10^5 teams. Each
    round the pool is rebuilt from the best observed team, the posterior-mean
    maximizer, their single-arm neighbourhoods, and random teams. Scoring
    every team exactly would cost a 10^5-line envelope computation per
    candidate per round; restricting to a pool is the standard practice for
    KGCB on large discrete sets, and the pool always contains the incumbent so
    the recommendation can never get worse for lack of coverage.
  - The measurement-noise variance lambda is estimated online from the
    Inverse-Gamma posterior rather than assumed known.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (the paper's initial design).
  - predict_best() maximizes the posterior mean by local search -- KG's own
    notion of the current best alternative.
"""

import math
from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z])


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


def _kg_from_lines(a: np.ndarray, b: np.ndarray) -> float:
    """Exact E[max_i(a_i + b_i Z)] - max_i a_i for Z ~ N(0,1).

    Frazier, Powell & Dayanik (2009), Algorithm 2: keep only the lines that
    appear on the upper envelope, then integrate the envelope against the
    standard normal.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    order = np.argsort(b, kind="stable")
    a, b = a[order], b[order]

    # among lines with identical slope only the highest intercept can matter
    keep = []
    i = 0
    M = len(b)
    while i < M:
        j = i
        while j + 1 < M and b[j + 1] == b[i]:
            j += 1
        keep.append(i + int(np.argmax(a[i:j + 1])))
        i = j + 1
    a, b = a[keep], b[keep]
    M = len(b)
    if M < 2:
        return 0.0

    # drop lines that never reach the upper envelope
    A = [0]
    C: List[float] = []          # C[k] = crossover between A[k] and A[k+1]
    for i in range(1, M):
        while True:
            j = A[-1]
            denom = b[i] - b[j]
            if denom <= 0:       # cannot happen after the dedup above; guard
                break
            c = (a[j] - a[i]) / denom
            if len(A) >= 2 and c <= C[-1]:
                A.pop()
                C.pop()
            else:
                A.append(i)
                C.append(c)
                break

    if len(A) < 2:
        return 0.0

    bb = b[A]
    cc = np.array(C, dtype=float)
    z = -np.abs(cc)
    f = z * _norm_cdf(z) + _norm_pdf(z)
    return float(np.sum(np.diff(bb) * f))


class KnowledgeGradient(RecommendationAlgorithm):
    name = "kg"

    N_INIT = 10          # random-exploration rounds before KG takes over
    POOL_SIZE = 40       # candidate alternatives scored per round
    N_RANDOM_POOL = 12   # random teams mixed into the pool
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

        # number of distinct teams, capped so tiny problems don't spin in
        # _build_pool looking for candidates that do not exist
        size = 1.0
        for k in self.n:
            size *= k
            if size > 1e6:
                break
        self._space_size = int(min(size, 1e6))

    # -- Features and posterior -----------------------------------------------

    def _active_indices(self, team) -> np.ndarray:
        idxs = [0]
        for d in range(self.D):
            idxs.append(self.first_offset[d] + team[d])
        for d in range(self.D):
            for e in range(d + 1, self.D):
                idxs.append(self.pair_offset[(d, e)] + team[d] * self.n[e] + team[e])
        return np.array(idxs)

    def _dense_features(self, teams) -> np.ndarray:
        Phi = np.zeros((len(teams), self.P))
        for i, t in enumerate(teams):
            Phi[i, self._active_indices(t)] = 1.0
        return Phi

    def _posterior_mean(self) -> np.ndarray:
        return self.A @ self.Xty

    def _noise_scale(self, m: np.ndarray) -> float:
        a_n = self.A0 + 0.5 * self.t
        b_n = self.B0 + 0.5 * max(self.yty - float(m @ self.Xty), 0.0)
        return b_n / max(a_n - 1.0, 1e-6)

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

    def _build_pool(self, m: np.ndarray) -> List[List[int]]:
        """Candidate alternatives: incumbents, their neighbourhoods, randoms."""
        best_obs = list(self.X_teams[int(np.argmax(self.y))])
        best_post, _ = self._local_search(
            lambda teams: self._mean_scores(teams, m), best_obs)

        pool: List[List[int]] = []
        seen = set()

        def add(team):
            key = tuple(team)
            if key not in seen:
                seen.add(key)
                pool.append(list(team))

        add(best_post)
        add(best_obs)
        for t in self._neighbors(best_post):
            add(t)
        for t in self._neighbors(best_obs):
            if len(pool) >= self.POOL_SIZE - self.N_RANDOM_POOL:
                break
            add(t)
        # Cap the pool by the size of the space: on small problems there may
        # simply be fewer distinct teams than POOL_SIZE, so bound the attempts
        # rather than spinning looking for new ones.
        target = min(self.POOL_SIZE, self._space_size)
        attempts = 0
        while len(pool) < target and attempts < 20 * self.POOL_SIZE:
            add([int(np.random.randint(k)) for k in self.n])
            attempts += 1
        return pool[:self.POOL_SIZE]

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            m = self._posterior_mean()
            lam = self._noise_scale(m)          # measurement-noise variance
            pool = self._build_pool(m)

            Phi = self._dense_features(pool)     # (K, P)
            a = Phi @ m                          # posterior means of the pool
            # S = lam * A is the posterior covariance of the weights
            B = lam * (Phi @ (self.A @ Phi.T))   # (K, K), B[i,x] = phi_i' S phi_x

            best_x, best_kg = 0, -np.inf
            for x in range(len(pool)):
                denom = math.sqrt(max(lam + B[x, x], 1e-12))
                b_vec = B[:, x] / denom
                kg = _kg_from_lines(a, b_vec)
                if kg > best_kg:
                    best_kg, best_x = kg, x

            # KG of zero everywhere means nothing left to learn from the pool;
            # fall back to the posterior-mean maximizer rather than stalling.
            self.last_choice = list(pool[best_x]) if best_kg > 0 else list(pool[0])
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

        if self.t % self.EXACT_REFRESH == 0:
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
