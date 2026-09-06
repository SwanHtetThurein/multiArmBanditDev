"""COMBO with slice-sampled hyperparameters (the paper's actual inference).

Base method: Oh, Tomczak, Gavves & Welling, "Combinatorial Bayesian
Optimization using the Graph Cartesian Product" (NeurIPS 2019),
arXiv:1902.00448.
Sampler: Murray & Adams, "Slice Sampling Covariance Hyperparameters of Latent
Gaussian Models" (NeurIPS 2010), arXiv:1006.0868; univariate slice sampling
with stepping-out and shrinkage from Neal, "Slice Sampling" (Annals of
Statistics 31(3), 2003).

Why this file exists separately from combo.py:
    Our `combo.py` fits kernel hyperparameters by maximizing the marginal
    likelihood (type-II MAP) -- a simplification our own comparison report
    lists as a fairness caveat. COMBO's appendix 2.3 shows this is not a
    detail: the paper *marginalizes* its hyperparameters by slice sampling
    (100 burn-in iterations, then 10 samples per BO round, no thinning) and
    puts horseshoe priors on the per-dimension diffusion parameters.

    That matters most at exactly our budget. With 20-100 observations and ten
    kernel hyperparameters, a point estimate is badly overconfident: the GP
    commits early to one story about which dimensions matter, its posterior
    variance collapses, and Expected Improvement stops exploring. Averaging
    the acquisition over hyperparameter samples is what keeps it honest.

    Kept as a separate arm rather than an edit to combo.py so that running
    `combo` against `combo_slice` measures what the shortcut cost, instead of
    silently changing already-published results.

What is sampled:
    log beta_d for each dimension (the ARD diffusion parameters, horseshoe /
    half-Cauchy prior as in the paper), log signal variance (log-normal), and
    log noise variance (half-Cauchy). Each is updated by a univariate slice
    sampler with stepping-out and shrinkage, which needs no gradients, no step
    size tuning and no accept/reject tuning -- the property that makes it the
    right tool here.

Kernel (identical to combo.py, so the two arms differ only in inference):
    per-dimension diffusion kernel on the complete graph K_n,
        k_same = (1 + (n-1) e^{-n beta}) / n
        k_diff = (1 - e^{-n beta}) / n
    multiplied across dimensions (graph Cartesian product => the kernel
    factorizes), scaled by the signal variance.

Speed adaptations vs. the paper (documented for honesty in comparisons):
  - Burn-in is BURN_IN sweeps once, when the surrogate first fits, rather than
    100 iterations restarted every round; the chain then persists across
    rounds and is advanced SWEEPS_PER_REFIT sweeps every REFIT_EVERY rounds.
    Consecutive rounds differ by one observation, so the previous state is a
    good warm start.
  - N_HYPER hyperparameter samples are retained and the acquisition is
    averaged over them, against the paper's 10.
    These reduce cost by roughly an order of magnitude and are the reason a
    genuine MCMC treatment is affordable inside the full experiment sweep.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (BO random initialization).
  - predict_best() maximizes the hyperparameter-averaged posterior mean.
"""

import math
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z])


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


class ComboSlice(RecommendationAlgorithm):
    name = "combo_slice"

    N_INIT = 10            # random-exploration rounds before the GP takes over
    BURN_IN = 20           # slice-sampling sweeps at the first fit
    REFIT_EVERY = 10       # advance the chain every k rounds
    SWEEPS_PER_REFIT = 2   # sweeps per advance
    N_HYPER = 3            # hyperparameter samples kept for marginalization
    N_RESTARTS = 4         # local-search restarts for acquisition maximization
    JITTER = 1e-8
    BOUND = 5.0            # slice-sampling bounds on every log-hyperparameter

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.arm_counts = np.array(config.arm_counts)
        self.n = list(config.arm_counts)

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

        # current chain state: [log beta_1..D, log sf2, log sn2]
        self.theta = np.concatenate([np.zeros(self.D), [math.log(0.25)], [math.log(0.05)]])
        self._samples: List[np.ndarray] = []
        self._cache: List[dict] = []      # per-sample (chol, alpha, ymean, theta)
        self._started = False
        self._rounds_since_refit = 0

    # -- Kernel ----------------------------------------------------------------

    def _per_dim_kernels(self, log_beta):
        beta = np.exp(log_beta)
        n = self.arm_counts
        e = np.exp(-n * beta)
        k_same = (1 + (n - 1) * e) / n
        k_diff = (1 - e) / n
        return k_same, k_diff

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray, log_beta, log_sf2):
        k_same, k_diff = self._per_dim_kernels(log_beta)
        match = (A[:, None, :] == B[None, :, :])
        factors = np.where(match, k_same[None, None, :], k_diff[None, None, :])
        return np.exp(log_sf2) * np.prod(factors, axis=2)

    # -- Log posterior over hyperparameters ------------------------------------

    @staticmethod
    def _log_half_cauchy(log_x, scale=1.0):
        """Half-Cauchy(scale) density on x = e^{log_x}, plus the log Jacobian.
        This is the horseshoe-style heavy tail COMBO puts on its beta_d."""
        x = math.exp(log_x)
        return -math.log(1.0 + (x / scale) ** 2) + log_x

    def _log_posterior(self, theta: np.ndarray, X: np.ndarray, yc: np.ndarray) -> float:
        log_beta, log_sf2, log_sn2 = theta[:self.D], theta[self.D], theta[self.D + 1]
        n = len(yc)
        K = self._kernel_matrix(X, X, log_beta, log_sf2)
        K[np.diag_indices_from(K)] += math.exp(log_sn2) + self.JITTER
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        log_lik = (-0.5 * float(yc @ alpha) - float(np.log(np.diag(L)).sum())
                   - 0.5 * n * math.log(2 * math.pi))
        # priors: horseshoe-style on the ARD betas and the noise, log-normal on sf2
        log_prior = sum(self._log_half_cauchy(b) for b in log_beta)
        log_prior += -0.5 * (log_sf2 ** 2) / 4.0
        log_prior += self._log_half_cauchy(log_sn2, scale=0.5)
        return log_lik + log_prior

    # -- Univariate slice sampler (Neal 2003: stepping out + shrinkage) --------

    def _slice_once(self, i: int, theta: np.ndarray, X, yc,
                    w: float = 1.0, max_steps: int = 6) -> np.ndarray:
        def logf(v):
            cand = theta.copy()
            cand[i] = v
            return self._log_posterior(cand, X, yc)

        x0 = float(theta[i])
        y_level = logf(x0) - np.random.exponential()

        # stepping out
        left = x0 - w * np.random.random()
        right = left + w
        j = int(max_steps * np.random.random())
        k = max_steps - 1 - j
        while j > 0 and left > -self.BOUND and logf(left) > y_level:
            left -= w
            j -= 1
        while k > 0 and right < self.BOUND and logf(right) > y_level:
            right += w
            k -= 1
        left = max(left, -self.BOUND)
        right = min(right, self.BOUND)

        # shrinkage
        for _ in range(12):
            x1 = left + np.random.random() * (right - left)
            if logf(x1) > y_level:
                theta = theta.copy()
                theta[i] = x1
                return theta
            if x1 < x0:
                left = x1
            else:
                right = x1
        return theta

    def _sweep(self, X, yc) -> None:
        for i in range(len(self.theta)):
            self.theta = self._slice_once(i, self.theta, X, yc)

    def _advance_chain(self) -> None:
        X = np.array(self.X_teams)
        y = np.array(self.y)
        yc = y - y.mean()

        sweeps = self.SWEEPS_PER_REFIT
        if not self._started:
            sweeps = self.BURN_IN
            self._started = True

        for _ in range(sweeps):
            self._sweep(X, yc)
            self._samples.append(self.theta.copy())
        self._samples = self._samples[-self.N_HYPER:]
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Cache a Cholesky and alpha per retained hyperparameter sample."""
        X = np.array(self.X_teams)
        y = np.array(self.y)
        ymean = float(y.mean())
        yc = y - ymean
        self._cache = []
        for theta in self._samples:
            log_beta, log_sf2, log_sn2 = theta[:self.D], theta[self.D], theta[self.D + 1]
            K = self._kernel_matrix(X, X, log_beta, log_sf2)
            K[np.diag_indices_from(K)] += math.exp(log_sn2) + self.JITTER
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
            self._cache.append({"theta": theta, "L": L, "alpha": alpha, "ymean": ymean})

    def _posterior(self, teams: np.ndarray, entry: dict):
        X = np.array(self.X_teams)
        theta = entry["theta"]
        log_beta, log_sf2 = theta[:self.D], theta[self.D]
        Ks = self._kernel_matrix(np.asarray(teams), X, log_beta, log_sf2)
        mu = entry["ymean"] + Ks @ entry["alpha"]
        v = np.linalg.solve(entry["L"], Ks.T)
        var = math.exp(log_sf2) - np.einsum("ij,ij->j", v, v)
        return mu, np.sqrt(np.maximum(var, 1e-12))

    # -- Acquisition: EI averaged over hyperparameter samples ------------------

    @staticmethod
    def _ei(mu, sd, best):
        z = (mu - best) / sd
        return (mu - best) * _norm_cdf(z) + sd * _norm_pdf(z)

    def _marginal_ei(self, teams, best_y):
        teams = np.asarray(teams)
        total = np.zeros(len(teams))
        for entry in self._cache:
            mu, sd = self._posterior(teams, entry)
            total += self._ei(mu, sd, best_y)
        return total / max(len(self._cache), 1)

    def _marginal_mean(self, teams):
        teams = np.asarray(teams)
        total = np.zeros(len(teams))
        for entry in self._cache:
            mu, _ = self._posterior(teams, entry)
            total += mu
        return total / max(len(self._cache), 1)

    # -- Discrete maximization -------------------------------------------------

    def _neighbors(self, team) -> np.ndarray:
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return np.array(neigh)

    def _local_search(self, score_fn, start):
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
            self._rounds_since_refit += 1
            if (not self._started) or self._rounds_since_refit >= self.REFIT_EVERY:
                self._advance_chain()
                self._rounds_since_refit = 0
            elif len(self._cache) and self._cache[0]["L"].shape[0] != len(self.y):
                self._rebuild_cache()      # keep hypers, refresh on new data
            if not self._cache:
                self._rebuild_cache()
            best_y = float(np.max(self.y))
            self.last_choice = self._maximize(
                lambda teams: self._marginal_ei(teams, best_y))
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))

    def predict_best(self) -> List[int]:
        if not self._started or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        if not self._cache or self._cache[0]["L"].shape[0] != len(self.y):
            self._rebuild_cache()
        if not self._cache:
            return list(self.X_teams[int(np.argmax(self.y))])
        team, _ = self._local_search(self._marginal_mean,
                                     list(self.X_teams[int(np.argmax(self.y))]))
        return team
