"""Three acquisition rules on ONE shared Gaussian-process surrogate.

  gp_nei  -- Noisy Expected Improvement
             Letham, Karrer, Ottoni & Bakshy, "Constrained Bayesian
             Optimization with Noisy Experiments", Bayesian Analysis 14(2), 2019.
  gp_ucb  -- IGP-UCB (improved confidence widths)
  gp_ts   -- GP Thompson Sampling
             both from Chowdhury & Gopalan, "On Kernelized Multi-armed
             Bandits", ICML 2017, arXiv:1704.00445.

WHY THESE THREE LIVE IN ONE FILE
    `bocs`, `linucb`, `kg` and `purexp` already form a controlled comparison:
    one surrogate (the second-order Bayesian linear model), four acquisition
    rules, everything else held fixed. That design could only be run on the
    *linear* surrogate, because every GP arm in the suite (`combo`,
    `combo_slice`, `gp_onehot`, `cocabo`) uses Expected Improvement and
    nothing else.

    These three complete the same grid on a Gaussian process. Together with
    `gp_onehot` (plain EI) they give:

        gp_onehot  Expected Improvement          (noise-free formula)
        gp_nei     Noisy Expected Improvement    (incumbent uncertainty integrated out)
        gp_ucb     upper confidence bound        (optimism)
        gp_ts      Thompson sampling             (posterior sampling)

    all on the identical kernel, the identical hyperparameter fit and the
    identical discrete optimizer. Being able to make the acquisition claim on
    two different surrogates rather than one is a materially stronger
    experimental design -- it separates "this acquisition rule is better" from
    "this acquisition rule happens to suit a linear model".

THE SHARED SURROGATE
    Identical to `gp_onehot.py`: a squared-exponential GP on the one-hot
    encoding, which collapses exactly to

        k(x, x') = sigma_f^2 * exp( -hamming(x, x') / rho )

    (an algebraic identity, not an approximation -- see gp_onehot.py for the
    derivation), with the signal variance profiled out in closed form and
    (rho, eta) fit by grid search on the concentrated log marginal likelihood.
    Deliberately the same surrogate as `gp_onehot` so that arm serves as the
    plain-EI member of the family.

WHY NOISY EI MATTERS HERE SPECIFICALLY
    Standard EI compares candidates against `max(observed y)`. That is correct
    only when observations are noise-free. This environment's noise is
    multiplicative -- `std = noise_level * p` -- so the best observed reward is
    systematically an *overestimate*: it is the maximum of a set of noisy
    draws, and the noisiest draws are exactly the ones from good teams. EI
    against that inflated incumbent under-explores.

    Letham et al.'s fix is to treat the incumbent as unknown and integrate
    over it. `gp_nei` draws joint samples of the true function values at the
    observed points from the GP posterior, takes the maximum of each sample as
    that scenario's incumbent, and averages EI across scenarios. `combo`,
    `smac` and `gp_onehot` all use the noise-free formula, so this arm also
    measures how much that shortcut costs.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams (BO random initialization).
  - predict_best() maximizes the posterior mean by local search in all three.
"""

import math
from typing import List, Optional

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


def _norm_cdf(z) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in np.atleast_1d(z)])


def _norm_pdf(z) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


class _OneHotGP(RecommendationAlgorithm):
    """Shared surrogate. Subclasses implement `_acquire()` only."""

    N_INIT = 10          # random-exploration rounds before the GP takes over
    REFIT_EVERY = 10     # re-tune hyperparameters every k rounds
    N_RESTARTS = 4       # local-search restarts for acquisition maximization
    JITTER = 1e-8

    RHO_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    ETA_GRID = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)

        self.rho = 3.0
        self.eta = 0.1
        self.sf2 = 0.25

        self._chol: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._ymean = 0.0
        self._rounds_since_tune = 0

    # -- Kernel and fitting (identical to gp_onehot.py) -----------------------

    @staticmethod
    def _hamming(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A[:, None, :] != B[None, :, :]).sum(axis=2).astype(float)

    def _tune_hypers(self, H: np.ndarray, yc: np.ndarray) -> None:
        n = len(yc)
        best = None
        for rho in self.RHO_GRID:
            R = np.exp(-H / rho)
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
                nll = 0.5 * n * math.log(sf2) + float(np.log(np.diag(L)).sum()) + 0.5 * n
                if best is None or nll < best[0]:
                    best = (nll, rho, eta, sf2)
        if best is not None:
            _, self.rho, self.eta, self.sf2 = best

    def _refresh(self, tune: bool) -> None:
        X = np.array(self.X_teams)
        y = np.array(self.y)
        n = len(y)
        self._ymean = float(y.mean())
        yc = y - self._ymean

        H = self._hamming(X, X)
        if tune and n >= 5:
            self._tune_hypers(H, yc)

        M = np.exp(-H / self.rho) + (self.eta + self.JITTER) * np.eye(n)
        try:
            self._chol = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            M[np.diag_indices_from(M)] += 1e-4
            self._chol = np.linalg.cholesky(M)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, yc))

    def _posterior(self, teams):
        """Marginal posterior mean and std at `teams`."""
        X = np.array(self.X_teams)
        Rs = np.exp(-self._hamming(np.asarray(teams), X) / self.rho)
        mu = self._ymean + Rs @ self._alpha
        v = np.linalg.solve(self._chol, Rs.T)
        var = self.sf2 * (1.0 - np.einsum("ij,ij->j", v, v))
        return mu, np.sqrt(np.maximum(var, 1e-12))

    def _posterior_joint(self, teams):
        """Joint posterior mean and covariance at `teams` (for GP-TS / NEI)."""
        X = np.array(self.X_teams)
        T = np.asarray(teams)
        Rs = np.exp(-self._hamming(T, X) / self.rho)
        mu = self._ymean + Rs @ self._alpha
        v = np.linalg.solve(self._chol, Rs.T)                 # (n, m)
        Ktt = self.sf2 * np.exp(-self._hamming(T, T) / self.rho)
        cov = Ktt - self.sf2 * (v.T @ v)
        cov[np.diag_indices_from(cov)] += 1e-9
        return mu, cov

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

    def _acquire(self) -> List[int]:
        raise NotImplementedError

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT or self._chol is None:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            self.last_choice = self._acquire()
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
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


# ══════════════════════════════════════════════════════════════════════════════

class GPNoisyEI(_OneHotGP):
    """Noisy Expected Improvement (Letham et al., 2019).

    Standard EI uses `best = max(observed y)`, which is only correct when
    observations are noise-free. Here the incumbent is itself uncertain -- and
    biased upward, since it is the maximum of noisy draws and our noise grows
    with the reward.

    NEI integrates the incumbent out. Draw S joint samples of the *true*
    function values at the observed points from the GP posterior; each sample
    is one plausible account of what actually happened. Take that sample's
    maximum as the incumbent for that scenario, compute EI against it, and
    average across scenarios.

    Simplification vs. the paper (documented for honesty): Letham et al.
    re-condition the GP on each sampled set of noiseless values before
    computing EI, giving a different posterior per scenario. We keep the
    single noise-conditioned posterior and marginalize only the *incumbent*.
    That captures the correction this environment actually needs -- an
    over-trusted best-so-far -- at 1/S of the cost, which matters inside a
    1080-run sweep. The full version would additionally sharpen the posterior.
    """

    name = "gp_nei"

    N_SCENARIOS = 24     # quasi-MC samples of the observed-point posterior

    def _incumbent_samples(self) -> np.ndarray:
        """S plausible values of max f over the observed points."""
        mu, cov = self._posterior_joint(np.array(self.X_teams))
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(mu)))
        Z = np.random.normal(size=(len(mu), self.N_SCENARIOS))
        draws = mu[:, None] + L @ Z              # (n, S)
        return draws.max(axis=0)                 # (S,)

    def _acquire(self) -> List[int]:
        bests = self._incumbent_samples()

        def nei(teams):
            mu, sd = self._posterior(teams)
            total = np.zeros(len(mu))
            for b in bests:
                z = (mu - b) / sd
                total += (mu - b) * _norm_cdf(z) + sd * _norm_pdf(z)
            return total / len(bests)

        return self._maximize(nei)


class GPUCB(_OneHotGP):
    """IGP-UCB (Chowdhury & Gopalan, ICML 2017).

    score(x) = mu(x) + beta_t * sd(x)

    Chowdhury & Gopalan tightened the confidence width of the original GP-UCB
    (Srinivas et al., 2010) via a self-normalized concentration bound, which is
    why this is the version implemented rather than the 2010 one. For a finite
    domain the width has the form

        beta_t = sqrt( 2 * log( |D| * t^2 * pi^2 / (6 * delta) ) )

    and |D| here is genuinely computable -- it is the number of teams -- so the
    schedule is used as written rather than replaced by a constant.

    One documented deviation: the schedule is multiplied by BETA_SCALE. The
    theory-prescribed width is famously over-conservative in practice; NeuralTS's
    own experiments needed an exploration coefficient two to five orders of
    magnitude below the theoretical value. At |D| ~ 10^5 the unscaled beta_t is
    above 6, which would swamp a reward that lives in [0, 1]. BETA_SCALE makes
    that adjustment explicit and tunable instead of hidden.
    """

    name = "gp_ucb"

    DELTA = 0.1
    BETA_SCALE = 0.2

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        log_size = float(np.sum(np.log(np.array(self.n, dtype=float))))
        self._log_domain = log_size          # log |D|

    def _beta(self, t: int) -> float:
        t = max(t, 2)
        inner = (self._log_domain + 2.0 * math.log(t)
                 + 2.0 * math.log(math.pi) - math.log(6.0 * self.DELTA))
        return self.BETA_SCALE * math.sqrt(max(2.0 * inner, 0.0))

    def _acquire(self) -> List[int]:
        beta = self._beta(len(self.y))

        def ucb(teams):
            mu, sd = self._posterior(teams)
            return mu + beta * sd

        return self._maximize(ucb)


class GPThompson(_OneHotGP):
    """GP Thompson Sampling (Chowdhury & Gopalan, ICML 2017).

    Draw one sample of the objective from the GP posterior and play its
    maximizer -- the GP counterpart of what `bocs` does with a linear model.

    A GP sample is a *function*, and drawing one consistently over a 10^5-team
    space is not possible directly. Instead we draw an exact joint sample over
    a candidate pool: the incumbent, the posterior-mean maximizer, their
    single-arm neighbourhoods, and random teams. Within that pool the sample is
    exact (a joint draw from the posterior covariance, not independent marginal
    draws -- which would be a different and much more erratic algorithm), and
    the pool always contains the incumbent so the play can never be worse than
    a local move away from it.

    The variance inflation factor is left at 1. Agrawal & Goyal's analysis
    prescribes inflating the posterior for linear TS, but Abeille & Lazaric
    (2017) showed the guarantee tolerates a wide family of sampling
    distributions, and every practical study in this bibliography shrinks
    rather than inflates it at short horizons.
    """

    name = "gp_ts"

    POOL_SIZE = 120
    N_RANDOM_POOL = 30

    def _build_pool(self) -> List[List[int]]:
        best_obs = list(self.X_teams[int(np.argmax(self.y))])

        def mean_score(teams):
            mu, _ = self._posterior(teams)
            return mu

        best_post, _ = self._local_search(mean_score, best_obs)

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
            add([int(v) for v in t])
        for t in self._neighbors(best_obs):
            if len(pool) >= self.POOL_SIZE - self.N_RANDOM_POOL:
                break
            add([int(v) for v in t])

        attempts = 0
        while len(pool) < self.POOL_SIZE and attempts < 20 * self.POOL_SIZE:
            add([int(np.random.randint(k)) for k in self.n])
            attempts += 1
        return pool[:self.POOL_SIZE]

    def _acquire(self) -> List[int]:
        pool = self._build_pool()
        mu, cov = self._posterior_joint(np.array(pool))
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(mu)))
        sample = mu + L @ np.random.normal(size=len(mu))
        return list(pool[int(np.argmax(sample))])
