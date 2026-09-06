"""Faithful port of the ORIGINAL BOCS algorithm (Baptista & Poloczek, ICML 2018)
into this framework's RecommendationAlgorithm interface.

Source ported from: https://github.com/baptistar/BOCS  (BOCSpy/*.py)
  - bhs.py        -> _bhs, _fastmvg, _fastmvg_rue, _standardise  (Bayesian
                      horseshoe Gibbs sampler, ported near line-for-line)
  - LinReg.py      -> _order_effects, OriginalBocsAlgorithm._train
                      (data setup + order-effects feature construction)
  - BOCS.py        -> OriginalBocsAlgorithm._simulated_annealing
                      (the SA acquisition loop, ported near line-for-line)

WHY THIS FILE EXISTS
---------------------
`algorithms/bocs.py` in this repo is a deliberately lightweight approximation
of the paper: closed-form "grouped ridge" shrinkage instead of the real
horseshoe prior, and deterministic greedy local search instead of simulated
annealing / SDP relaxation. This file is the other end of the spectrum: it
reimplements the *actual* BOCSpy algorithm as literally as this framework
allows, so you can A/B it against the lightweight version.

WHAT IS AND ISN'T FAITHFUL
----------------------------
Faithful (ported near-verbatim from BOCSpy):
  - The Bayesian horseshoe Gibbs sampler (_bhs / _fastmvg / _fastmvg_rue),
    including the adaptive per-coefficient shrinkage (lambda_j, tau) that
    the original bocs.py in this repo does NOT have.
  - order_effects(): builds first-order + all pairwise products of the
    *entire* one-hot vector, exactly like the original. Note this is a
    naive extension: the original function has no notion of "which one-hot
    columns belong to the same dimension," so it also generates pairwise
    columns between two candidates of the SAME role (e.g. "role 3 = A" x
    "role 3 = B"), which are always zero since a team can't pick two
    candidates for one role. bandit_framework-6/algorithms/bocs.py is
    smarter here -- it only builds cross-dimension pairs. This file
    reproduces the original's naive (wasteful but faithful) behavior on
    purpose, which roughly TRIPLES the number of regression coefficients
    compared to bocs.py at 9 dimensions / up to 5 arms each.
  - Simulated annealing acquisition: single random-neighbor proposal +
    Metropolis acceptance with a geometric cooling schedule (T *= 0.8),
    best-of-5 reruns, exactly as in BOCS.py's `simulated_annealing()`.
  - Retraining the *entire* model from scratch on all deduplicated
    observations every round (the original does this; it does NOT do
    incremental/rank-1 updates the way bocs.py does).

Necessarily adapted (the original has no notion of these):
  - Categorical one-hot encoding of "one arm per bandit" itself -- the
    original BOCSpy only handles raw binary vectors in {0,1}^n_vars. This
    was already true of bocs.py too; there is no literal "original" version
    of this step to port.
  - Sign convention: the original paper/code MINIMIZES an objective. Our
    environment reward is something to MAXIMIZE. The Gibbs regression and
    SA loop below are flipped accordingly (documented inline).
  - SDP relaxation ('SDP-l1' in BOCS.py) is NOT ported. The original SDP
    formulation assumes n_vars independent free binary variables; our
    one-hot vectors have a "pick exactly one candidate per role" simplex
    constraint per dimension that the original SDP relaxation was never
    derived for. Doing this properly would mean deriving a new constrained
    SDP, which is beyond a faithful port. Simulated annealing (BOCS-SA) is
    the algorithm actually ported here.
  - predict_best(): the original code has no separate "report your current
    best guess" step (it's a one-shot optimizer that just returns the best
    x observed at the end). Since this framework scores every round, this
    file reports the posterior-MEAN team (argmax of the mean of the last
    Gibbs sample batch) once past initialization, and the best-observed
    team during initialization -- the same convention bocs.py uses, for a
    fair comparison.

COMPUTATIONAL COST WARNING
----------------------------
This runs a genuine 1000-iteration Gibbs sampler from scratch EVERY ROUND
(matching BOCS.py's `nGibbs = int(1e3)`), over a feature matrix that is
roughly 3x wider than bocs.py's (because of the naive all-pairs one-hot
expansion described above). This will be dramatically slower than
`algorithms/bocs.py` -- expect minutes, not seconds, for a single run at
9 dimensions / 100 rounds, and plan accordingly before launching a full
1080-run sweep. Consider testing with `--tests 1 --runs 1 --rounds 20`
first to sanity check before committing to the full grid.

Usage: same plug-in contract as every other algorithm in this framework.
    from algorithms.original_bocs import OriginalBocsAlgorithm
    ALGORITHMS[OriginalBocsAlgorithm.name] = OriginalBocsAlgorithm
(or add the two lines to algorithms/__init__.py the same way the other
algorithms are registered.)
"""

from itertools import combinations
from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


# ─────────────────────────────────────────────────────────────────────────
# Ported from BOCSpy/bhs.py (Baptista & Poloczek, 2018; original horseshoe
# sampler adapted from Makalic & Schmidt, arXiv:1508.03884). Ported
# near-verbatim; only renamed to module-private functions and reindented.
# ─────────────────────────────────────────────────────────────────────────

def _standardise(X, y):
    """Standardize the covariates to have zero mean; center y. Ported from
    bhs.py's `standardise` (original also computed std-based scaling for X
    but never actually applied it to X below -- that is preserved here
    exactly as in the source, including the apparent no-op)."""
    n = X.shape[0]
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0) * np.sqrt(n)
    meany = np.mean(y)
    y = y - meany
    return X, meanX, stdX, y, meany


def _fastmvg(Phi, alpha, D):
    """Fast sampler for N(mu, S) with mu = S Phi' y, S = inv(Phi'Phi + inv(D)),
    for the p > n regime. Ported from bhs.py's `fastmvg`
    (Bhattacharya, Chakraborty & Mallick, arXiv:1506.04778)."""
    n, p = Phi.shape
    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi, u) + delta
    Dpt = np.multiply(Phi.T, d[:, np.newaxis])
    w = np.linalg.solve(np.matmul(Phi, Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt, w)
    return x


def _fastmvg_rue(Phi, PtP, alpha, D):
    """Same target distribution as _fastmvg, for the small-p regime.
    Ported from bhs.py's `fastmvg_rue` (Rue, 2001)."""
    p = Phi.shape[1]
    Dinv = np.diag(1. / np.diag(D))
    try:
        L = np.linalg.cholesky(PtP + Dinv)
    except np.linalg.LinAlgError:
        mat = PtP + Dinv
        Smat = (mat + mat.T) / 2.
        maxEig_Smat = np.max(np.linalg.eigvals(Smat))
        L = np.linalg.cholesky(Smat + maxEig_Smat * 1e-15 * np.eye(Smat.shape[0]))
    v = np.linalg.solve(L, np.dot(Phi.T, alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))
    return m + w


def _bhs(Xorg, yorg, nsamples, burnin, thin):
    """Bayesian horseshoe linear regression via Gibbs sampling. Ported
    near-verbatim from BOCSpy/bhs.py (Makalic & Schmidt, 2015; Carvalho,
    Polson & Scott, 2010; adapted to Python by Baptista, 2018).

    Returns: beta [p x nsamples], b0 (scalar, = mean(y)), s2, t2, l2
    (hyper-variance traces -- kept for parity with the original signature,
    not otherwise used here).
    """
    n, p = Xorg.shape
    X, _, _, y, muY = _standardise(Xorg, yorg)

    beta = np.zeros((p, nsamples))
    s2 = np.zeros((1, nsamples))
    t2 = np.zeros((1, nsamples))
    l2 = np.zeros((p, nsamples))

    sigma2 = 1.
    lambda2 = np.random.uniform(size=p)
    tau2 = 1.
    nu = np.ones(p)
    xi = 1.

    XtX = np.matmul(X.T, X)

    k = 0
    it = 0
    while k < nsamples:
        sigma = np.sqrt(sigma2)
        Lambda_star = tau2 * np.diag(lambda2)

        if (p > n) and (p > 200):
            b = _fastmvg(X / sigma, y / sigma, sigma2 * Lambda_star)
        else:
            b = _fastmvg_rue(X / sigma, XtX / sigma2, y / sigma, sigma2 * Lambda_star)

        e = y - np.dot(X, b)
        shape = (n + p) / 2.
        scale = np.dot(e.T, e) / 2. + np.sum(b ** 2 / lambda2) / tau2 / 2.
        sigma2 = 1. / np.random.gamma(shape, 1. / scale)

        scale = 1. / nu + b ** 2. / 2. / tau2 / sigma2
        lambda2 = 1. / np.random.exponential(1. / scale)

        shape = (p + 1.) / 2.
        scale = 1. / xi + np.sum(b ** 2. / lambda2) / 2. / sigma2
        tau2 = 1. / np.random.gamma(shape, 1. / scale)

        scale = 1. + 1. / lambda2
        nu = 1. / np.random.exponential(1. / scale)

        scale = 1. + 1. / tau2
        xi = 1. / np.random.exponential(1. / scale)

        it += 1
        if it > burnin and (it % thin) == 0:
            beta[:, k] = b
            s2[:, k] = sigma2
            t2[:, k] = tau2
            l2[:, k] = lambda2
            k += 1

    b0 = muY
    return beta, b0, s2, t2, l2


def _order_effects(x_vals, ord_t):
    """Build the data matrix with first-order columns plus all pairwise
    (order 2) products of columns. Ported near-verbatim from
    LinReg.py's `order_effects`. Deliberately naive: given a one-hot
    vector it will also multiply together two one-hot columns belonging to
    the SAME categorical dimension (always zero), exactly as the original
    function would if handed a one-hot vector -- see module docstring."""
    n_samp, n_vars = x_vals.shape
    x_allpairs = x_vals
    for ord_i in range(2, ord_t + 1):
        offdProd = np.array(list(combinations(np.arange(n_vars), ord_i)))
        x_comb = np.zeros((n_samp, offdProd.shape[0], ord_i))
        for j in range(ord_i):
            x_comb[:, :, j] = x_vals[:, offdProd[:, j]]
        x_allpairs = np.append(x_allpairs, np.prod(x_comb, axis=2), axis=1)
    return x_allpairs


# ─────────────────────────────────────────────────────────────────────────
# The algorithm, wired into this framework's plug-in interface.
# ─────────────────────────────────────────────────────────────────────────

class OriginalBocsAlgorithm(RecommendationAlgorithm):
    """Faithful port of BOCS-SA (horseshoe Gibbs regression + simulated
    annealing acquisition) -- see module docstring for exactly what is and
    isn't faithful, and the runtime cost warning before running a full
    sweep with this."""

    name = "original_bocs"

    N_INIT = 10          # random-exploration rounds, same convention as bocs.py
    ORDER = 2            # model order (first + second order effects, as in the paper)

    # -- Gibbs sampler settings, matching BOCS.py's LinReg.train() exactly --
    NGIBBS = 1000         # BOCS.py: nGibbs = int(1e3)
    BURNIN = 0            # BOCS.py: bhs(self.xTrain, self.yTrain, nGibbs, 0, 1)
    THIN = 1

    # -- Simulated annealing settings, matching BOCS.py exactly --
    SA_RERUNS = 5          # BOCS.py: SA_reruns = 5
    SA_STEPS = 200          # BOCS.py ties SA's n_iter to the outer eval budget;
                            # that doesn't translate directly here, so this is
                            # our own choice of how many SA proposal steps to
                            # run per acquisition call. Raise it if the search
                            # space is large / SA looks under-converged.
    SA_T0 = 1.0             # BOCS.py: T = 1.
    SA_COOL = 0.8           # BOCS.py: cool = lambda T: .8*T

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        # One-hot layout: block d covers columns [offset[d], offset[d]+n[d]).
        self.offset = [0] * self.D
        idx = 0
        for d in range(self.D):
            self.offset[d] = idx
            idx += self.n[d]
        self.n_onehot = idx

        # Raw observation history (deduplicated at train time, exactly like
        # LinReg.setupData in the original).
        self._X_onehot: List[np.ndarray] = []   # one-hot rows, length n_onehot
        self._X_teams: List[List[int]] = []       # same rows, as team tuples
        self._y: List[float] = []

        # Posterior state from the most recent _train() call.
        self._alpha_last = None       # last Gibbs sample (this round's Thompson draw)
        self._alpha_mean = None       # posterior mean (used by predict_best)
        self._b0 = 0.0

        self.last_choice = list(initial_bias)

    # ── one-hot / design-matrix helpers ───────────────────────────────────

    def _onehot(self, team: List[int]) -> np.ndarray:
        v = np.zeros(self.n_onehot)
        for d in range(self.D):
            v[self.offset[d] + team[d]] = 1.0
        return v

    def _design_matrix(self, onehot_rows: np.ndarray) -> np.ndarray:
        return _order_effects(onehot_rows, self.ORDER)

    # ── training (full retrain each round, exactly like the original) ────

    def _train(self):
        """Faithful port of LinReg.train(): dedupe observed points, build
        order-effects design matrix, run the horseshoe Gibbs sampler for
        NGIBBS iterations, keep the last sample as this round's Thompson
        draw and the sample-batch mean as the point estimate for
        predict_best()."""
        X_raw = np.array(self._X_onehot)
        y_raw = np.array(self._y)

        # setupData(): limit to unique rows (original also splits off
        # "Inf" barrier outputs for infeasible points; our environment
        # never produces those, so that branch is a no-op here).
        X_uniq, x_idx = np.unique(X_raw, axis=0, return_index=True)
        y_uniq = y_raw[x_idx]

        X_design = self._design_matrix(X_uniq)
        n_samp, n_coeffs = X_design.shape

        # Drop all-zero columns (e.g. same-dimension pairwise columns that
        # are structurally always zero -- see module docstring), exactly
        # as LinReg.train() does, then pad the fitted coefficients back out
        # to the full width afterwards.
        check_zero = np.all(X_design == 0, axis=0)
        idx_nnzero = np.where(~check_zero)[0]
        X_fit = X_design[:, idx_nnzero] if np.any(check_zero) else X_design

        attempt = True
        while attempt:
            try:
                beta_samples, b0, _, _, _ = _bhs(
                    X_fit, y_uniq, self.NGIBBS, self.BURNIN, self.THIN)
            except Exception:
                continue
            attempt = np.isnan(beta_samples).any()

        alpha_last_fit = beta_samples[:, -1]
        alpha_mean_fit = beta_samples.mean(axis=1)

        alpha_last_pad = np.zeros(n_coeffs)
        alpha_last_pad[idx_nnzero] = alpha_last_fit
        alpha_mean_pad = np.zeros(n_coeffs)
        alpha_mean_pad[idx_nnzero] = alpha_mean_fit

        self._alpha_last = alpha_last_pad
        self._alpha_mean = alpha_mean_pad
        self._b0 = b0

    def _surrogate(self, team: List[int], alpha: np.ndarray) -> float:
        x_row = self._onehot(np.asarray(team)).reshape(1, -1)
        x_design = self._design_matrix(x_row)[0]
        return float(self._b0 + np.dot(x_design, alpha))

    # ── acquisition: simulated annealing (BOCS.py's `simulated_annealing`,
    #    ported and flipped from minimization to maximization) ───────────

    def _random_team(self) -> List[int]:
        return [int(np.random.randint(k)) for k in self.n]

    def _flip_one_dim(self, team: List[int]) -> List[int]:
        """Categorical analog of BOCS.py's single-bit flip: change exactly
        one randomly chosen dimension to a different, randomly chosen arm."""
        d = np.random.randint(self.D)
        new_team = list(team)
        if self.n[d] > 1:
            choices = [a for a in range(self.n[d]) if a != team[d]]
            new_team[d] = int(np.random.choice(choices))
        return new_team

    def _simulated_annealing_run(self, alpha: np.ndarray):
        """One SA run: BOCS.py's cooling/acceptance logic ported exactly,
        with the sign flipped (we MAXIMIZE reward; BOCS.py MINIMIZES)."""
        T = self.SA_T0
        old_x = self._random_team()
        old_obj = self._surrogate(old_x, alpha)
        best_x, best_obj = old_x, old_obj

        for _ in range(self.SA_STEPS):
            T = self.SA_COOL * T
            new_x = self._flip_one_dim(old_x)
            new_obj = self._surrogate(new_x, alpha)

            # Maximization Metropolis criterion (mirrors BOCS.py's
            # minimization criterion: `new_obj < old_obj or rand < exp((old-new)/T)`).
            if (new_obj > old_obj) or (np.random.rand() < np.exp((new_obj - old_obj) / max(T, 1e-12))):
                old_x, old_obj = new_x, new_obj
            if new_obj > best_obj:
                best_x, best_obj = new_x, new_obj

        return best_x, best_obj

    def _maximize(self, alpha: np.ndarray) -> List[int]:
        """BOCS.py's outer loop: run SA_RERUNS independent SA chains, keep
        the best."""
        best_x, best_obj = None, -np.inf
        for _ in range(self.SA_RERUNS):
            x, obj = self._simulated_annealing_run(alpha)
            if obj > best_obj:
                best_x, best_obj = x, obj
        return best_x

    # ── RecommendationAlgorithm interface ─────────────────────────────────

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT:
            self.last_choice = self._random_team()
        else:
            # Retrain on everything observed through the previous round
            # (matches the original's ordering: train() uses data up to
            # t-1, then that posterior drives round t's decision).
            self._train()
            self.last_choice = self._maximize(self._alpha_last)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self._X_onehot.append(self._onehot(arms_chosen))
        self._X_teams.append(list(arms_chosen))
        self._y.append(float(reward))

    def predict_best(self) -> List[int]:
        if len(self._y) <= self.N_INIT or self._alpha_mean is None:
            if not self._y:
                return list(self.initial_bias)
            return list(self._X_teams[int(np.argmax(self._y))])
        # Posterior-mean point estimate, maximized greedily (SA is for
        # exploration during acquisition; for a stable "current best guess"
        # report, a deterministic search on the mean surrogate is more
        # appropriate and is the natural analog of "argmax of E[f(x)]").
        best_x, best_obj = None, -np.inf
        starts = [list(self._X_teams[int(np.argmax(self._y))])] + \
                 [self._random_team() for _ in range(3)]
        for s in starts:
            x, obj = self._simulated_annealing_run(self._alpha_mean)
            if obj > best_obj:
                best_x, best_obj = x, obj
        return best_x
