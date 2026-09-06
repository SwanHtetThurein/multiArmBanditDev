"""Bootstrapped neural-network ensemble plug-in.

Baseline in three of the papers this project follows:
  - Riquelme, Tucker & Snoek, "Deep Bayesian Bandits Showdown" (ICLR 2018),
    arXiv:1802.09127 ("BootstrappedNN").
  - Zhou, Li & Gu, "Neural Contextual Bandits with UCB-based Exploration"
    (NeuralUCB, arXiv:1911.04462).
  - Zhang, Xie, Zhang, Zhu & Gu, "Neural Thompson Sampling"
    (arXiv:2010.00827).

Method: Osband, Blundell, Pritzel & Van Roy, "Deep Exploration via
Bootstrapped DQN" (NIPS 2016), arXiv:1602.04621.

The idea is to replace an analytic posterior with an ensemble. K networks are
trained on bootstrap resamples of the same data; their disagreement in regions
with little data is the uncertainty estimate, and acting greedily w.r.t. one
uniformly-sampled ensemble member is an approximate Thompson-sampling step
("bootstrap Thompson sampling"). No variational inference, no kernel, no
last-layer conjugacy.

Contrast with neural_linear.py (worth stating explicitly in the paper):
    NeuralLinear gets its uncertainty from an exact Bayesian linear posterior
    on the *last hidden layer* of a single network -- the representation is
    treated as fixed and known. Bootstrapped NN instead gets uncertainty from
    variation across *independently trained networks*, so its uncertainty
    includes disagreement about the representation itself. The two are the
    standard cheap alternatives to full Bayesian deep learning, and they fail
    in different ways: NeuralLinear underestimates uncertainty when the
    representation is wrong, the bootstrap underestimates it when all members
    collapse to the same fit.

Implementation notes:
  - Input encoding: concatenated one-hot vectors of the team (one block per
    bandit), identical to neural_linear.py so the two are comparable.
  - Each observation is assigned a fixed Poisson(1) weight vector over the K
    members when it arrives (the online bootstrap), so member k trains on a
    weighted resample of the data and the resample is stable across
    retrainings rather than being redrawn each time.
  - The MLPs are implemented directly in numpy (manual backprop + Adam), as
    in neural_linear.py; no deep learning framework is required at this scale.
  - Networks are retrained every TRAIN_EVERY rounds after warm-up.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - Rounds 2..N_INIT play random teams before the ensemble takes over.
  - predict_best() maximizes the *ensemble mean* prediction by local search
    (no exploration bonus); before enough data exists it returns the best
    observed team.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class _MLP:
    """Small 2-hidden-layer ReLU regressor with manual backprop + Adam."""

    def __init__(self, in_dim, hidden, lr, weight_decay):
        h1, h2 = hidden
        self.W1 = np.random.normal(0, np.sqrt(2 / in_dim), (in_dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = np.random.normal(0, np.sqrt(2 / h1), (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = np.random.normal(0, np.sqrt(2 / h2), (h2, 1))
        self.b3 = np.zeros(1)
        self.lr = lr
        self.wd = weight_decay
        self._state = {}
        self._t = 0

    def forward(self, Z):
        A1 = np.maximum(Z @ self.W1 + self.b1, 0)
        A2 = np.maximum(A1 @ self.W2 + self.b2, 0)
        out = (A2 @ self.W3 + self.b3).ravel()
        return A1, A2, out

    def predict(self, Z):
        return self.forward(Z)[2]

    def _adam_step(self, grads):
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, g in grads.items():
            p = getattr(self, name)
            g = g + self.wd * p
            m, v = self._state.get(name, (np.zeros_like(p), np.zeros_like(p)))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            self._state[name] = (m, v)
            mhat = m / (1 - beta1 ** self._t)
            vhat = v / (1 - beta2 ** self._t)
            setattr(self, name, p - self.lr * mhat / (np.sqrt(vhat) + eps))

    def train(self, Z, y, w, epochs):
        """Weighted-MSE training; `w` holds this member's bootstrap weights."""
        wsum = max(float(w.sum()), 1e-9)
        for _ in range(epochs):
            A1, A2, pred = self.forward(Z)
            err = w * (pred - y) / wsum        # d(weighted MSE/2)/d pred
            gW3 = A2.T @ err[:, None]
            gb3 = np.array([err.sum()])
            dA2 = err[:, None] @ self.W3.T
            dA2[A2 <= 0] = 0
            gW2 = A1.T @ dA2
            gb2 = dA2.sum(axis=0)
            dA1 = dA2 @ self.W2.T
            dA1[A1 <= 0] = 0
            gW1 = Z.T @ dA1
            gb1 = dA1.sum(axis=0)
            self._adam_step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
                             "W3": gW3, "b3": gb3})


class BootstrappedNN(RecommendationAlgorithm):
    name = "bootnn"

    N_INIT = 10           # random-exploration rounds before the ensemble takes over
    N_MODELS = 5          # ensemble size
    HIDDEN = (32, 16)     # MLP hidden layer sizes
    TRAIN_EVERY = 10      # retrain every k rounds
    EPOCHS = 150          # full-batch Adam epochs per retraining
    LR = 1e-2
    WEIGHT_DECAY = 1e-4
    N_RESTARTS = 4

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)
        self.in_dim = sum(self.n)
        self.offsets = np.concatenate([[0], np.cumsum(self.n)[:-1]])

        self.models = [_MLP(self.in_dim, self.HIDDEN, self.LR, self.WEIGHT_DECAY)
                       for _ in range(self.N_MODELS)]

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.masks: List[np.ndarray] = []       # per-observation bootstrap weights
        self.last_choice = list(initial_bias)
        self._trained = False

    # -- Encoding -------------------------------------------------------------

    def _encode(self, teams) -> np.ndarray:
        teams = np.asarray(teams)
        Z = np.zeros((len(teams), self.in_dim))
        cols = self.offsets[None, :] + teams
        Z[np.arange(len(teams))[:, None], cols] = 1.0
        return Z

    # -- Training -------------------------------------------------------------

    def _train_all(self) -> None:
        Z = self._encode(self.X_teams)
        y = np.array(self.y)
        W = np.array(self.masks, dtype=float)          # (n, K)
        for k, model in enumerate(self.models):
            w = W[:, k]
            if w.sum() < 2:      # degenerate resample -> fall back to all data
                w = np.ones(len(y))
            model.train(Z, y, w, self.EPOCHS)
        self._trained = True

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
            if (not self._trained) or (len(self.y) % self.TRAIN_EVERY == 0):
                self._train_all()
            # bootstrap Thompson sampling: act greedily w.r.t. one random member
            member = self.models[int(np.random.randint(self.N_MODELS))]
            self.last_choice = self._maximize(
                lambda teams: member.predict(self._encode(teams)))
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
        # fixed online-bootstrap weights, drawn once when the point arrives
        self.masks.append(np.random.poisson(1.0, size=self.N_MODELS))

    def predict_best(self) -> List[int]:
        if not self._trained or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))

        def ensemble_mean(teams):
            Z = self._encode(teams)
            return np.mean([m.predict(Z) for m in self.models], axis=0)

        team, _ = self._local_search(ensemble_mean,
                                     list(self.X_teams[int(np.argmax(self.y))]))
        return team
