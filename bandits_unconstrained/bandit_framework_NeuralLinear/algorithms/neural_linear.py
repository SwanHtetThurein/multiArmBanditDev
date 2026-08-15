"""NeuralLinear algorithm plug-in.

Based on: Riquelme, Tucker & Snoek, "Deep Bayesian Bandits Showdown: An
Empirical Comparison of Bayesian Deep Networks for Thompson Sampling"
(ICLR 2018), arXiv:1802.09127.

NeuralLinear trains a neural network to predict reward, but performs
Thompson sampling only on a Bayesian *linear* regression fitted to the
network's last hidden layer representation. The network provides a learned
nonlinear feature map; the exact conjugate linear-Gaussian posterior on top
of it provides cheap, well-calibrated exploration. In the paper's benchmark
this simple hybrid was among the most reliable methods, beating more
elaborate approximate-posterior deep methods.

Mapping to this framework:
  - Input encoding: concatenated one-hot vectors of the team (one block per
    bandit) -> sum(arm_counts) binary features.
  - Feature network: 2-hidden-layer MLP (ReLU), trained on all observed
    (team, reward) pairs by Adam on MSE, periodically (every TRAIN_EVERY
    rounds after warm-up), as in the paper's periodic retraining.
  - Bayesian last layer: Normal-Inverse-Gamma conjugate regression on
    phi(x) = [1, last-hidden-activations]. After each network retraining the
    sufficient statistics are recomputed from scratch on the new
    representation of ALL past data (the paper's key bookkeeping detail);
    between retrainings they are updated incrementally.
  - Acquisition: Thompson sample (sigma^2, w) from the NIG posterior and
    maximize w . phi(team) over teams by multi-start greedy local search
    over single-arm changes.

Implementation notes:
  - The MLP is implemented directly in numpy (manual backprop + Adam);
    no deep learning framework is required at this scale.
  - Round 1 plays the initial bias (comparable to DreamTeam); rounds
    2..N_INIT play random teams before the model takes over.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class NeuralLinearAlgorithm(RecommendationAlgorithm):
    name = "neurallinear"

    N_INIT = 10           # random-exploration rounds before the model takes over
    HIDDEN = (48, 24)     # MLP hidden layer sizes; last one is the representation
    TRAIN_EVERY = 10      # retrain the network every k rounds
    EPOCHS = 300          # full-batch Adam epochs per retraining
    LR = 1e-2
    WEIGHT_DECAY = 1e-4
    A0, B0 = 2.0, 0.05    # Inverse-Gamma prior on observation noise
    G_PRIOR = 4.0         # prior variance scale for last-layer weights
    N_RESTARTS = 4
    JITTER = 1e-9

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)
        self.in_dim = sum(self.n)
        self.offsets = np.concatenate([[0], np.cumsum(self.n)[:-1]])

        # ── MLP parameters (He init) ──────────────────────────────────────────
        h1, h2 = self.HIDDEN
        rng = np.random
        self.W1 = rng.normal(0, np.sqrt(2 / self.in_dim), (self.in_dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, np.sqrt(2 / h2), (h2, 1))
        self.b3 = np.zeros(1)
        self._adam_state = {}
        self._adam_t = 0

        # ── Bayesian last layer over phi = [1, h2 activations] ───────────────
        self.P = h2 + 1
        self.XtX = np.zeros((self.P, self.P))
        self.Xty = np.zeros(self.P)
        self.yty = 0.0

        self.X_teams: List[List[int]] = []
        self.y: List[float] = []
        self.last_choice = list(initial_bias)
        self._trained = False

    # ── Encoding and network forward/backward ────────────────────────────────

    def _encode(self, teams) -> np.ndarray:
        teams = np.asarray(teams)
        Z = np.zeros((len(teams), self.in_dim))
        cols = self.offsets[None, :] + teams
        Z[np.arange(len(teams))[:, None], cols] = 1.0
        return Z

    def _forward(self, Z):
        A1 = np.maximum(Z @ self.W1 + self.b1, 0)
        A2 = np.maximum(A1 @ self.W2 + self.b2, 0)
        out = (A2 @ self.W3 + self.b3).ravel()
        return A1, A2, out

    def _representation(self, teams) -> np.ndarray:
        """phi(team) = [1, last hidden activations]."""
        _, A2, _ = self._forward(self._encode(teams))
        return np.hstack([np.ones((len(A2), 1)), A2])

    def _adam_step(self, grads):
        self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, g in grads.items():
            p = getattr(self, name)
            g = g + self.WEIGHT_DECAY * p
            m, v = self._adam_state.get(name, (np.zeros_like(p), np.zeros_like(p)))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            self._adam_state[name] = (m, v)
            mhat = m / (1 - beta1 ** self._adam_t)
            vhat = v / (1 - beta2 ** self._adam_t)
            setattr(self, name, p - self.LR * mhat / (np.sqrt(vhat) + eps))

    def _train_network(self):
        Z = self._encode(self.X_teams)
        y = np.array(self.y)
        n = len(y)
        for _ in range(self.EPOCHS):
            A1, A2, pred = self._forward(Z)
            err = (pred - y) / n                      # d(MSE/2)/d pred
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
        self._trained = True
        # Representation changed -> recompute last-layer sufficient statistics
        # from scratch on all past data (Riquelme et al.'s bookkeeping).
        Phi = self._representation(self.X_teams)
        self.XtX = Phi.T @ Phi
        self.Xty = Phi.T @ y
        self.yty = float(y @ y)

    # ── Bayesian last layer ───────────────────────────────────────────────────

    def _posterior_factors(self):
        prec = self.XtX + np.eye(self.P) / self.G_PRIOR
        prec[np.diag_indices_from(prec)] += self.JITTER
        L = np.linalg.cholesky(prec)
        m = np.linalg.solve(L.T, np.linalg.solve(L, self.Xty))
        a_n = self.A0 + 0.5 * len(self.y)
        b_n = self.B0 + 0.5 * max(self.yty - m @ self.Xty, 0.0)
        return L, m, a_n, b_n

    def _sample_weights(self):
        L, m, a_n, b_n = self._posterior_factors()
        sigma2 = b_n / np.random.gamma(a_n)
        z = np.random.normal(size=self.P)
        return m + np.sqrt(sigma2) * np.linalg.solve(L.T, z)

    # ── Discrete maximization (multi-start greedy local search) ──────────────

    def _neighbors(self, team):
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return neigh

    def _local_search(self, w, start):
        current = list(start)
        current_score = float((self._representation([current]) @ w)[0])
        while True:
            neigh = self._neighbors(current)
            scores = self._representation(neigh) @ w
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current, current_score

    def _maximize(self, w):
        starts = [list(self.X_teams[int(np.argmax(self.y))])]
        for _ in range(self.N_RESTARTS - 1):
            starts.append([int(np.random.randint(k)) for k in self.n])
        best_team, best_score = None, -np.inf
        for s in starts:
            team, score = self._local_search(w, s)
            if score > best_score:
                best_team, best_score = team, score
        return best_team

    # ── RecommendationAlgorithm interface ─────────────────────────────────────

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= self.N_INIT:
            self.last_choice = [int(np.random.randint(k)) for k in self.n]
        else:
            if (not self._trained) or (len(self.y) % self.TRAIN_EVERY == 0):
                self._train_network()
            w = self._sample_weights()
            self.last_choice = self._maximize(w)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
        if self._trained:
            # incremental sufficient-statistic update between retrainings
            phi = self._representation([arms_chosen])[0]
            self.XtX += np.outer(phi, phi)
            self.Xty += reward * phi
            self.yty += reward * reward

    def predict_best(self) -> List[int]:
        if not self._trained or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        _, m, _, _ = self._posterior_factors()
        team, _ = self._local_search(m, list(self.X_teams[int(np.argmax(self.y))]))
        return team
