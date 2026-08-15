"""NeuralUCB and NeuralTS algorithm plug-ins.

Based on:
  - Zhou, Li & Gu, "Neural Contextual Bandits with UCB-based Exploration"
    (ICML 2020), arXiv:1911.04462.  [NeuralUCB]
  - Zhang, Zhou, Li & Gu, "Neural Thompson Sampling" (ICLR 2021),
    arXiv:2010.00827.  [NeuralTS]

Both train a neural network f(x; theta) to predict reward and derive
exploration from the network's *gradient features* g(x) = grad_theta f(x):

  sigma^2(x) = g(x)^T  Z^{-1}  g(x),     Z = lambda*I + sum_t g(x_t) g(x_t)^T

  NeuralUCB: choose argmax_x  f(x) + nu * sigma(x)          (optimism)
  NeuralTS:  choose argmax_x  r~(x),  r~(x) ~ N(f(x), nu^2 * sigma^2(x))

Mapping to this framework:
  - x = concatenated one-hot encoding of the team (one block per bandit).
  - f is the same small MLP used by NeuralLinear (2 hidden ReLU layers),
    trained periodically on all observed (team, reward) pairs; the MLP core
    (encoding, forward pass, Adam training loop) is inherited from
    NeuralLinearAlgorithm.
  - Per-candidate gradient features are computed by vectorized manual
    backprop; the argmax over teams uses the same multi-start greedy local
    search over single-arm changes as the other plug-ins.

Practical simplifications (both noted in / consistent with the papers'
own experimental sections):
  - Diagonal approximation of Z (the NeuralUCB paper uses this for
    computational efficiency; the full matrix over ~2.4k parameters would
    dominate runtime at this experiment scale).
  - The network is retrained every TRAIN_EVERY rounds rather than every
    round; gradient features accumulate in Z across retrainings (theory
    fixes gradients at initialization, so this is comparably faithful).

Interface notes:
  - Round 1 plays the initial bias; rounds 2..N_INIT play random teams.
  - predict_best() maximizes the plain network prediction f(x) (no bonus)
    by local search from the best observed team.
"""

from typing import List

import numpy as np

from .neural_linear import NeuralLinearAlgorithm


class NeuralUCBAlgorithm(NeuralLinearAlgorithm):
    name = "neuralucb"

    LAMBDA = 1.0     # ridge regularizer on the design matrix Z
    NU = 0.5         # exploration strength

    def __init__(self, config, initial_bias, total_rounds):
        super().__init__(config, initial_bias, total_rounds)
        # number of MLP parameters (for gradient feature vectors)
        h1, h2 = self.HIDDEN
        self.n_params = (self.in_dim * h1 + h1) + (h1 * h2 + h2) + (h2 + 1)
        self.Z_diag = np.full(self.n_params, self.LAMBDA)

    # ── Gradient features ─────────────────────────────────────────────────────

    def _grad_features(self, teams) -> np.ndarray:
        """Per-candidate gradient of the network output w.r.t. all parameters.
        Returns (m, n_params), flattened in the order W1,b1,W2,b2,W3,b3."""
        Z = self._encode(teams)
        A1 = np.maximum(Z @ self.W1 + self.b1, 0)
        A2 = np.maximum(A1 @ self.W2 + self.b2, 0)
        m = len(teams)

        gW3 = A2                                                # (m, h2)
        gb3 = np.ones((m, 1))
        delta2 = (A2 > 0) * self.W3.ravel()[None, :]            # (m, h2)
        gb2 = delta2
        gW2 = (A1[:, :, None] * delta2[:, None, :]).reshape(m, -1)
        delta1 = (delta2 @ self.W2.T) * (A1 > 0)                # (m, h1)
        gb1 = delta1
        gW1 = (Z[:, :, None] * delta1[:, None, :]).reshape(m, -1)

        return np.hstack([gW1, gb1, gW2, gb2, gW3, gb3])

    def _predict_and_sigma(self, teams):
        _, _, mu = self._forward(self._encode(teams))
        G = self._grad_features(teams)
        sigma = np.sqrt(np.einsum("ij,ij->i", G, G / self.Z_diag[None, :]))
        return mu, sigma

    # ── Acquisition score (UCB; overridden by NeuralTS) ──────────────────────

    def _acquisition(self, teams) -> np.ndarray:
        mu, sigma = self._predict_and_sigma(teams)
        return mu + self.NU * sigma

    def _local_search_acq(self, start):
        current = list(start)
        current_score = float(self._acquisition([current])[0])
        while True:
            neigh = self._neighbors(current)
            scores = self._acquisition(neigh)
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current, current_score

    def _maximize_acq(self):
        starts = [list(self.X_teams[int(np.argmax(self.y))])]
        for _ in range(self.N_RESTARTS - 1):
            starts.append([int(np.random.randint(k)) for k in self.n])
        best_team, best_score = None, -np.inf
        for s in starts:
            team, score = self._local_search_acq(s)
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
            self.last_choice = self._maximize_acq()
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.X_teams.append(list(arms_chosen))
        self.y.append(float(reward))
        if self._trained:
            g = self._grad_features([arms_chosen])[0]
            self.Z_diag += g * g

    def predict_best(self) -> List[int]:
        if not self._trained or len(self.y) <= self.N_INIT:
            return (list(self.X_teams[int(np.argmax(self.y))])
                    if self.y else list(self.initial_bias))
        # maximize the plain prediction (no exploration bonus)
        current = list(self.X_teams[int(np.argmax(self.y))])
        current_score = float(self._forward(self._encode([current]))[2][0])
        while True:
            neigh = self._neighbors(current)
            scores = self._forward(self._encode(neigh))[2]
            j = int(np.argmax(scores))
            if scores[j] > current_score + 1e-12:
                current, current_score = list(neigh[j]), float(scores[j])
            else:
                return current


class NeuralTSAlgorithm(NeuralUCBAlgorithm):
    name = "neuralts"

    def _acquisition(self, teams) -> np.ndarray:
        mu, sigma = self._predict_and_sigma(teams)
        return np.random.normal(mu, self.NU * sigma)
