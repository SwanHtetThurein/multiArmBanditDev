"""CUCB -- Combinatorial Upper Confidence Bound.

Based on: Chen, Wang & Yuan, "Combinatorial Multi-Armed Bandit: General
Framework, Results and Applications" (ICML 2013), and the extended
"Combinatorial Multi-Armed Bandit and Its Extension to Probabilistically
Triggered Arms" (JMLR 2016).

This is the formal framework for exactly our problem shape -- choose one
configuration out of a product space -- and it is the family none of the five
source papers (BOCS, COMBO, NeuralLinear, NeuralUCB, NeuralTS) engage with at
all.

Mapping to this framework:
  base arm    = one (bandit, arm) pair, i.e. one candidate for one role
  super arm   = a full team (exactly one base arm per bandit)
  oracle      = argmax over super arms of the summed base-arm estimates.
                Because that objective is separable across dimensions, the
                oracle here is *exact and instant*: independently take the
                best-scoring arm in each bandit. CUCB's theory only assumes an
                (alpha, beta)-approximation oracle, so an exact one is the
                best case its guarantee allows.

Algorithm:
  - Maintain an empirical mean mu_hat and a play count T for every base arm.
  - Each round form the optimistic estimate
        mu_bar[d][a] = mu_hat[d][a] + sqrt(3 * ln(t) / (2 * T[d][a]))
    (the confidence radius from the paper) and call the oracle on it.
  - After observing the reward, update every base arm in the played team.

Adaptation note -- IMPORTANT for the writeup:
    CUCB assumes *semi-bandit* feedback: the outcome of each played base arm
    is observed individually. Our environment returns only ONE joint scalar
    for the whole team, so we credit that joint reward to every base arm in
    the played team. This is a real departure from the paper's feedback model
    and it is the same credit-assignment shortcut DreamTeam makes -- which is
    precisely what makes CUCB an interesting comparison: it is the principled,
    theoretically-analysed version of "per-dimension estimates updated from a
    joint reward", with an optimism bonus in place of DreamTeam's stickiness
    schedule and switching budget.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam).
  - The next max(arm_counts) rounds sweep every base arm at least once, which
    CUCB requires before its confidence radii are defined.
  - predict_best() returns the per-bandit argmax of the empirical means (no
    optimism bonus).
"""

import math
from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class CUCBAlgorithm(RecommendationAlgorithm):
    name = "cucb"

    #: Coefficient inside the confidence radius sqrt(C * ln t / (2 T)).
    RADIUS_C = 3.0

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        self.sums = [np.zeros(k) for k in self.n]
        self.counts = [np.zeros(k) for k in self.n]
        self.t = 0
        self.last_choice = list(initial_bias)

        #: rounds 2..(1 + INIT_SWEEP) force every base arm to be played once
        self.INIT_SWEEP = max(self.n)

    # -- helpers --------------------------------------------------------------

    def _means(self) -> List[np.ndarray]:
        out = []
        for d in range(self.D):
            c = np.maximum(self.counts[d], 1.0)
            out.append(self.sums[d] / c)
        return out

    def _oracle(self, scores: List[np.ndarray]) -> List[int]:
        """Exact separable oracle: best arm per bandit independently."""
        return [int(np.argmax(scores[d])) for d in range(self.D)]

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        elif round_num <= 1 + self.INIT_SWEEP:
            # round-robin sweep so every base arm gets at least one play
            step = round_num - 2
            self.last_choice = [step % self.n[d] for d in range(self.D)]
        else:
            means = self._means()
            log_t = math.log(max(self.t, 2))
            optimistic = []
            for d in range(self.D):
                radius = np.sqrt(self.RADIUS_C * log_t
                                 / (2.0 * np.maximum(self.counts[d], 1.0)))
                # a never-played arm is infinitely optimistic
                radius = np.where(self.counts[d] > 0, radius, np.inf)
                optimistic.append(means[d] + radius)
            self.last_choice = self._oracle(optimistic)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.t += 1
        # joint reward credited to every base arm in the played super arm
        for d, a in enumerate(arms_chosen):
            self.sums[d][a] += reward
            self.counts[d][a] += 1.0

    def predict_best(self) -> List[int]:
        if self.t == 0:
            return list(self.initial_bias)
        return self._oracle(self._means())
