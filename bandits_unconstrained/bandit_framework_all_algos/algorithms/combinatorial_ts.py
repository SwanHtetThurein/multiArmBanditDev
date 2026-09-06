"""CTS -- Combinatorial Thompson Sampling.

Based on: Wang & Chen, "Thompson Sampling for Combinatorial Semi-Bandits"
(ICML 2018). The Thompson-sampling counterpart of CUCB (see cucb.py), and the
first CMAB analysis showing TS matches UCB-style regret for combinatorial
action sets with an exact oracle.

Mapping to this framework:
  base arm  = one (bandit, arm) pair
  super arm = a full team (one base arm per bandit)
  oracle    = per-bandit argmax of the sampled values (exact, because the
              objective is separable across dimensions)

Algorithm:
  - Keep a Beta(alpha, beta) posterior for every base arm, starting Beta(1,1).
  - Each round sample theta[d][a] ~ Beta(alpha, beta) independently and call
    the oracle on the sampled values.
  - Update the posteriors of the played base arms from the observed reward.

Two adaptation notes -- both matter for the writeup:

1. Feedback model. CTS, like CUCB, assumes *semi-bandit* feedback (each played
   base arm's outcome is observed separately). Our environment returns one
   joint scalar per team, so that reward is credited to every base arm in the
   played team. Same departure as cucb.py, documented there at length.

2. Bounded rewards. The Beta posterior is conjugate to Bernoulli outcomes, but
   our rewards are continuous in [0, 1]. We use the standard Agrawal & Goyal
   (2012) device: on observing reward r, draw a single Bernoulli(r) and update
   alpha += 1 on a success, beta += 1 on a failure. This keeps the posterior
   exactly conjugate and is the construction CTS's analysis assumes for
   [0, 1]-valued rewards -- not an approximation.

Contrast with DreamTeam (worth stating explicitly in the paper):
    DreamTeam also keeps a Beta posterior per (bandit, arm) and also feeds it
    the joint reward. CTS is what that design looks like when done by the
    book: an unbiased Bernoulli update instead of the "+reward on success,
    +1000 to beta on failure" rule, and no stickiness schedule or global
    switching budget. It is therefore the cleanest available control for
    isolating what DreamTeam's bespoke machinery actually contributes.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam). No forced
    exploration sweep is needed -- the Beta(1,1) prior handles cold starts.
  - predict_best() returns the per-bandit argmax of the posterior means.
"""

from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class CombinatorialTS(RecommendationAlgorithm):
    name = "cts"

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        # Beta(1, 1) prior on every base arm
        self.alpha = [np.ones(k) for k in self.n]
        self.beta = [np.ones(k) for k in self.n]
        self.t = 0
        self.last_choice = list(initial_bias)

    # -- helpers --------------------------------------------------------------

    def _oracle(self, scores: List[np.ndarray]) -> List[int]:
        """Exact separable oracle: best arm per bandit independently."""
        return [int(np.argmax(scores[d])) for d in range(self.D)]

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.last_choice = list(self.initial_bias)
        else:
            sampled = [np.random.beta(self.alpha[d], self.beta[d])
                       for d in range(self.D)]
            self.last_choice = self._oracle(sampled)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self.t += 1
        # Agrawal & Goyal Bernoulli device keeps the Beta posterior conjugate
        # for a reward in [0, 1].
        r = min(max(float(reward), 0.0), 1.0)
        success = np.random.random() < r
        for d, a in enumerate(arms_chosen):
            if success:
                self.alpha[d][a] += 1.0
            else:
                self.beta[d][a] += 1.0

    def predict_best(self) -> List[int]:
        if self.t == 0:
            return list(self.initial_bias)
        means = [self.alpha[d] / (self.alpha[d] + self.beta[d])
                 for d in range(self.D)]
        return self._oracle(means)
