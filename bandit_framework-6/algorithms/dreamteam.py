"""DreamTeam algorithm: Thompson sampling with per-bandit switching schedules
('early'/'late'/'ongoing') and a global constraint on total switching mass.

This is the algorithm from the original new_global_Dynamic_Parameters.py,
refactored to the RecommendationAlgorithm interface. The decision logic and
update rules are preserved exactly:

  - Beta(1,1) priors per arm; Thompson-sample and normalize each round.
  - posterior_normalization: sigmoid-scheduled stickiness toward the currently
    held arm, depending on the bandit's type.
  - global_constraint: total off-current probability mass across all bandits
    is capped by an inverted-parabola budget peaking mid-run (m=2).
  - Round 1 always plays the initial bias.
  - Update: if reward > 0.1, alpha += reward on every chosen arm;
    otherwise beta += 1000 on every chosen arm.
"""

import math
from typing import List

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig


class DreamTeamAlgorithm(RecommendationAlgorithm):
    name = "dreamteam"

    #: Reward threshold below which the harsh beta penalty is applied.
    PENALTY_THRESHOLD = 0.1
    #: Beta increment applied on failure.
    PENALTY_BETA = 1
    #: Peak of the global switching budget (parameter m in the original).
    GLOBAL_BUDGET_PEAK = 2

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        # Beta parameters: self.betas[bandit][arm] = [alpha, beta]
        self.betas = [[[1.0, 1.0] for _ in range(n)] for n in config.arm_counts]
        self.arms_chosen = list(initial_bias)

    # ── Internal helpers (identical math to the original) ────────────────────

    @staticmethod
    def _beta_mean(arm) -> float:
        alpha, beta = arm
        return alpha / (alpha + beta)

    @staticmethod
    def _normalize(values) -> List[float]:
        total = sum(values)
        return [v / total for v in values]

    def _posterior_normalization(self, current_arm, nbv, bandit_type, current_round):
        if current_arm is None:
            return nbv
        pbv = nbv[:]
        half = self.total_rounds / 2
        if bandit_type == 'early':
            delta = 1 / (1 + math.e ** (current_round - half))
        elif bandit_type == 'late':
            delta = 1 / (1 + math.e ** (half - current_round))
        else:
            delta = 1

        accumulator = 0
        for i in range(len(pbv)):
            if i != current_arm:
                accumulator += pbv[i] * (1 - delta)
                pbv[i] = pbv[i] * delta
        pbv[current_arm] += accumulator
        return pbv

    def _global_constraint(self, pre, current_round):
        post = [[b[0], b[1][:]] for b in pre]
        half = self.total_rounds / 2
        y = self.GLOBAL_BUDGET_PEAK * (1 - (((current_round - half) / half) ** 2))

        zd_list = []
        for bandit in pre:
            curr_arm = bandit[0]
            off_mass = sum(bandit[1]) - bandit[1][curr_arm]
            zd_list.append(off_mass)

        z = sum(zd_list)
        if z <= y:
            return post

        scale = y / z
        for ind, bandit in enumerate(pre):
            curr_arm = bandit[0]
            moved_prob = 0
            for arm_idx, prob in enumerate(bandit[1]):
                if arm_idx != curr_arm:
                    new_prob = prob * scale
                    moved_prob += prob - new_prob
                    post[ind][1][arm_idx] = new_prob
            post[ind][1][curr_arm] += moved_prob
        return post

    # ── RecommendationAlgorithm interface ─────────────────────────────────────

    def choose(self, round_num: int) -> List[int]:
        # Thompson sample + normalize per bandit
        all_normed = []
        for b_idx, n_arms in enumerate(self.config.arm_counts):
            beta_vals = [np.random.beta(a, b) for a, b in self.betas[b_idx]]
            all_normed.append(self._normalize(beta_vals))

        # Per-bandit stickiness schedule
        all_posterior = []
        for b_idx, normed in enumerate(all_normed):
            btype = self.config.bandit_types[b_idx] if self.config.bandit_types else 'ongoing'
            all_posterior.append(
                self._posterior_normalization(self.arms_chosen[b_idx], normed, btype, round_num)
            )

        pre = [[self.arms_chosen[b_idx], all_posterior[b_idx]]
               for b_idx in range(self.config.n_bandits)]

        if round_num == 1:
            # First round always plays the initial bias (as in the original).
            self.arms_chosen = list(self.initial_bias)
        else:
            post = self._global_constraint(pre, round_num)
            for b_idx, n_arms in enumerate(self.config.arm_counts):
                self.arms_chosen[b_idx] = np.random.choice(range(n_arms), p=post[b_idx][1])

        return list(self.arms_chosen)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        for b_idx, arm in enumerate(arms_chosen):
            if reward > self.PENALTY_THRESHOLD:
                self.betas[b_idx][arm][0] += reward
            else:
                self.betas[b_idx][arm][1] += self.PENALTY_BETA

    def predict_best(self) -> List[int]:
        best = []
        for b_idx in range(self.config.n_bandits):
            means = [self._beta_mean(arm) for arm in self.betas[b_idx]]
            best.append(int(np.argmax(means)))
        return best
