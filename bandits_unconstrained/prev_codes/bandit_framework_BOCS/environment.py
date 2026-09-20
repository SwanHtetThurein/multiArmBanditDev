"""Environment: owns the hidden optimal-arm vector and the reward function.

Kept separate from the algorithms so that (a) algorithms can never peek at
the answer, and (b) alternative reward structures can be swapped in later
without touching any algorithm.
"""

from typing import List

import numpy as np


class TeamRewardEnvironment:
    """Reward = fraction of dimensions chosen correctly, with multiplicative
    Gaussian noise (std = noise * reward), clipped to [0, 1].

    Identical to reward_generator in the original script.
    """

    def __init__(self, optimal_arm: List[int], noise: float):
        self.optimal_arm = list(optimal_arm)
        self.noise = noise

    def reward(self, arms_chosen: List[int]) -> float:
        mismatches = sum(1 for chosen, opt in zip(arms_chosen, self.optimal_arm)
                         if chosen != opt)
        p = 1 - (mismatches / len(arms_chosen))
        std = self.noise * p
        noisy_reward = np.random.normal(p, std)
        return float(np.clip(noisy_reward, 0, 1))

    def evaluate_prediction(self, predicted_best: List[int]) -> int:
        """Number of dimensions where the algorithm's best guess is correct.
        Used by the harness for the per-round performance metric."""
        return sum(1 for pred, opt in zip(predicted_best, self.optimal_arm)
                   if pred == opt)
