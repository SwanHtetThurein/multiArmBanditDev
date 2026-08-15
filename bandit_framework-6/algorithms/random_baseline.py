"""Random baseline — the smallest possible plug-in algorithm.

Included purely as a template/sanity-check: it demonstrates the minimum an
algorithm needs to implement, and gives a floor to compare real algorithms
against. Delete or ignore once real alternatives exist.
"""

import random
from typing import List

from .base import RecommendationAlgorithm, ProblemConfig


class RandomBaseline(RecommendationAlgorithm):
    name = "random"

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.last_choice = list(initial_bias)

    def choose(self, round_num: int) -> List[int]:
        self.last_choice = [random.randrange(n) for n in self.config.arm_counts]
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        pass  # learns nothing

    def predict_best(self) -> List[int]:
        return list(self.last_choice)
