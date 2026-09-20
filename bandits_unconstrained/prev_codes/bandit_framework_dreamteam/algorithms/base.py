"""Base interface for recommendation algorithms.

Any algorithm that implements this interface can be plugged into the
experiment harness (experiment.py) without changing the testing structure.

The contract per run:
    1. The harness constructs the algorithm with a ProblemConfig,
       an initial_bias vector, and the total number of rounds.
    2. Each round, the harness calls choose(round_num) to get one arm
       per bandit (a list of ints, one index per dimension).
    3. The harness computes the (noisy) reward from the environment and
       feeds it back via update(arms_chosen, reward).
    4. For evaluation, the harness calls predict_best() to get the
       algorithm's current best-guess arm per bandit. Performance is the
       fraction of dimensions where predict_best() matches the hidden
       optimal arm.

Algorithms never see the optimal arm or the noise level — only rewards.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProblemConfig:
    """Static description of the multi-dimensional bandit problem.

    arm_counts:   number of arms for each bandit, e.g. [3, 2, 5, ...]
    bandit_types: per-bandit metadata string, e.g. 'early'/'late'/'ongoing'.
                  Algorithms that don't use type information may ignore it.
    """
    arm_counts: List[int]
    bandit_types: List[str] = field(default_factory=list)

    @property
    def n_bandits(self) -> int:
        return len(self.arm_counts)

    def __post_init__(self):
        if self.bandit_types and len(self.bandit_types) != len(self.arm_counts):
            raise ValueError("bandit_types must match arm_counts in length")


class RecommendationAlgorithm(ABC):
    """Interface every recommendation algorithm must implement."""

    #: Registry key; override in subclasses (used for CLI selection & filenames).
    name: str = "base"

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        self.config = config
        self.initial_bias = list(initial_bias)
        self.total_rounds = total_rounds

    @abstractmethod
    def choose(self, round_num: int) -> List[int]:
        """Return the arm chosen for each bandit this round (1-indexed round_num)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, arms_chosen: List[int], reward: float) -> None:
        """Incorporate the observed reward for the chosen arm combination."""
        raise NotImplementedError

    @abstractmethod
    def predict_best(self) -> List[int]:
        """Return the algorithm's current best-guess arm for each bandit."""
        raise NotImplementedError
