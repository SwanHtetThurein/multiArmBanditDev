"""Regularized Evolution (aging evolution) baseline.

Baseline in: Oh, Gavves & Welling, "Combinatorial Bayesian Optimization using
the Graph Cartesian Product" (NeurIPS 2019), arXiv:1902.00448, where it is the
strong non-Bayesian competitor on the neural-architecture-search benchmarks.

Original method: Real, Aggarwal, Huang & Le, "Regularized Evolution for Image
Classifier Architecture Search" (AAAI 2019), arXiv:1802.01548.

The algorithm is deliberately tiny:
  1. Fill a fixed-size population with random teams.
  2. Each round, sample a tournament of TOURNAMENT members uniformly from the
     population, take the best-scoring one as the parent, and mutate it by
     changing one randomly chosen dimension to a different arm.
  3. Evaluate the child, append it to the population, and evict the *oldest*
     member -- not the worst. That aging step is the "regularization": good
     solutions only survive by being re-discovered, so the search cannot lock
     onto a lucky noisy evaluation forever.

The aging rule is what makes this a genuinely useful baseline for a noisy
objective like ours: plain (non-aging) evolution is badly biased toward teams
that happened to draw a high-noise reward once.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam); rounds
    2..POPULATION fill the population with random teams.
  - predict_best() returns the team with the best *average* observed reward,
    which is more stable than the best single draw under multiplicative noise.
"""

import random
from collections import deque
from typing import Deque, Dict, List, Tuple

from .base import RecommendationAlgorithm, ProblemConfig


class RegularizedEvolution(RecommendationAlgorithm):
    name = "regevo"

    #: Number of individuals kept alive (oldest is evicted once full).
    POPULATION = 16
    #: Tournament size for parent selection.
    TOURNAMENT = 5

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)
        self._mutable = [d for d in range(self.D) if self.n[d] > 1]

        self._population: Deque[Tuple[List[int], float]] = deque()
        self._pending: List[int] = list(initial_bias)
        self._stats: Dict[Tuple[int, ...], List[float]] = {}

    # -- helpers --------------------------------------------------------------

    def _random_team(self) -> List[int]:
        return [random.randrange(k) for k in self.n]

    def _mutate(self, team: List[int]) -> List[int]:
        if not self._mutable:
            return list(team)
        child = list(team)
        d = random.choice(self._mutable)
        child[d] = random.choice([a for a in range(self.n[d]) if a != child[d]])
        return child

    def _record(self, team: List[int], reward: float) -> None:
        key = tuple(team)
        stat = self._stats.get(key)
        if stat is None:
            self._stats[key] = [reward, 1.0]
        else:
            stat[0] += reward
            stat[1] += 1.0

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self._pending = list(self.initial_bias)
        elif round_num <= self.POPULATION or not self._population:
            self._pending = self._random_team()
        else:
            k = min(self.TOURNAMENT, len(self._population))
            tournament = random.sample(list(self._population), k)
            parent = max(tournament, key=lambda item: item[1])[0]
            self._pending = self._mutate(parent)
        return list(self._pending)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self._record(arms_chosen, reward)
        self._population.append((list(arms_chosen), float(reward)))
        if len(self._population) > self.POPULATION:
            self._population.popleft()      # aging: oldest dies, not worst

    def predict_best(self) -> List[int]:
        if not self._stats:
            return list(self.initial_bias)
        best = max(self._stats, key=lambda k: self._stats[k][0] / self._stats[k][1])
        return list(best)
