"""Simulated Annealing baseline.

Used as an experimental baseline in both combinatorial-BO papers this project
follows:
  - Baptista & Poloczek, "Bayesian Optimization of Combinatorial Structures"
    (ICML 2018), arXiv:1806.08838. SA appears both as a standalone baseline
    and as the inner acquisition optimizer of BOCS-SA.
  - Oh, Gavves & Welling, "Combinatorial Bayesian Optimization using the
    Graph Cartesian Product" (NeurIPS 2019), arXiv:1902.00448.

SA keeps no surrogate model at all. It holds a single incumbent team, proposes
a random single-arm change each round, and accepts the proposal with the
Metropolis criterion

    P(accept) = 1                     if new reward >= incumbent reward
              = exp(delta / T_t)      otherwise (delta = new - incumbent < 0)

under a geometric cooling schedule from T_START down to T_END across the run:
early rounds accept almost anything (exploration), late rounds accept only
improvements (exploitation).

This is the "guided local search, no model" reference point: whatever the
model-based algorithms (BOCS / COMBO / NeuralLinear) buy has to show up as an
improvement over this, not merely over the random baseline.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam); its reward seeds
    the incumbent score.
  - The environment's rewards are noisy, so the acceptance test compares noisy
    draws. That is standard SA behaviour on a stochastic objective and is kept
    as-is rather than being silently de-noised.
  - predict_best() returns the team with the best *average* observed reward
    rather than the single best draw -- a much more stable estimate under the
    environment's multiplicative noise, and the same convention used by the
    other model-free baselines here.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from .base import RecommendationAlgorithm, ProblemConfig


class SimulatedAnnealing(RecommendationAlgorithm):
    name = "sa"

    #: Initial temperature (rewards live in [0, 1], so this is a generous start).
    T_START = 0.25
    #: Final temperature at the last round.
    T_END = 0.01

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)
        #: dimensions that actually have something to switch to
        self._mutable = [d for d in range(self.D) if self.n[d] > 1]

        self.current: List[int] = list(initial_bias)
        self.current_score: Optional[float] = None
        self.proposal: List[int] = list(initial_bias)
        self._round = 1
        self._stats: Dict[Tuple[int, ...], List[float]] = {}

    # -- helpers --------------------------------------------------------------

    def _temperature(self, round_num: int) -> float:
        if self.total_rounds <= 1:
            return self.T_END
        frac = (round_num - 1) / (self.total_rounds - 1)
        return self.T_START * (self.T_END / self.T_START) ** frac

    def _random_neighbor(self, team: List[int]) -> List[int]:
        """A team differing from `team` in exactly one dimension."""
        if not self._mutable:
            return list(team)
        t = list(team)
        d = random.choice(self._mutable)
        t[d] = random.choice([a for a in range(self.n[d]) if a != t[d]])
        return t

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
        self._round = round_num
        if round_num == 1:
            self.proposal = list(self.initial_bias)
        else:
            self.proposal = self._random_neighbor(self.current)
        return list(self.proposal)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self._record(arms_chosen, reward)

        if self.current_score is None:          # first observation seeds SA
            self.current = list(arms_chosen)
            self.current_score = reward
            return

        delta = reward - self.current_score
        if delta >= 0:
            accept = True
        else:
            temperature = max(self._temperature(self._round), 1e-12)
            accept = random.random() < math.exp(delta / temperature)

        if accept:
            self.current = list(arms_chosen)
            self.current_score = reward

    def predict_best(self) -> List[int]:
        if not self._stats:
            return list(self.initial_bias)
        best = max(self._stats, key=lambda k: self._stats[k][0] / self._stats[k][1])
        return list(best)
