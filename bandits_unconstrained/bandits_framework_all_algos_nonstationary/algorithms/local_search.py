"""Oblivious Local Search (OLS) baseline.

Baseline in: Baptista & Poloczek, "Bayesian Optimization of Combinatorial
Structures" (ICML 2018), arXiv:1806.08838.

OLS is the simplest possible structured search: hold an incumbent team,
evaluate every team at Hamming distance 1 from it (i.e. every single-arm
change), move to the best one if it beats the incumbent, and repeat. When no
neighbour improves on the incumbent, the incumbent is a local optimum and the
search restarts "obliviously" from a fresh random team -- it deliberately
carries no memory of the landscape between restarts, which is exactly the
property that makes it a useful lower reference for model-based methods.

Online adaptation to this framework:
  The paper's OLS assumes it can evaluate a whole neighbourhood at will; here
  every evaluation costs one round of the budget. So the neighbourhood sweep
  is spread across rounds -- one neighbour evaluated per round, in random
  order -- and the accept/restart decision is taken when the sweep completes.
  With ~9 dimensions of 2-5 arms a sweep costs roughly 20 rounds, so a
  100-round run completes a handful of hill-climbing moves.

Interface notes:
  - Round 1 plays the initial bias (comparable to DreamTeam) and its reward
    becomes the incumbent's score.
  - predict_best() returns the team with the best *average* observed reward,
    which is far more stable than the best single draw under the
    environment's multiplicative noise.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from .base import RecommendationAlgorithm, ProblemConfig


class ObliviousLocalSearch(RecommendationAlgorithm):
    name = "ols"

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        self.D = config.n_bandits
        self.n = list(config.arm_counts)

        self.incumbent: List[int] = list(initial_bias)
        self.incumbent_score: Optional[float] = None

        self._queue: List[List[int]] = []          # neighbours still to evaluate
        self._best_neighbor: Optional[List[int]] = None
        self._best_neighbor_score: float = -math.inf
        self._phase = "eval_incumbent"             # or "sweep"

        self._pending: List[int] = list(initial_bias)
        self._stats: Dict[Tuple[int, ...], List[float]] = {}

    # -- helpers --------------------------------------------------------------

    def _neighbors(self, team: List[int]) -> List[List[int]]:
        """Every team differing from `team` in exactly one dimension."""
        neigh = []
        for d in range(self.D):
            for a in range(self.n[d]):
                if a != team[d]:
                    t = list(team)
                    t[d] = a
                    neigh.append(t)
        return neigh

    def _start_sweep(self) -> None:
        self._queue = self._neighbors(self.incumbent)
        random.shuffle(self._queue)
        self._best_neighbor = None
        self._best_neighbor_score = -math.inf
        self._phase = "sweep"

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
        if self._phase == "eval_incumbent" or not self._queue:
            self._pending = list(self.incumbent)
        else:
            self._pending = list(self._queue.pop())
        return list(self._pending)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        self._record(arms_chosen, reward)

        if self._phase == "eval_incumbent":
            self.incumbent = list(arms_chosen)
            self.incumbent_score = reward
            self._start_sweep()
            return

        # sweeping the neighbourhood of the incumbent
        if reward > self._best_neighbor_score:
            self._best_neighbor = list(arms_chosen)
            self._best_neighbor_score = reward

        if not self._queue:                        # sweep finished
            improved = (self._best_neighbor is not None
                        and self.incumbent_score is not None
                        and self._best_neighbor_score > self.incumbent_score)
            if improved:
                self.incumbent = list(self._best_neighbor)
                self.incumbent_score = self._best_neighbor_score
                self._start_sweep()
            else:
                # local optimum -> oblivious random restart
                self.incumbent = [random.randrange(k) for k in self.n]
                self.incumbent_score = None
                self._phase = "eval_incumbent"

    def predict_best(self) -> List[int]:
        if not self._stats:
            return list(self.initial_bias)
        best = max(self._stats, key=lambda k: self._stats[k][0] / self._stats[k][1])
        return list(best)
