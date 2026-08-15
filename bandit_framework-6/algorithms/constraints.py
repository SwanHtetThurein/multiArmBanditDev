"""Reusable switching-constraint machinery (DreamTeam's rules).

Any algorithm can be given DreamTeam's switching constraints by mixing this
in. It provides a faithful translation of DreamTeam's two mechanisms:

1. Per-bandit switching schedules (early / late / ongoing): each round,
   bandit d is UNLOCKED (allowed to change its arm) with probability
   delta_d(t) - the same sigmoid schedule DreamTeam applies to non-current
   probability mass:
     early:   delta = 1 / (1 + e^(t - T/2))
     late:    delta = 1 / (1 + e^(T/2 - t))
     ongoing: delta = 1

2. Global switching budget: at most k(t) dimensions may change per round,
   with E[k(t)] = y(t) = m * (1 - ((t - T/2)/(T/2))^2), m = 2 - since
   DreamTeam's off-current probability mass equals its expected number of
   switches per round.

Helpers provided (hosts must set self.current_team, self.n, self.D and call
_init_constraints in __init__):
  _unlocked_dims(round_num)         -> list of movable dimensions
  _switch_budget(round_num)         -> integer k with E[k] = y(t)
  _constrained_random(unlocked, k)  -> random changes within the budget
  _constrained_greedy(score_fn, unlocked, k)
        -> from current_team, greedily apply up to k best single-arm
           improvements of score_fn (only on unlocked dims, each dim at
           most once per round). score_fn maps a list of teams to scores.
"""

import math
from typing import Callable, List

import numpy as np


class SwitchingConstraintMixin:
    GLOBAL_BUDGET_PEAK = 2   # same m as DreamTeam's global_constraint

    def _init_constraints(self, config, initial_bias):
        self.current_team = list(initial_bias)
        self.bandit_types = (list(config.bandit_types)
                             if config.bandit_types
                             else ["ongoing"] * len(config.arm_counts))

    # ── DreamTeam's schedules, translated ─────────────────────────────────────

    def _delta(self, d: int, round_num: int) -> float:
        half = self.total_rounds / 2
        btype = self.bandit_types[d]
        if btype == "early":
            return 1 / (1 + math.e ** (round_num - half))
        if btype == "late":
            return 1 / (1 + math.e ** (half - round_num))
        return 1.0

    def _unlocked_dims(self, round_num: int) -> List[int]:
        return [d for d in range(self.D)
                if np.random.random() < self._delta(d, round_num)]

    def _switch_budget(self, round_num: int) -> int:
        half = self.total_rounds / 2
        y = self.GLOBAL_BUDGET_PEAK * (1 - (((round_num - half) / half) ** 2))
        y = max(y, 0.0)
        k = int(y)
        if np.random.random() < (y - k):
            k += 1
        return k

    # ── Constrained moves ─────────────────────────────────────────────────────

    def _constrained_random(self, unlocked: List[int], k: int) -> List[int]:
        """Random exploration within the budget: change up to k unlocked dims."""
        team = list(self.current_team)
        for d in list(np.random.permutation(unlocked))[:k]:
            team[d] = int(np.random.randint(self.n[d]))
        return team

    def _constrained_greedy(self, score_fn: Callable, unlocked: List[int],
                            k: int) -> List[int]:
        """From the current team, greedily apply up to k best single-arm
        improvements of score_fn, touching only unlocked dims (each dimension
        at most once per round)."""
        team = list(self.current_team)
        movable = set(unlocked)
        score = float(score_fn([team])[0])
        for _ in range(k):
            if not movable:
                break
            candidates, moves = [], []
            for d in movable:
                for a in range(self.n[d]):
                    if a != team[d]:
                        t = list(team)
                        t[d] = a
                        candidates.append(t)
                        moves.append(d)
            if not candidates:
                break
            scores = score_fn(candidates)
            j = int(np.argmax(scores))
            if scores[j] > score + 1e-12:
                team = candidates[j]
                score = float(scores[j])
                movable.discard(moves[j])
            else:
                break
        return team
