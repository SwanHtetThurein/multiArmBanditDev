"""Constrained COMBO: COMBO under DreamTeam's switching rules.

Keeps the COMBO machinery intact - the Gaussian-process surrogate with the
ARD diffusion kernel on the graph Cartesian product, marginal-likelihood
hyperparameter fitting, and Expected Improvement acquisition - but
restricts which teams may be played each round using the shared translation
of DreamTeam's per-bandit switching schedules and global change budget
(see constraints.py).

Differences from unconstrained COMBO:
  - No free random-initialization phase: rounds 2..N_INIT explore randomly
    but only within the unlocked dimensions and the round's switch budget.
  - Acquisition: instead of a free multi-start local search over Expected
    Improvement, start from the CURRENT team and greedily apply up to k(t)
    best single-arm EI improvements, touching only unlocked bandits. EI is
    deterministic given the fitted GP, so the greedy loop is internally
    consistent (like constrained NeuralUCB, and unlike NeuralTS, which
    needs a per-round frozen posterior sample).

predict_best() is unchanged (unconstrained posterior-mean maximization):
it is a report of belief, not an action, matching how DreamTeam's own
argmax-of-means prediction is scored.
"""

from typing import List

import numpy as np

from .combo import ComboAlgorithm
from .constraints import SwitchingConstraintMixin


class ConstrainedComboAlgorithm(SwitchingConstraintMixin, ComboAlgorithm):
    name = "combo_constrained"

    def __init__(self, config, initial_bias, total_rounds):
        super().__init__(config, initial_bias, total_rounds)
        self._init_constraints(config, initial_bias)
        # the constraint mixin indexes arm counts as self.n
        self.n = list(config.arm_counts)

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.current_team = list(self.initial_bias)
        else:
            unlocked = self._unlocked_dims(round_num)
            k = min(self._switch_budget(round_num), len(unlocked))
            if k > 0:
                if len(self.y) <= self.N_INIT:
                    self.current_team = self._constrained_random(unlocked, k)
                else:
                    # same fit/refit cadence as unconstrained COMBO
                    self._rounds_since_fit += 1
                    refit = (self._chol is None) or (self._rounds_since_fit >= self.REFIT_EVERY)
                    self._fit(optimize_hypers=refit)
                    if refit:
                        self._rounds_since_fit = 0
                    best_y = float(np.max(self.y))

                    def ei_score(teams):
                        mu, sd = self._posterior(np.array(teams))
                        return self._ei(mu, sd, best_y)

                    self.current_team = self._constrained_greedy(ei_score, unlocked, k)
        self.last_choice = list(self.current_team)
        return list(self.current_team)
