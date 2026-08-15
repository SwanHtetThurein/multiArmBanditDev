"""Constrained BOCS: BOCS forced to obey DreamTeam's switching rules.

Purpose: an apples-to-apples comparison with DreamTeam. Keeps BOCS's
surrogate and Thompson-sampling acquisition intact but restricts WHICH
teams it may play, using the shared translation of DreamTeam's per-bandit
switching schedules and global change budget (see constraints.py for the
full derivation). There is no free random-initialization phase: early
rounds are exactly as frozen as they are for DreamTeam (y(1) ~ 0), so
exploration must happen within the same budget DreamTeam gets.

predict_best() is unchanged from BOCS (posterior-mean maximization): it is
a report of belief, not an action, and DreamTeam's equivalent (argmax of
Beta means) is likewise unconstrained.
"""

from typing import List

from .bocs import BocsAlgorithm
from .constraints import SwitchingConstraintMixin


class ConstrainedBocsAlgorithm(SwitchingConstraintMixin, BocsAlgorithm):
    name = "bocs_constrained"

    def __init__(self, config, initial_bias, total_rounds):
        super().__init__(config, initial_bias, total_rounds)
        self._init_constraints(config, initial_bias)

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            self.current_team = list(self.initial_bias)
        else:
            unlocked = self._unlocked_dims(round_num)
            k = min(self._switch_budget(round_num), len(unlocked))
            if k > 0:
                if self.t <= self.N_INIT:
                    self.current_team = self._constrained_random(unlocked, k)
                else:
                    w = self._sample_weights()
                    self.current_team = self._constrained_greedy(
                        lambda ts: self._score_teams(ts, w), unlocked, k)
        self.last_choice = list(self.current_team)
        return list(self.current_team)
