"""Constrained NeuralUCB: NeuralUCB under DreamTeam's switching rules.

Keeps the NeuralUCB machinery intact - the MLP reward model and the
optimism bonus derived from network-gradient (neural tangent) features -
but restricts which teams may be played each round using the shared
translation of DreamTeam's per-bandit switching schedules and global change
budget (see constraints.py).

Differences from unconstrained NeuralUCB:
  - No free random-initialization phase: rounds 2..N_INIT explore randomly
    but only within the unlocked dimensions and the round's switch budget.
  - Acquisition: instead of a free multi-start local search over
    f(x) + nu * sigma(x), start from the CURRENT team and greedily apply up
    to k(t) best single-arm improvements of the UCB score, touching only
    unlocked bandits. The UCB score is deterministic, so the greedy loop is
    internally consistent.

predict_best() is unchanged (unconstrained maximization of the plain
network prediction): it is a report of belief, not an action, matching how
DreamTeam's own argmax-of-means prediction is scored.
"""

from typing import List

from .neural_ucb_ts import NeuralUCBAlgorithm
from .constraints import SwitchingConstraintMixin


class ConstrainedNeuralUCBAlgorithm(SwitchingConstraintMixin, NeuralUCBAlgorithm):
    name = "neuralucb_constrained"

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
                if len(self.y) <= self.N_INIT:
                    self.current_team = self._constrained_random(unlocked, k)
                else:
                    if (not self._trained) or (len(self.y) % self.TRAIN_EVERY == 0):
                        self._train_network()
                    self.current_team = self._constrained_greedy(
                        self._acquisition, unlocked, k)
        self.last_choice = list(self.current_team)
        return list(self.current_team)
