"""Constrained NeuralTS: NeuralTS under DreamTeam's switching rules.

Keeps the NeuralTS machinery intact - the MLP reward model and posterior
sampling from gradient-feature (neural tangent) uncertainty - but restricts
which teams may be played each round using the shared translation of
DreamTeam's per-bandit switching schedules and global change budget (see
constraints.py).

One design point deserves note. Unconstrained NeuralTS samples a reward
r~(x) ~ N(f(x), nu^2 * sigma^2(x)) independently PER CANDIDATE, which is
fine for a one-shot argmax but would make a multi-step greedy search
incoherent (each call to the score would re-roll the dice). The constrained
variant therefore draws ONE parameter-space Thompson sample per round:

    delta ~ N(0, nu^2 * Z^{-1})            (diagonal Z, as elsewhere)
    score(x) = f(x) + g(x) . delta

i.e. a single sampled perturbation of the network's linearization, giving
every candidate a consistent randomized score for the whole round. This is
the exact linear-Gaussian Thompson sample of the model NeuralTS's theory
linearizes to, so the exploration semantics are preserved: the marginal of
score(x) is N(f(x), nu^2 * sigma^2(x)), matching the paper's posterior,
with candidate scores now correlated through shared parameters rather than
independent.

Other differences from unconstrained NeuralTS mirror the other constrained
variants: no free random-initialization phase (early exploration happens
within the unlocked dimensions and switch budget), acquisition starts from
the CURRENT team and greedily applies up to k(t) single-arm improvements of
the sampled score on unlocked bandits, and predict_best() remains the
unconstrained maximization of the plain network prediction (a report of
belief, not an action).
"""

from typing import List

import numpy as np

from .neural_ucb_ts import NeuralTSAlgorithm
from .constraints import SwitchingConstraintMixin


class ConstrainedNeuralTSAlgorithm(SwitchingConstraintMixin, NeuralTSAlgorithm):
    name = "neuralts_constrained"

    def __init__(self, config, initial_bias, total_rounds):
        super().__init__(config, initial_bias, total_rounds)
        self._init_constraints(config, initial_bias)

    def _sampled_score_fn(self):
        """One parameter-space Thompson sample for this round."""
        delta = np.random.normal(0.0, self.NU / np.sqrt(self.Z_diag))

        def score(teams):
            _, _, mu = self._forward(self._encode(teams))
            G = self._grad_features(teams)
            return mu + G @ delta

        return score

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
                        self._sampled_score_fn(), unlocked, k)
        self.last_choice = list(self.current_team)
        return list(self.current_team)
