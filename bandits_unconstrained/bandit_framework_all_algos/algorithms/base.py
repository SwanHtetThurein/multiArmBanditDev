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

A note on the switching limit
-----------------------------
The environment may cap how many dimensions are allowed to change between
consecutive rounds (see environment.SwitchLimiter). When a cap is in force,
the team an algorithm requests from choose() is not necessarily the team that
gets played: some of the requested changes may be reverted.

This is handled for every algorithm without any of them needing to know:

  - update() is always called with the team that was ACTUALLY played, and
    every algorithm in this package reads its arms_chosen argument rather
    than whatever it stashed during choose(), so all surrogates, posteriors
    and design matrices train on reality automatically.

  - notify_played() is called immediately before update() with both the
    played and the requested team. The default implementation keeps the
    conventional `last_choice` attribute in sync. Algorithms that carry their
    own notion of "the team I am currently holding" -- or whose update rule
    is only valid when the played action came from their own distribution --
    override it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProblemConfig:
    """Static description of the multi-dimensional bandit problem.

    arm_counts:   number of arms for each bandit, e.g. [3, 2, 5, ...]
    bandit_types: per-bandit metadata string. Every setting in this
                  benchmark's sampling protocol is 'ongoing'; the field is kept
                  because the published DreamTeam algorithm defines early/late
                  types, and a future protocol could reintroduce them. No
                  registered algorithm currently reads it.
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

    # ── Optional hook ────────────────────────────────────────────────────────

    def notify_played(self, arms_played: List[int],
                      arms_requested: Optional[List[int]] = None) -> None:
        """Tell the algorithm which team was actually fielded this round.

        Called by the harness after choose() and before update(), on every
        round, whether or not the switching limit altered anything. When no
        limit is in force, arms_played == arms_requested and this is a no-op
        in effect.

        The default keeps `last_choice` in sync, which is the attribute most
        algorithms in this package use as a pre-warmup fallback inside
        predict_best(). Override when the algorithm tracks an incumbent team
        of its own, or when its update is only valid for actions drawn from
        its own proposal distribution.
        """
        if hasattr(self, "last_choice"):
            self.last_choice = list(arms_played)

    def diagnostics(self) -> dict:
        """Internal state worth recording, as a flat dict of scalars.

        Called once per round by the trace recorder when `--trace full` is in
        effect (the default), and the values land as `diag_<key>` columns in
        that algorithm's trace file. Keys may vary between algorithms -- each
        writes its own file -- but must stay CONSTANT within one algorithm, or
        the per-round rows will not line up into columns.

        Two rules, both because this runs 300,000+ times per sweep:

          1. Read cached state only. Never refit a model, never re-solve a
             posterior, never loop over the whole search space.
          2. Return scalars -- float, int or bool. Not arrays. A BOCS-family
             posterior is ~470 numbers; writing that every round is over half
             a gigabyte per algorithm for one field, and it is almost never
             what anyone wanted. Summarize instead (a norm, a mean, a count).

        The default returns the observation count when the algorithm keeps
        one, which covers the model-based arms without them overriding
        anything.
        """
        out = {}
        y = getattr(self, "y", None)
        if isinstance(y, list):
            out["n_obs"] = len(y)
            if y:
                out["best_y"] = float(max(y))
                out["mean_y"] = float(sum(y) / len(y))
        return out
