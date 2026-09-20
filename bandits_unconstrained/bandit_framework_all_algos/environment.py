"""Environment: owns the hidden optimal-arm vector, the reward function, and
the rules governing how much a team may change from one round to the next.

Kept separate from the algorithms so that (a) algorithms can never peek at
the answer, (b) alternative reward structures can be swapped in later
without touching any algorithm, and (c) the switching constraint is applied
uniformly to every algorithm rather than being re-implemented inside each of
them.

Two things live here:

  TeamRewardEnvironment  the hidden answer and the noisy reward signal.
  SwitchLimiter          the cap on simultaneous role changes per round.

On the switching limit
----------------------
Real teams cannot rearrange every role every week. The published DreamTeam
algorithm (Zhou, Valentine & Bernstein, CHI 2018) encoded that as an internal
budget on how much of its own probability mass could sit off the currently
held arm. That made DreamTeam the only switch-constrained arm in the suite
while every competitor was free to jump anywhere in the space each round --
a documented fairness problem.

The constraint now lives here instead, as a property of the *world* rather
than of any one algorithm. Every algorithm is subject to the identical rule,
enforced identically, and no algorithm needs to know it exists. Selecting a
mode turns the switching limit from an uncontrolled difference between arms
into an experimental factor that can be swept.
"""

import math
import random
from typing import List, Optional, Sequence

import numpy as np


# ── Switching-limit modes ────────────────────────────────────────────────────

#: No cap. Every algorithm may change any number of roles each round.
NO_LIMIT = "none"
#: A constant cap: at most `max_changes` roles may change in any round.
FLAT = "flat"
#: A time-varying cap: a downward parabola peaking at the midpoint of the run,
#: zero at both ends. This is the shape DreamTeam used internally, lifted out
#: and renamed so it is no longer tied to that algorithm.
PARABOLIC = "parabolic"

SWITCH_LIMIT_MODES = (NO_LIMIT, FLAT, PARABOLIC)

#: Default cap / parabola peak. 2 is the value DreamTeam's paper used.
DEFAULT_MAX_CHANGES = 2.0


class SwitchLimiter:
    """Caps how many dimensions may differ between consecutive played teams.

    Modes
    -----
    ``none``
        No cap at all. `project()` returns the requested team untouched.
        This is the unconstrained baseline and the default.

    ``flat``
        At most `max_changes` roles may change in any round, from the first
        round to the last. A constant, easily stated rule: "you may not churn
        more than K roles per round."

    ``parabolic``
        The allowance varies over the run as

            y(t) = max_changes * (1 - ((t - T/2) / (T/2))^2)

        which is 0 at t = 0, peaks at `max_changes` at the midpoint, and
        returns to 0 at t = T. The intent is that a team settles in at the
        start, experiments most freely in the middle, and converges towards
        the end. This is the schedule DreamTeam applied to itself; here it is
        applied to every algorithm equally.

    Fractional allowances and randomized rounding
    ---------------------------------------------
    `y(t)` is real-valued, but the number of roles that actually change is an
    integer. Truncating would systematically under-spend the budget, so the
    allowance is rounded *randomly*: `floor(y)` changes are always permitted,
    plus one more with probability `y - floor(y)`. That makes

        E[allowance(t)] == y(t)

    exactly, so a hard per-round cap remains directly comparable to the soft,
    expectation-based budget DreamTeam used to enforce internally.

    Which changes survive
    ---------------------
    When an algorithm requests more changes than the allowance permits, a
    uniformly random subset of the requested changes is kept and the rest are
    reverted to the currently held arm. Random selection keeps enforcement
    entirely inside this class -- no algorithm needs to expose a scoring hook,
    and no algorithm is favoured by the truncation rule.

    Note that this limits *velocity*, not *reach*. Under the default settings
    a team starting ~6 roles away from the optimum can reach it within a
    handful of rounds out of 100. What the cap really restricts is an
    algorithm's ability to probe a distant team in order to learn from it.
    """

    def __init__(self, mode: str = NO_LIMIT,
                 max_changes: float = DEFAULT_MAX_CHANGES,
                 total_rounds: int = 100,
                 rng: Optional[random.Random] = None):
        mode = (mode or NO_LIMIT).strip().lower()
        if mode not in SWITCH_LIMIT_MODES:
            raise ValueError(
                f"unknown switch-limit mode '{mode}'. "
                f"Choose from: {', '.join(SWITCH_LIMIT_MODES)}"
            )
        if float(max_changes) < 0:
            raise ValueError(f"max_changes must be >= 0, got {max_changes}")
        if int(total_rounds) <= 0:
            raise ValueError(f"total_rounds must be positive, got {total_rounds}")

        self.mode = mode
        self.max_changes = float(max_changes)
        self.total_rounds = int(total_rounds)
        # Its own stream, so randomized rounding and truncation do not consume
        # from the same sequence the algorithm draws on. Without this, turning
        # the limit on would shift every subsequent draw and the constrained
        # and unconstrained conditions could not be compared run-for-run.
        self._rng = rng if rng is not None else random
        # What the LAST project() actually used. The allowance is a random
        # draw whenever the budget is fractional, so a caller that wants to
        # record it must read it here rather than calling allowance() again --
        # a second call would return a different number AND consume another
        # draw, which would make results depend on whether tracing was on.
        self.last_allowance: Optional[int] = None
        self.last_budget: Optional[float] = None

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """False for `none`; True for any real cap."""
        return self.mode != NO_LIMIT

    def label(self) -> str:
        """Short tag for filenames and log lines: 'none', 'flat2', 'parabolic2p5'.

        Used by the harness as a `limit-<label>` filename segment. The hyphen
        matters: the result-aggregation scripts parse algorithm names out of
        filenames with `[a-z_]+`, so an untagged separator would let the
        algorithm group swallow the tag (or, with a digit in it, fail to match
        at all and silently skip the file).
        """
        if not self.enabled:
            return "none"
        peak_txt = f"{self.max_changes:g}".replace(".", "p")
        return f"{self.mode}{peak_txt}"

    def describe(self) -> str:
        """One human-readable line, printed by the harness."""
        if not self.enabled:
            return "switch limit: none (algorithms may change any number of roles per round)"
        if self.mode == FLAT:
            return (f"switch limit: flat -- at most {self.max_changes:g} "
                    f"role change(s) per round")
        return (f"switch limit: parabolic -- allowance peaks at "
                f"{self.max_changes:g} at round {self.total_rounds // 2}, "
                f"0 at both ends")

    # ── the budget ───────────────────────────────────────────────────────────

    def budget(self, round_num: int) -> float:
        """The real-valued allowance for this round.

        Returns `math.inf` when no limit is in force.
        """
        if self.mode == NO_LIMIT:
            return math.inf
        if self.mode == FLAT:
            return self.max_changes

        half = self.total_rounds / 2.0
        if half <= 0:
            return 0.0
        y = self.max_changes * (1.0 - ((round_num - half) / half) ** 2)
        return max(y, 0.0)

    def allowance(self, round_num: int) -> int:
        """The integer number of changes permitted this round.

        Randomized rounding of `budget()`, so the expectation equals the
        real-valued budget exactly.
        """
        y = self.budget(round_num)
        if math.isinf(y):
            return self.total_rounds  # effectively unbounded
        whole = int(math.floor(y))
        frac = y - whole
        if frac > 0.0 and self._rng.random() < frac:
            whole += 1
        return whole

    # ── enforcement ──────────────────────────────────────────────────────────

    def project(self, requested: Sequence[int], current: Optional[Sequence[int]],
                round_num: int) -> List[int]:
        """Return the team that may actually be played this round.

        `requested` is what the algorithm asked for; `current` is the team
        presently in place. If the two differ in more dimensions than the
        allowance permits, a uniformly random subset of the requested changes
        is applied and the remaining dimensions stay as they are.
        """
        if self.mode == NO_LIMIT or current is None:
            self.last_allowance = None
            self.last_budget = None
            return list(requested)

        changed = [d for d in range(len(requested)) if requested[d] != current[d]]
        allowed = self.allowance(round_num)
        self.last_allowance = allowed
        self.last_budget = self.budget(round_num)

        if len(changed) <= allowed:
            return list(requested)

        played = list(current)
        for d in self._rng.sample(changed, allowed):
            played[d] = requested[d]
        return played


def make_switch_limiter(mode: str = NO_LIMIT,
                        max_changes: float = DEFAULT_MAX_CHANGES,
                        total_rounds: int = 100,
                        rng: Optional[random.Random] = None) -> SwitchLimiter:
    """Convenience constructor, so callers need not import the class."""
    return SwitchLimiter(mode=mode, max_changes=max_changes,
                         total_rounds=total_rounds, rng=rng)


# ── The reward environment ───────────────────────────────────────────────────

class TeamRewardEnvironment:
    """Reward = fraction of dimensions chosen correctly, with multiplicative
    Gaussian noise (std = noise * reward), clipped to [0, 1].

    Identical to reward_generator in the original script.

    Optionally owns a SwitchLimiter. When one is supplied, the harness routes
    each round's requested team through `apply_switch_limit()` before scoring
    it, so the constraint is enforced by the world rather than by any
    algorithm.

    COMMON RANDOM NUMBERS. Pass `rng` (a numpy Generator) to give the reward
    noise its own stream, seeded per (setting, noise level) and independent of
    the algorithm. This matters more than it looks:

      Without it, every algorithm draws noise from the one global stream, and
      since a GP fit consumes far more random numbers than, say, `random`
      does, the streams diverge immediately. Two algorithms on the same
      problem then face *different* noise realizations, so a comparison
      between them carries the variance of the noise as well as the
      difference between the methods.

      With it, every algorithm sees the identical noise sequence on the
      identical problem. Differences are attributable to the algorithms, the
      comparison becomes paired, and error bars shrink for free. It also makes
      results independent of how work was scheduled across processes.

    Omitting `rng` falls back to the global numpy stream, i.e. the original
    behaviour.
    """

    def __init__(self, optimal_arm: List[int], noise: float,
                 switch_limiter: Optional[SwitchLimiter] = None,
                 rng=None):
        self.optimal_arm = list(optimal_arm)
        self.noise = noise
        self.switch_limiter = switch_limiter or SwitchLimiter(NO_LIMIT)
        self._rng = rng if rng is not None else np.random

    def reward(self, arms_chosen: List[int]) -> float:
        mismatches = sum(1 for chosen, opt in zip(arms_chosen, self.optimal_arm)
                         if chosen != opt)
        p = 1 - (mismatches / len(arms_chosen))
        std = self.noise * p
        noisy_reward = self._rng.normal(p, std)
        return float(np.clip(noisy_reward, 0, 1))

    def evaluate_prediction(self, predicted_best: List[int]) -> int:
        """Number of dimensions where the algorithm's best guess is correct.
        Used by the harness for the per-round performance metric."""
        return sum(1 for pred, opt in zip(predicted_best, self.optimal_arm)
                   if pred == opt)

    # ── switching constraint ─────────────────────────────────────────────────

    def apply_switch_limit(self, requested: Sequence[int],
                           current: Optional[Sequence[int]],
                           round_num: int) -> List[int]:
        """Filter a requested team through this environment's switching rule.

        Returns the team that is actually fielded. With no limiter in force
        this is `requested` unchanged.
        """
        return self.switch_limiter.project(requested, current, round_num)
