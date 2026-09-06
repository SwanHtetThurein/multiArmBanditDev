"""DreamTeam as published -- a faithful reproduction of the CHI 2018 algorithm.

Zhou, Valentine & Bernstein, "In Search of the Dream Team: Temporally
Constrained Multi-Armed Bandits for Identifying Effective Team Structures"
(CHI 2018), DOI 10.1145/3173574.3173682.

This file exists because `dreamteam.py` in this package is NOT the published
algorithm -- it is this project's modified descendant of it. Keeping both lets
the paper measure what those modifications actually did, instead of describing
them in prose.

======================================================================
WHERE THIS DIFFERS FROM dreamteam.py (the whole point of the file)
======================================================================

1. THE POSTERIOR UPDATE -- the largest difference by far.

       published (here):  alpha += r          beta += (1 - r)
       dreamteam.py:      if r > 0.1: alpha += r
                          else:       beta += 1000

   The published rule is the standard Beta update for a reward in [0, 1]:
   every observation moves the posterior by exactly one unit of evidence,
   split between success and failure in proportion to the reward.

   The modified rule is not a Bayesian update at all. A reward at or below
   0.1 adds 1000 to beta, which drives that arm's posterior mean to
   essentially zero in a single observation and makes it unrecoverable --
   1000 subsequent perfect rewards would be needed to undo one such hit. It
   also never credits failure proportionally: a reward of 0.11 and a reward
   of 0.99 both only ever increase alpha. Under this project's noise model,
   where reward noise scales with the reward, a good team can draw a low
   reward by chance and have several of its correct arms permanently
   condemned.

2. THE GLOBAL CONSTRAINT'S REDISTRIBUTION RULE.

   Both versions cap the expected number of simultaneous changes at the same
   parabolic budget y(t), but they share the cut out differently:

       published (here):  d_global = 1 - excess / (z_d * D)   per dimension
                          -> subtracts excess/D from EVERY dimension equally
       dreamteam.py:      scale = y / z, applied to all dimensions
                          -> scales every dimension PROPORTIONALLY

   Both drive the total anticipated changes to y when nothing clamps. They
   differ in who pays: the published rule takes an equal absolute share from
   each dimension, so a dimension that only mildly wants to change can be
   silenced entirely (its d_global clamps at 0), whereas proportional scaling
   preserves the relative ordering of how much each dimension wants to move.

   One measurable consequence, verified empirically for this implementation:
   when a dimension's z_d is smaller than excess/D, its d_global goes negative
   and clamps to 0, so that dimension surrenders only z_d rather than the
   excess/D it was asked for. The shortfall is not redistributed, so the
   realised total sits ABOVE y. The published budget is therefore a soft cap,
   exceeded precisely when some dimension is already nearly frozen; whenever
   no dimension clamps it is hit exactly. `dreamteam.py`'s proportional rule
   (scale = y/z) hits y exactly in every case. Neither is wrong -- but the
   published version permits slightly more churn than its own budget states,
   which is worth knowing before attributing any behavioural difference
   between the two arms to the posterior update alone.

3. SCOPE. The published algorithm is defined over 5 named team-structure
   dimensions (405 possible structures). `dreamteam.py` generalizes to an
   arbitrary number of dimensions with arbitrary arm counts. Both are
   provided below: `DreamTeam` is the paper's fixed 5-dimension system,
   `DreamTeamOriginal` is the harness plug-in that applies the identical
   published math to whatever ProblemConfig it is given.

Everything else matches: Beta(1,1) priors, Thompson sampling followed by
normalization into a probability vector, the sigmoid EARLY/LATE/ONGOING
schedules, the posterior renormalization procedure, the downward parabola
budget peaking at T/2, and probabilistic sampling (not argmax) from the
final distribution.

======================================================================
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import RecommendationAlgorithm, ProblemConfig

EPS = 1e-12

# ── Temporal types ────────────────────────────────────────────────────────────

EARLY = "early"
LATE = "late"
ONGOING = "ongoing"


#: The paper's five team-structure dimensions.
#: name -> (arm value labels, temporal type)
DIMENSIONS = (
    ("Hierarchy",
     ("centralized", "decentralized", "none"), EARLY),
    ("Interaction Patterns",
     ("equally_distributed", "round_robin", "emergent"), LATE),
    ("Norms of Engagement",
     ("informal", "formal", "none"), ONGOING),
    ("Decision-Making Norms",
     ("convergent", "divergent", "rapid", "informed", "none"), LATE),
    ("Feedback Norms",
     ("encouraging", "critical", "none"), ONGOING),
)
#: 3 x 3 x 3 x 5 x 3 = 405 possible team structures.
N_STRUCTURES = 405


# ── Core procedures (module level so they can be tested directly) ─────────────

def _safe_sigmoid(x: float) -> float:
    """1 / (1 + e^x), guarded against overflow for large |x|."""
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def dimensional_discount(temporal_type: str, t: float, T: float) -> float:
    """Step 3: the per-dimension discount factor d_dim in (0, 1].

    EARLY   d = 1 / (1 + e^(t - T/2))   unconstrained early, restricted late
    LATE    d = 1 / (1 + e^(T/2 - t))   restricted early, unconstrained late
    ONGOING d = 1                        never restricted
    """
    half = T / 2.0
    if temporal_type == EARLY:
        return _safe_sigmoid(t - half)
    if temporal_type == LATE:
        return _safe_sigmoid(half - t)
    return 1.0


def posterior_renormalization(probs: Sequence[float], current_arm: int,
                              d: float) -> np.ndarray:
    """Step 4: the paper's core procedure.

        q'_i = q_i * d                        for i != current_arm
        q'_c = q_c + sum_{j != c} q_j (1 - d)

    d = 1 leaves the distribution untouched; d = 0 moves all mass onto the
    current arm (no change possible); intermediate d retains that fraction of
    the probability of changing. The result still sums to 1.
    """
    q = np.asarray(probs, dtype=float).copy()
    if current_arm is None:
        return q
    moved = 0.0
    for i in range(len(q)):
        if i != current_arm:
            keep = q[i] * d
            moved += q[i] - keep
            q[i] = keep
    q[current_arm] += moved
    return q


def allowed_changes(t: float, T: float, max_allowed_changes: float) -> float:
    """Step 5a: downward parabola peaking at t = T/2, zero at t = 0 and t = T."""
    half = T / 2.0
    if half <= 0:
        return 0.0
    y = max_allowed_changes * (1.0 - ((t - half) / half) ** 2)
    return max(y, 0.0)


def _sanitize(probs: np.ndarray) -> np.ndarray:
    """Step 6.2: force a valid distribution; fall back to uniform if degenerate."""
    q = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    total = q.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(q), 1.0 / len(q))
    return q / total


# ── One bandit ────────────────────────────────────────────────────────────────

class _BetaBandit:
    """One dimension: Beta(alpha, beta) per arm plus the current active arm."""

    def __init__(self, n_arms: int, temporal_type: str):
        self.n_arms = n_arms
        self.temporal_type = temporal_type
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.current_arm: Optional[int] = None

    def posterior_means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def update(self, arm: int, reward: float) -> None:
        """Step 7: the published Beta update, alpha += r and beta += (1 - r)."""
        r = min(max(float(reward), 0.0), 1.0)
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r


# ── The shared engine (used by both public classes) ──────────────────────────

class _DreamTeamCore:
    """Steps 2-6 of the published algorithm over an arbitrary set of bandits."""

    def __init__(self, arm_counts: Sequence[int], temporal_types: Sequence[str],
                 time_horizon: int, max_allowed_changes: float = 2.0,
                 seed: Optional[int] = None):
        self.T = int(time_horizon)
        self.max_allowed_changes = float(max_allowed_changes)
        self.bandits = [_BetaBandit(int(n), ttype)
                        for n, ttype in zip(arm_counts, temporal_types)]
        self.D = len(self.bandits)
        # seed=None deliberately falls through to numpy's global stream, so an
        # experiment harness that calls np.random.seed() still controls this.
        self._gen = np.random.default_rng(seed) if seed is not None else None

    # -- random helpers -------------------------------------------------------

    def _beta_draw(self, a, b):
        if self._gen is not None:
            return self._gen.beta(a, b)
        return np.random.beta(a, b)

    def _choice(self, n: int, p) -> int:
        if self._gen is not None:
            return int(self._gen.choice(n, p=p))
        return int(np.random.choice(n, p=p))

    def _randrange(self, n: int) -> int:
        if self._gen is not None:
            return int(self._gen.integers(n))
        return int(np.random.randint(n))

    # -- the algorithm --------------------------------------------------------

    def thompson_probabilities(self) -> List[np.ndarray]:
        """Step 2: sample each arm's Beta, then normalize into a distribution."""
        out = []
        for b in self.bandits:
            q = self._beta_draw(b.alpha, b.beta)
            out.append(q / (q.sum() + EPS))
        return out

    def constrained_probabilities(self, t: int) -> List[np.ndarray]:
        """Steps 2-5: Thompson sample, apply the dimensional schedule, then the
        global change budget. Returns one final distribution per dimension."""
        probs = self.thompson_probabilities()

        # Step 3 + 4: per-dimension temporal constraint
        constrained = []
        for b, q in zip(self.bandits, probs):
            d_dim = dimensional_discount(b.temporal_type, t, self.T)
            constrained.append(posterior_renormalization(q, b.current_arm, d_dim))

        # Step 5b: anticipated changes, per dimension and in total
        z_list = []
        for b, q in zip(self.bandits, constrained):
            if b.current_arm is None:
                z_list.append(float(q.sum()))
            else:
                z_list.append(float(q.sum() - q[b.current_arm]))
        z = float(sum(z_list))

        # Step 5a + 5c: global constraint
        y = allowed_changes(t, self.T, self.max_allowed_changes)
        if z > y:
            excess = z - y
            for idx, (b, q) in enumerate(zip(self.bandits, constrained)):
                z_d = z_list[idx]
                if z_d > 0.0:
                    d_global = 1.0 - (excess / (z_d * self.D))
                    d_global = min(max(d_global, 0.0), 1.0)
                    constrained[idx] = posterior_renormalization(
                        q, b.current_arm, d_global)

        return constrained

    def select_arms(self, t: int) -> List[int]:
        """Step 6: sample -- not argmax -- one arm per dimension."""
        final = self.constrained_probabilities(t)
        chosen = []
        for b, q in zip(self.bandits, final):
            p = _sanitize(q)
            arm = self._choice(b.n_arms, p)
            b.current_arm = arm
            chosen.append(arm)
        return chosen

    def observe(self, reward: float) -> None:
        """Step 7: update every dimension's played arm with the shared reward."""
        for b in self.bandits:
            if b.current_arm is not None:
                b.update(b.current_arm, reward)

    def random_arms(self) -> List[int]:
        return [self._randrange(b.n_arms) for b in self.bandits]


# ══════════════════════════════════════════════════════════════════════════════
# 1. The paper's system: five named dimensions, 405 structures
# ══════════════════════════════════════════════════════════════════════════════

class DreamTeam:
    """The published DreamTeam system over its five team-structure dimensions.

    Standalone -- it does not implement the RecommendationAlgorithm interface
    and is not registered with the experiment harness. Use it to reproduce or
    inspect the paper's behaviour directly:

        dt = DreamTeam(time_horizon=10, seed=0)
        dt.initialize()
        print(dt.get_current_structure())
        info = dt.step(reward=0.8)
        print(info["n_changes"], info["new_structure"])

    See `DreamTeamOriginal` below for the version that plugs into the
    experiment framework.
    """

    def __init__(self, time_horizon: int, max_allowed_changes: float = 2.0,
                 seed: Optional[int] = None):
        self.names = [d[0] for d in DIMENSIONS]
        self.values = [d[1] for d in DIMENSIONS]
        self.temporal_types = [d[2] for d in DIMENSIONS]
        self.core = _DreamTeamCore(
            arm_counts=[len(v) for v in self.values],
            temporal_types=self.temporal_types,
            time_horizon=time_horizon,
            max_allowed_changes=max_allowed_changes,
            seed=seed,
        )
        self.time_horizon = int(time_horizon)
        self.round = 1
        self._initialized = False

    # -- public interface -----------------------------------------------------

    def initialize(self, initial_arms: Optional[List[int]] = None) -> None:
        """Set the starting structure; random per dimension when not given."""
        if initial_arms is None:
            initial_arms = self.core.random_arms()
        if len(initial_arms) != len(self.core.bandits):
            raise ValueError(
                f"initial_arms must have {len(self.core.bandits)} entries")
        for b, arm in zip(self.core.bandits, initial_arms):
            if not 0 <= int(arm) < b.n_arms:
                raise ValueError(f"arm {arm} out of range for a {b.n_arms}-arm dimension")
            b.current_arm = int(arm)
        self.round = 1
        self._initialized = True

    def get_current_structure(self) -> Dict[str, str]:
        """{dimension name: selected value}."""
        if not self._initialized:
            raise RuntimeError("call initialize() before get_current_structure()")
        return {name: vals[b.current_arm]
                for name, vals, b in zip(self.names, self.values, self.core.bandits)}

    def get_current_arms(self) -> List[int]:
        return [b.current_arm for b in self.core.bandits]

    def step(self, reward: float) -> Dict:
        """Called after a round completes.

        Updates the posteriors for the arms just played, advances the round,
        and -- while still inside the time horizon -- selects the next
        structure using the full constrained procedure.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before step()")

        old_structure = self.get_current_structure()
        old_arms = self.get_current_arms()

        self.core.observe(reward)              # step 7
        self.round += 1                        # advance

        if self.round <= self.time_horizon:    # step 2-6
            self.core.select_arms(self.round)

        new_structure = self.get_current_structure()
        new_arms = self.get_current_arms()

        changes = [
            {"dimension": self.names[i],
             "from": self.values[i][old_arms[i]],
             "to": self.values[i][new_arms[i]]}
            for i in range(len(self.names)) if old_arms[i] != new_arms[i]
        ]

        return {
            "round": self.round,
            "old_structure": old_structure,
            "new_structure": new_structure,
            "changes": changes,
            "n_changes": len(changes),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Harness plug-in: the identical published math, arbitrary dimensions
# ══════════════════════════════════════════════════════════════════════════════

class DreamTeamOriginal(RecommendationAlgorithm):
    """The published DreamTeam algorithm, as an experiment-harness plug-in.

    Identical mathematics to `DreamTeam` above -- same Beta update, same
    sigmoid schedules, same renormalization, same equal-share global
    constraint -- generalized from the paper's fixed five dimensions to
    whatever ProblemConfig the harness supplies. Run it against `dreamteam`
    to measure what this project's modifications changed.

    Note on predict_best(): the published algorithm has no notion of a
    "best guess" separate from the structure it is currently running, since
    its output is the structure a real team then works under. To keep the
    comparison with `dreamteam` about the algorithm rather than about the
    readout, this reports the per-dimension argmax of the posterior means,
    exactly as `dreamteam.py` does. Set PREDICT_FROM_CURRENT = True to score
    the currently deployed structure instead, which is the more literal
    reading of the paper.
    """

    name = "dreamteam_orig"

    #: Peak of the parabolic budget on simultaneous changes (the paper's default).
    MAX_ALLOWED_CHANGES = 2.0
    #: Score the deployed structure rather than the posterior-mean argmax.
    PREDICT_FROM_CURRENT = False

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        types = (list(config.bandit_types) if config.bandit_types
                 else [ONGOING] * config.n_bandits)
        self.core = _DreamTeamCore(
            arm_counts=list(config.arm_counts),
            temporal_types=types,
            time_horizon=total_rounds,
            max_allowed_changes=self.MAX_ALLOWED_CHANGES,
            seed=None,                      # follow the harness's global seeding
        )
        for b, arm in zip(self.core.bandits, self.initial_bias):
            b.current_arm = int(arm)
        self.last_choice = list(initial_bias)

    # -- RecommendationAlgorithm interface ------------------------------------

    def choose(self, round_num: int) -> List[int]:
        if round_num == 1:
            # the initial structure the team starts from
            for b, arm in zip(self.core.bandits, self.initial_bias):
                b.current_arm = int(arm)
            self.last_choice = list(self.initial_bias)
        else:
            self.last_choice = self.core.select_arms(round_num)
        return list(self.last_choice)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        for b, arm in zip(self.core.bandits, arms_chosen):
            b.update(int(arm), reward)

    def predict_best(self) -> List[int]:
        if self.PREDICT_FROM_CURRENT:
            return list(self.last_choice)
        return [int(np.argmax(b.posterior_means())) for b in self.core.bandits]
