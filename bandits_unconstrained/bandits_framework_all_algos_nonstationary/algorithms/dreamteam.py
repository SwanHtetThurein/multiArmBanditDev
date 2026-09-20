"""DreamTeam -- the published CHI 2018 algorithm.

Zhou, Valentine & Bernstein, "In Search of the Dream Team: Temporally
Constrained Multi-Armed Bandits for Identifying Effective Team Structures"
(CHI 2018), DOI 10.1145/3173574.3173682.

This is the algorithm as the paper defines it. An earlier version of this
package also carried a modified descendant of it -- one that replaced the Beta
update with a threshold-and-penalty rule (alpha += r above a 0.1 threshold,
beta += 1000 below it) -- under this same name. That variant has been removed;
what remains is the published mathematics only.

WHAT THE PUBLISHED ALGORITHM DOES
=================================
    1. Beta(1, 1) prior on every arm of every dimension.
    2. Thompson-sample each arm, normalize the draws into a distribution.
    3. Apply the per-dimension temporal schedule (see below).
    4. Sample -- not argmax -- one arm per dimension.
    5. Update: alpha += r, beta += (1 - r).

Step 5 is the standard Beta update for a reward in [0, 1]: every observation
moves the posterior by exactly one unit of evidence, split between success and
failure in proportion to the reward.

TWO MECHANISMS THAT ARE NOT ACTIVE IN THIS BENCHMARK
====================================================
1. THE TEMPORAL SCHEDULE. The paper gives each dimension a type -- early, late
   or ongoing -- and discounts its probability of changing on a sigmoid
   schedule:

       early    d = 1 / (1 + e^(t - T/2))     free early, frozen late
       late     d = 1 / (1 + e^(T/2 - t))     frozen early, free late
       ongoing  d = 1                          never restricted

   This benchmark's sampling protocol makes every dimension `ongoing`, so
   d = 1 always and the renormalization is the identity. The harness plug-in
   below therefore omits the schedule entirely: it is not a simplification,
   it is dead code under this protocol. The standalone `DreamTeam` class
   further down keeps it, because that class reproduces the paper's own
   five-dimension system where the types do vary.

2. THE GLOBAL CONSTRAINT. The paper caps the expected number of simultaneous
   changes with a downward parabola peaking at T/2. That budget has been lifted
   out of the algorithm into `environment.SwitchLimiter`, where it is the
   `parabolic` mode and is applied to every algorithm in the suite equally
   rather than to this one alone. Run the suite with
   `--switch-limit parabolic --max-changes 2` to put everyone under the budget
   DreamTeam used to impose on itself.

   Keeping it inside the algorithm would have made DreamTeam the only
   switch-constrained arm in a field of unconstrained competitors, which is a
   fairness problem rather than a feature.

Everything else matches the paper: Beta(1,1) priors, Thompson sampling followed
by normalization, and probabilistic sampling from the final distribution.
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


# NOTE: `allowed_changes()` -- the downward parabola y(t) that used to cap the
# number of simultaneous changes -- has been removed from this module. The same
# curve is now `environment.SwitchLimiter` in `parabolic` mode, applied by the
# environment to every algorithm in the suite. Keeping a second copy here would
# only invite the two from drifting apart.


def _sanitize(probs: np.ndarray) -> np.ndarray:
    """Step 6.2: force a valid distribution; fall back to uniform if degenerate."""
    q = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    total = q.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(q), 1.0 / len(q))
    return q / total


# ── One bandit ────────────────────────────────────────────────────────────────

class _BetaBandit:
    """One dimension: Beta(alpha, beta) per arm plus the current active arm.

    `temporal_type` is retained because the standalone `DreamTeam` class below
    reproduces the paper's five named dimensions, where the types genuinely
    differ. The harness plug-in passes ONGOING for every dimension, which makes
    the schedule the identity.
    """

    def __init__(self, n_arms: int, temporal_type: str = ONGOING):
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

    def __init__(self, arm_counts: Sequence[int],
                 temporal_types: Optional[Sequence[str]] = None,
                 time_horizon: int = 100, seed: Optional[int] = None):
        self.T = int(time_horizon)
        if temporal_types is None:
            temporal_types = [ONGOING] * len(arm_counts)
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
        """Steps 2-4: Thompson sample, then apply the per-dimension temporal
        schedule. Returns one final distribution per dimension.

        Step 5 of the published algorithm -- the global budget on simultaneous
        changes -- is deliberately absent. It now lives in
        `environment.SwitchLimiter`, which applies it to every algorithm in
        the suite instead of to this one alone.
        """
        probs = self.thompson_probabilities()

        # Step 3 + 4: per-dimension temporal constraint
        constrained = []
        for b, q in zip(self.bandits, probs):
            d_dim = dimensional_discount(b.temporal_type, t, self.T)
            constrained.append(posterior_renormalization(q, b.current_arm, d_dim))

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

    See `DreamTeamAlgorithm` below for the version that plugs into the
    experiment framework.
    """

    def __init__(self, time_horizon: int, seed: Optional[int] = None):
        self.names = [d[0] for d in DIMENSIONS]
        self.values = [d[1] for d in DIMENSIONS]
        self.temporal_types = [d[2] for d in DIMENSIONS]
        self.core = _DreamTeamCore(
            arm_counts=[len(v) for v in self.values],
            temporal_types=self.temporal_types,
            time_horizon=time_horizon,
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
        structure using the Thompson-sample / dimensional-schedule procedure.

        Note that no cap on the number of simultaneous changes is applied
        here; `n_changes` in the returned dict is therefore whatever the
        posteriors produced. Wrap this class in
        `environment.SwitchLimiter.project()` if you want the paper's budget.
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

class DreamTeamAlgorithm(RecommendationAlgorithm):
    """The published DreamTeam algorithm, as an experiment-harness plug-in.

    Identical mathematics to `DreamTeam` above -- same Beta update, same
    renormalization -- generalized from the paper's fixed five dimensions to
    whatever ProblemConfig the harness supplies.

    Two of the paper's mechanisms are inactive here, both explained at length
    in the module docstring:

      - the temporal schedule, because every dimension in this benchmark is
        `ongoing`, which makes the discount 1 and the renormalization the
        identity;
      - the global switching budget, which now lives in
        `environment.SwitchLimiter` and is applied to every algorithm equally
        rather than to this one alone.

    Note on predict_best(): the published algorithm has no notion of a "best
    guess" separate from the structure it is currently running, since its
    output is the structure a real team then works under. To keep the
    comparison with the other arms about the algorithm rather than about the
    readout, this reports the per-dimension argmax of the posterior means, as
    every other arm in the suite does. Set PREDICT_FROM_CURRENT = True to score
    the currently deployed structure instead, which is the more literal reading
    of the paper.
    """

    name = "dreamteam"

    #: Score the deployed structure rather than the posterior-mean argmax.
    PREDICT_FROM_CURRENT = False

    def __init__(self, config: ProblemConfig, initial_bias: List[int], total_rounds: int):
        super().__init__(config, initial_bias, total_rounds)
        # Every dimension is 'ongoing' in this benchmark, so the temporal
        # schedule is the identity and is simply not applied. ProblemConfig's
        # bandit_types are ignored deliberately rather than silently: if a
        # future protocol reintroduces early/late dimensions, this is the line
        # to change, and `dimensional_discount()` above is still here for it.
        self.core = _DreamTeamCore(
            arm_counts=list(config.arm_counts),
            temporal_types=None,
            time_horizon=total_rounds,
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

    def notify_played(self, arms_played, arms_requested=None) -> None:
        """The switching limit may have reverted some of our requested changes.

        Each dimension's `current_arm` is what the renormalization schedule is
        defined against, so it has to track the team the environment actually
        fielded rather than the one we sampled.
        """
        for b, arm in zip(self.core.bandits, arms_played):
            b.current_arm = int(arm)
        self.last_choice = list(arms_played)

    def update(self, arms_chosen: List[int], reward: float) -> None:
        for b, arm in zip(self.core.bandits, arms_chosen):
            b.update(int(arm), reward)

    def diagnostics(self) -> dict:
        """Posterior shape under the published Beta update.

        `n_condemned` counts arms whose beta has grown past 1000. Under the
        published rule that essentially cannot happen -- beta rises by at most
        1 per round -- so it should stay at zero. It is recorded because the
        modified variant this package used to carry drove it up sharply, and
        keeping the column makes that difference measurable against any
        archived results from that version.
        """
        import numpy as _np
        means, condemned, total = [], 0, 0
        for b in self.core.bandits:
            pm = b.posterior_means()
            means.extend(float(x) for x in pm)
            total += len(pm)
            condemned += int(_np.sum(b.beta >= 1000.0))
        return {
            "n_arms": total,
            "n_condemned": condemned,
            "frac_condemned": condemned / total if total else 0.0,
            "mean_posterior": sum(means) / len(means) if means else 0.0,
            "max_posterior": max(means) if means else 0.0,
            "min_posterior": min(means) if means else 0.0,
        }

    def predict_best(self) -> List[int]:
        if self.PREDICT_FROM_CURRENT:
            return list(self.last_choice)
        return [int(np.argmax(b.posterior_means())) for b in self.core.bandits]
