"""Non-stationary perturbation driver — 500-setting protocol, parallel, resumable.

The hidden optimal arm is swapped at two rounds (101 and 201 by default), each
time flipping `--p_perturb_dims` of that setting's dimensions. Algorithm state
is never reset, so what is measured is how fast each method notices the answer
moved and recovers.

Protocol (see sampling.py)
--------------------------
500 settings are sampled once into a shared JSON file: each is
<n_bandits from {3,6,9}, arm counts 2-5, all dimensions 'ongoing'> plus one
initial_bias and one optimal_arm. Every algorithm loads that same file, so all
of them face identical problems.

One run = one (setting, noise level) pair. With 6 noise levels that is
500 x 6 = 3000 runs per algorithm. No repetitions -- the 500 independent
settings carry the variation that repetitions used to.

The perturbation chain for each setting is derived from the setting id alone,
so every algorithm also faces the identical swaps at the identical rounds.

Dimensions now VARY per setting
-------------------------------
This matters for reading the results. One flipped dimension is a 1/3 shock on
a 3-dimension setting and a 1/9 shock on a 9-dimension one, so the size of the
dip depends on the setting as much as on the algorithm. `n_bandits` is written
into every row for exactly this reason: group by it before comparing dips or
recovery times. The recovery metrics below use each run's own `n_bandits`.

Execution
---------
`--n-cores` worker processes with a tqdm progress bar; results are
checkpointed per run. Ctrl-C at any time and rerun the same command to resume.
Task seeding is independent of core count and completion order, so 1 core,
16 cores, and an interrupted-then-resumed run all produce identical output.

Switching limit
---------------
`--switch-limit none|flat|parabolic` with `--max-changes K`, enforced by the
environment for every algorithm equally. `--max-changes` is an ABSOLUTE number
of dimensions, so at K=2 it constrains a 9-dimension setting far more than a
3-dimension one.

READ THIS BEFORE USING `parabolic` HERE. The parabolic allowance is one
settle-explore-converge arc sized to the whole run, and it knows nothing about
the perturbations. At the swaps themselves the allowance is nearly identical
(y(101) = 1.79, y(201) = 1.77), but the recovery windows sit on opposite sides
of the peak: rounds 102-201 carry a mean allowance of 1.93 (193 changes in
total) while rounds 202-300 carry only 1.02 (101 changes), decaying to zero.
Recovery from the second perturbation therefore gets about half the churn
budget of the first, which confounds recovery-time comparisons between the two
events. Use `flat` for recovery work unless that interaction is the object of
study.

Output (Parquet)
----------------
    <stem>_trace.parquet    one row per ROUND: the team the algorithm wanted,
                            the team it was allowed to play, which changes the
                            constraint blocked, the reward, its recommendation,
                            its internal diagnostics, plus `phase` and the
                            run's pre-perturbation `baseline`
    <stem>.parquet          the results projection
    <stem>_summary.parquet  per-run recovery metrics, four per perturbation
    <stem>_manifest.json    seeds, settings digest, switch limit, versions

`--trace` controls the detail level (off / basic / full; default full).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PKG = "bandits_unconstrained.bandits_framework_all_algos_nonstationary"

from bandits_unconstrained.bandits_framework_all_algos_nonstationary import parallel  # noqa: E402
from bandits_unconstrained.bandits_framework_all_algos_nonstationary.parallel import (  # noqa: E402
    Checkpoint, derive_streams, seed_globals, task_id, task_seed,
)
from bandits_unconstrained.bandits_framework_all_algos_nonstationary import sampling, tracing  # noqa: E402
from bandits_unconstrained.bandits_framework_all_algos_nonstationary.algorithms import (  # noqa: E402
    ALGORITHMS, get_algorithm, ProblemConfig,
)
from bandits_unconstrained.bandits_framework_all_algos_nonstationary.environment import (  # noqa: E402
    NO_LIMIT, DEFAULT_MAX_CHANGES, SWITCH_LIMIT_MODES, SwitchLimiter,
    TeamRewardEnvironment,
)

DEFAULT_OUTPUT_DIR = "Unconstrained Perturbation Results"
DEFAULT_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_PERTURBATION_ROUNDS = (101, 201)
DEFAULT_PERTURB_DIMS = 1



@dataclass
class PerturbationSettings:
    algorithm: str = "dreamteam"
    total_rounds: int = 300
    perturbation_rounds: Tuple[int, ...] = DEFAULT_PERTURBATION_ROUNDS
    p_perturb_dims: int = DEFAULT_PERTURB_DIMS
    noise_levels: List[float] = field(default_factory=lambda: list(DEFAULT_NOISE_LEVELS))
    seed: int = 42
    settings_seed: int = sampling.DEFAULT_SETTINGS_SEED
    n_settings: int = sampling.N_SETTINGS
    settings_path: Optional[str] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    baseline_window: Tuple[int, int] = (80, 100)
    switch_limit: str = NO_LIMIT
    max_changes: float = DEFAULT_MAX_CHANGES
    n_cores: Optional[int] = None
    resume: bool = True
    trace: str = tracing.DEFAULT_TRACE_LEVEL

    def validate(self):
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if not self.perturbation_rounds:
            raise ValueError("perturbation_rounds must be non-empty")
        for pr in self.perturbation_rounds:
            if not (1 <= pr < self.total_rounds):
                raise ValueError(
                    f"perturbation_round {pr} must be in [1, {self.total_rounds - 1}]")
        if list(self.perturbation_rounds) != sorted(self.perturbation_rounds):
            raise ValueError("perturbation_rounds must be in ascending order")
        if self.p_perturb_dims < 1:
            raise ValueError("p_perturb_dims must be >= 1")
        # Dimensions vary per setting now, so the old
        # `p_perturb_dims <= n_bandits` check has to be made against the
        # SMALLEST setting rather than one global n_bandits.
        if self.switch_limit not in SWITCH_LIMIT_MODES:
            raise ValueError(
                f"switch_limit must be one of {', '.join(SWITCH_LIMIT_MODES)}, "
                f"got '{self.switch_limit}'")
        if self.max_changes < 0:
            raise ValueError("max_changes must be >= 0")
        if self.trace not in tracing.TRACE_LEVELS:
            raise ValueError(
                f"trace must be one of {', '.join(tracing.TRACE_LEVELS)}, "
                f"got '{self.trace}'")
        lo, hi = self.baseline_window
        if not (1 <= lo <= hi <= self.total_rounds):
            raise ValueError(
                f"--baseline-window {lo} {hi} is out of range for a "
                f"{self.total_rounds}-round run")
        if hi > self.perturbation_rounds[0]:
            raise ValueError(
                f"--baseline-window {lo} {hi} must end at or before the first "
                f"perturbation (round {self.perturbation_rounds[0]}). The baseline "
                f"is each run's pre-perturbation performance, so measuring it after "
                f"a swap would be wrong. Pass a matching --baseline-window, e.g. "
                f"--baseline-window {max(1, self.perturbation_rounds[0] - 20)} "
                f"{self.perturbation_rounds[0]}")

    def make_limiter(self, rng=None) -> SwitchLimiter:
        return SwitchLimiter(mode=self.switch_limit, max_changes=self.max_changes,
                             total_rounds=self.total_rounds, rng=rng)

    def limit_tag(self) -> str:
        return self.make_limiter().label()


# ── perturbation chains ──────────────────────────────────────────────────────

def perturb_rng_for(settings_seed: int, setting_id: int) -> random.Random:
    """A per-setting RNG for the swaps, independent of algorithm and of the
    master run seed, so every algorithm and every switch-limit condition sees
    the identical perturbations."""
    key = f"perturb|{settings_seed}|{setting_id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def make_perturbed_optimal(previous: List[int], arm_counts: List[int],
                           p: int, rng: random.Random) -> List[int]:
    """A new optimum differing from `previous` in exactly p random dimensions."""
    new = list(previous)
    candidates = list(range(len(previous)))
    rng.shuffle(candidates)
    for d in candidates[:p]:
        options = [a for a in range(arm_counts[d]) if a != previous[d]]
        new[d] = rng.choice(options)
    return new


def perturbation_chain(setting: Dict, n_events: int, p: int,
                       settings_seed: int) -> List[List[int]]:
    """The chain of optima for one setting. Each flip is applied to the
    PREVIOUS optimum, so a later swap may revert an earlier dimension."""
    rng = perturb_rng_for(settings_seed, setting["setting_id"])
    p_eff = min(p, setting["n_bandits"])
    chain, current = [], list(setting["optimal_arm"])
    for _ in range(n_events):
        current = make_perturbed_optimal(current, setting["arm_counts"], p_eff, rng)
        chain.append(current)
    return chain


def phase_for_round(round_num: int, perturbation_rounds: Tuple[int, ...]) -> str:
    for i, pr in enumerate(perturbation_rounds):
        if round_num <= pr:
            return "pre" if i == 0 else f"post{i}"
    return f"post{len(perturbation_rounds)}"


# ── one run ──────────────────────────────────────────────────────────────────

def run_single_with_perturbations(algo_cls, config, initial_bias, optimal_arm,
                                  total_rounds, noise, perturbation_rounds,
                                  perturbed_optima, limiter=None, env_rng=None,
                                  trace=None, row_base=None):
    """One run, with the optimum swapped at the scheduled rounds.

    The limiter is NOT reset at a perturbation: the budget is a property of the
    world and does not know the optimum moved.
    """
    algorithm = algo_cls(config, initial_bias, total_rounds)
    environment = TeamRewardEnvironment(optimal_arm, noise,
                                        switch_limiter=limiter, rng=env_rng)
    swap_at = {pr + 1: perturbed_optima[i] for i, pr in enumerate(perturbation_rounds)}

    detailed = trace is not None and trace.detailed
    want_diag = trace is not None and trace.wants_diagnostics
    row_base = row_base or {}
    rows = []

    performances = []
    played = list(initial_bias)
    cum_changes = 0
    optimum_version = 0
    for round_num in range(1, total_rounds + 1):
        if round_num in swap_at:
            environment.optimal_arm = swap_at[round_num]
            optimum_version += 1
        requested = algorithm.choose(round_num)
        prev = played
        played = environment.apply_switch_limit(requested, prev, round_num)
        algorithm.notify_played(played, requested)
        reward = environment.reward(played)
        algorithm.update(played, reward)
        best = algorithm.predict_best()
        correct = environment.evaluate_prediction(best)
        perf = correct / config.n_bandits
        performances.append(perf)

        if trace is None:
            continue
        row = dict(row_base)
        row.update({"round": round_num, "performance": float(perf),
                    "phase": phase_for_round(round_num, tuple(perturbation_rounds)),
                    "optimum_version": optimum_version})
        if detailed:
            n_req = sum(1 for d in range(len(requested)) if requested[d] != prev[d])
            n_played = sum(1 for d in range(len(played)) if played[d] != prev[d])
            cum_changes += n_played
            # Read what project() used; recomputing would redraw the
            # randomized rounding and desynchronise the limiter's RNG.
            row.update({
                "requested_team": list(requested),
                "played_team": list(played),
                "n_requested_changes": n_req,
                "n_played_changes": n_played,
                "blocked_dims": [d for d in range(len(requested))
                                 if requested[d] != prev[d] and played[d] == prev[d]],
                "budget": (None if limiter is None or limiter.last_budget is None
                           else float(limiter.last_budget)),
                "allowance": (None if limiter is None or limiter.last_allowance is None
                              else int(limiter.last_allowance)),
                "reward": float(reward),
                "predict_best": list(best),
                "n_correct": int(correct),
                "hamming_to_optimum": sum(1 for a, b in zip(played, environment.optimal_arm)
                                          if a != b),
                "cum_changes": cum_changes,
            })
        rows.append((row, algorithm.diagnostics() if want_diag else None))

    return performances, rows


def run_task(task):
    """One (setting, noise) task. Top-level and picklable, for the pool."""
    (algo_name, setting, chain, noise, total_rounds, perturbation_rounds,
     master_seed, switch_limit, max_changes, baseline_window, run_id,
     trace_level) = task

    seed = task_seed(master_seed, setting["setting_id"], noise)
    algo_seed, env_rng, limiter_rng = derive_streams(seed)
    seed_globals(algo_seed)

    config = ProblemConfig(arm_counts=list(setting["arm_counts"]),
                           bandit_types=list(setting["bandit_types"]))
    limiter = SwitchLimiter(mode=switch_limit, max_changes=max_changes,
                            total_rounds=total_rounds, rng=limiter_rng)

    buf = tracing.TraceBuffer(level=trace_level,
                              extra_fields=tracing.PERTURBATION_FIELDS)
    row_base = {
        "run_id": run_id,
        "setting_id": setting["setting_id"],
        "n_bandits": setting["n_bandits"],
        "noise_level": float(noise),
        "noise_pct": int(round(noise * 100)),
    }

    performances, rows = run_single_with_perturbations(
        get_algorithm(algo_name), config, list(setting["initial_bias"]),
        list(setting["optimal_arm"]), total_rounds, noise,
        tuple(perturbation_rounds), chain, limiter=limiter, env_rng=env_rng,
        trace=buf, row_base=row_base,
    )

    # The baseline is a property of the whole run, so it can only be filled in
    # once every round is done.
    lo, hi = baseline_window
    baseline = sum(performances[lo - 1:hi]) / (hi - lo + 1)
    for row, diag in rows:
        row["baseline"] = baseline
        buf.add(row, diag)

    return task_id(setting["setting_id"], noise), buf.to_table()


# ── recovery summary ─────────────────────────────────────────────────────────

def compute_recovery_summary(performances, baseline, perturbation_rounds,
                             total_rounds, n_bandits, p_perturb_dims):
    """Recovery metrics for one run, one block of four values per event.

    The threshold is the achievable post-perturbation ceiling: after k events
    each flipping p dimensions, the best attainable is
    (n_bandits - k*p)/n_bandits of the original baseline. `n_bandits` is the
    RUN's own dimension count, which now varies between settings.
    """
    out = []
    p_eff = min(p_perturb_dims, n_bandits)
    for i, pr in enumerate(perturbation_rounds):
        post_start = pr + 1
        post_end = (perturbation_rounds[i + 1]
                    if i + 1 < len(perturbation_rounds) else total_rounds)
        ceiling_frac = max(0.0, (n_bandits - (i + 1) * p_eff) / n_bandits)
        threshold = baseline * ceiling_frac
        recovery_threshold = time_to_baseline = None
        max_dip = 0.0
        for t in range(post_start, post_end + 1):
            perf = performances[t - 1]
            if recovery_threshold is None and perf >= threshold:
                recovery_threshold = t - pr
            if time_to_baseline is None and perf >= baseline:
                time_to_baseline = t - pr
            if t <= min(pr + 100, post_end):
                max_dip = max(max_dip, 1.0 - perf)
        out.extend([
            recovery_threshold if recovery_threshold is not None else float("nan"),
            time_to_baseline if time_to_baseline is not None else float("nan"),
            max_dip,
            int(recovery_threshold is not None),
        ])
    return out


def write_summary(results_path: str, summary_path: str,
                  settings: PerturbationSettings, label: str) -> int:
    """Per-run recovery metrics, computed from the assembled results table.

    Each run uses its OWN `n_bandits` for the recovery ceiling, because
    dimensions vary between settings under the 500-setting protocol.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pq.read_table(results_path, columns=["run_id", "setting_id", "n_bandits",
                                               "noise_level", "round", "performance",
                                               "baseline"])
    df = tbl.to_pandas().sort_values(["run_id", "round"])

    recs = []
    for rid, g in df.groupby("run_id", sort=True):
        perfs = g["performance"].tolist()
        nb = int(g["n_bandits"].iloc[0])
        baseline = float(g["baseline"].iloc[0])
        metrics = compute_recovery_summary(
            perfs, baseline, settings.perturbation_rounds,
            settings.total_rounds, nb, settings.p_perturb_dims)
        rec = {"algorithm": label, "setting_id": int(g["setting_id"].iloc[0]),
               "n_bandits": nb, "noise_level": float(g["noise_level"].iloc[0]),
               "run_id": int(rid), "baseline": baseline}
        for i in range(len(settings.perturbation_rounds)):
            b = metrics[i * 4:(i + 1) * 4]
            rec[f"recovery_threshold_rounds_{i+1}"] = b[0]
            rec[f"time_to_baseline_rounds_{i+1}"] = b[1]
            rec[f"max_dip_{i+1}"] = b[2]
            rec[f"recovered_within_horizon_{i+1}"] = bool(b[3])
        recs.append(rec)

    # An explicit schema, not inference. Left to itself, pyarrow types each
    # metric column from the values it happens to see: a perturbation whose
    # recovery always succeeded gets int64, one with a NaN gets double. The
    # schema would then differ between perturbation events and between
    # algorithms, and concatenating summaries would fail or silently coerce.
    fields = [pa.field("algorithm", pa.string()),
              pa.field("setting_id", pa.int32()),
              pa.field("n_bandits", pa.int8()),
              pa.field("noise_level", pa.float64()),
              pa.field("run_id", pa.int32()),
              pa.field("baseline", pa.float64())]
    for i in range(1, len(settings.perturbation_rounds) + 1):
        fields += [pa.field(f"recovery_threshold_rounds_{i}", pa.float64()),
                   pa.field(f"time_to_baseline_rounds_{i}", pa.float64()),
                   pa.field(f"max_dip_{i}", pa.float64()),
                   pa.field(f"recovered_within_horizon_{i}", pa.bool_())]
    schema = pa.schema(fields)
    out = pa.Table.from_pylist(recs, schema=schema)
    tmp = summary_path + ".tmp"
    pq.write_table(out, tmp, compression=tracing.COMPRESSION,
                   compression_level=tracing.COMPRESSION_LEVEL)
    os.replace(tmp, summary_path)
    return len(recs)


# ── the sweep ────────────────────────────────────────────────────────────────

def output_paths(settings: PerturbationSettings, base_dir: Optional[str] = None):
    base_dir = base_dir or HERE
    save_dir = os.path.join(base_dir, settings.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    rounds_str = "_t" + "_t".join(str(pr) for pr in settings.perturbation_rounds)
    stem = (f"{settings.algorithm}_limit-{settings.limit_tag()}"
            f"_perturbation_rounds{settings.total_rounds}"
            f"_settings{settings.n_settings}"
            f"_perturbs{len(settings.perturbation_rounds)}{rounds_str}"
            f"_seed{settings.seed}")
    return {
        "results": os.path.join(save_dir, stem + ".parquet"),
        "trace": os.path.join(save_dir, stem + "_trace.parquet"),
        "summary": os.path.join(save_dir, stem + "_summary.parquet"),
        "manifest": os.path.join(save_dir, stem + "_manifest.json"),
        "parts": os.path.join(save_dir, stem + "_parts"),
        "stem": stem,
    }


def run_perturbation_experiment(settings: PerturbationSettings,
                                base_dir: Optional[str] = None,
                                verbose: bool = True):
    settings.validate()
    base_dir = base_dir or HERE
    started = time.time()

    doc = sampling.ensure_settings(base_dir, seed=settings.settings_seed,
                                   n_settings=settings.n_settings,
                                   path=settings.settings_path)
    problem_settings = doc["settings"]

    smallest = min(s["n_bandits"] for s in problem_settings)
    if settings.p_perturb_dims > smallest and verbose:
        print(f"  note: --p_perturb_dims {settings.p_perturb_dims} exceeds the "
              f"smallest setting's {smallest} dimensions; it is clamped per "
              f"setting to that setting's dimension count.")

    paths = output_paths(settings, base_dir)

    tasks, run_id = [], 0
    n_events = len(settings.perturbation_rounds)
    for noise in settings.noise_levels:
        for ps in problem_settings:
            chain = perturbation_chain(ps, n_events, settings.p_perturb_dims,
                                       settings.settings_seed)
            tasks.append((settings.algorithm, ps, chain, noise, settings.total_rounds,
                          settings.perturbation_rounds, settings.seed,
                          settings.switch_limit, settings.max_changes,
                          settings.baseline_window, run_id, settings.trace))
            run_id += 1
    total = len(tasks)

    # Expected task ids for THIS configuration. A checkpoint may also hold ids
    # from a different one (n_settings or the noise list changed between runs);
    # intersecting means those are ignored and dropped rather than counted
    # towards completion, which would otherwise leave the run never finishing.
    expected = {task_id(t[1]["setting_id"], t[3]) for t in tasks}

    checkpoint = Checkpoint(paths["parts"], settings.total_rounds)
    checkpoint.sweep()                       # clear .tmp files from a hard kill
    if not settings.resume:
        checkpoint.discard()
        checkpoint = Checkpoint(paths["parts"], settings.total_rounds)
        done = set()
    else:
        done = checkpoint.completed() & expected

    remaining = [t for t in tasks if task_id(t[1]["setting_id"], t[3]) not in done]
    n_cores = parallel.resolve_n_cores(settings.n_cores)

    if verbose:
        print(f"Algorithm:      {settings.algorithm}")
        print(f"Settings file:  {os.path.basename(doc['path'])}")
        print(f"                digest {doc['digest'][:16]}...  "
              f"({doc['n_settings']} settings, seed {doc['seed']})")
        print(f"Perturbations:  after rounds {list(settings.perturbation_rounds)}, "
              f"flipping {settings.p_perturb_dims} dimension(s); state never reset")
        print(settings.make_limiter().describe())
        print(f"Trace level:    {settings.trace}")
        print(f"Noise levels:   {settings.noise_levels}")
        print(f"Rounds per run: {settings.total_rounds}")
        print(f"Total runs:     {total}")
        print(f"Cores:          {n_cores}")
        if done:
            print(f"Resuming:       {len(done)} of {total} runs already done, "
                  f"{len(remaining)} to go")
        if not parallel.HAVE_TQDM:
            print("  (tqdm not installed -- no progress bar; pip install tqdm)")
        print()

    interrupted = False
    if remaining:
        with checkpoint:
            _, interrupted = parallel.run_tasks(
                run_task, remaining, checkpoint, n_cores,
                desc=f"{settings.algorithm}/{settings.limit_tag()}",
                already_done=len(done), total=total)

    finished = checkpoint.completed() & expected
    complete = finished >= expected
    result = {"algorithm": settings.algorithm, "total_tasks": total,
              "completed_tasks": len(finished), "complete": complete,
              "interrupted": interrupted, "settings_digest": doc["digest"],
              **paths}

    if complete:
        detailed = settings.trace != tracing.TRACE_OFF
        target = paths["trace"] if detailed else paths["results"]
        n_rows = checkpoint.compact(target, keep=expected)
        if detailed:
            tracing.derive_results(paths["trace"], paths["results"],
                                   columns=tracing.PERTURBATION_RESULTS_COLUMNS)
        label = (settings.algorithm if settings.limit_tag() == "none"
                 else f"{settings.algorithm} [{settings.limit_tag()}]")
        n_runs = write_summary(paths["results"], paths["summary"], settings, label)
        tracing.write_manifest(paths["manifest"], {
            "algorithm": settings.algorithm,
            "total_rounds": settings.total_rounds,
            "perturbation_rounds": list(settings.perturbation_rounds),
            "p_perturb_dims": settings.p_perturb_dims,
            "baseline_window": list(settings.baseline_window),
            "noise_levels": settings.noise_levels,
            "n_settings": settings.n_settings,
            "master_seed": settings.seed,
            "settings_seed": doc["seed"],
            "settings_digest": doc["digest"],
            "switch_limit": settings.switch_limit,
            "max_changes": settings.max_changes,
            "trace_level": settings.trace,
            "n_runs": total, "n_trace_rows": n_rows, "n_cores": n_cores,
            "wall_seconds": round(time.time() - started, 1),
        })
        result.update(rows=n_rows, n_run_summaries=n_runs)
        checkpoint.discard()
        if verbose:
            print(f"\nComplete in {time.time()-started:.0f}s.")
            if detailed:
                print(f"  trace    {n_rows:,} rows -> {os.path.basename(paths['trace'])} "
                      f"({os.path.getsize(paths['trace'])/1e6:.1f} MB)")
            print(f"  results  -> {os.path.basename(paths['results'])}")
            print(f"  summary  {n_runs} runs -> {os.path.basename(paths['summary'])}")
            print(f"  manifest -> {os.path.basename(paths['manifest'])}")
    elif verbose:
        print(f"\nStopped with {len(finished)} of {total} runs done.")
        print(f"Progress saved in {os.path.basename(paths['parts'])}/ — "
              f"rerun the same command to resume.")
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Non-stationary perturbation sweep (500-setting protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algorithm", default=None, choices=sorted(ALGORITHMS))
    p.add_argument("--all", action="store_true",
                   help="run every registered algorithm, one after another")
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--perturbation_rounds", type=int, nargs="+",
                   default=list(DEFAULT_PERTURBATION_ROUNDS),
                   help="rounds AFTER which the optimal arm is swapped")
    p.add_argument("--p_perturb_dims", type=int, default=DEFAULT_PERTURB_DIMS,
                   help="dimensions flipped at each perturbation (clamped per "
                        "setting to that setting's dimension count)")
    p.add_argument("--baseline-window", dest="baseline_window", type=int, nargs=2,
                   default=[80, 100], metavar=("LO", "HI"),
                   help="inclusive round window for each run's pre-perturbation baseline")
    p.add_argument("--noise", type=float, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="master seed for per-run RNG streams")
    p.add_argument("--settings-seed", type=int, default=sampling.DEFAULT_SETTINGS_SEED)
    p.add_argument("--n-settings", type=int, default=sampling.N_SETTINGS)
    p.add_argument("--settings-path", default=None)
    p.add_argument("--switch-limit", dest="switch_limit", default=NO_LIMIT,
                   choices=list(SWITCH_LIMIT_MODES),
                   help="cap on role changes per round; prefer 'flat' for recovery "
                        "work (see the module docstring)")
    p.add_argument("--max-changes", dest="max_changes", type=float,
                   default=DEFAULT_MAX_CHANGES)
    p.add_argument("--n-cores", dest="n_cores", type=int, default=None,
                   help="worker processes (default: all cores but one)")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--trace", default=tracing.DEFAULT_TRACE_LEVEL,
                   choices=list(tracing.TRACE_LEVELS),
                   help="per-round detail to record: 'off' keeps only the "
                        "results columns, 'basic' adds the teams and the "
                        "constraint's effect, 'full' adds algorithm internals")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.algorithm and not args.all:
        print("Pick an algorithm with --algorithm NAME, or --all for every one.\n")
        print("Available: " + ", ".join(sorted(ALGORITHMS)))
        return 2

    parallel.limit_blas_threads(parallel.resolve_n_cores(args.n_cores))
    algos = sorted(ALGORITHMS) if args.all else [args.algorithm]
    results = []
    for i, algo in enumerate(algos, start=1):
        if len(algos) > 1:
            print("=" * 62)
            print(f"[{i}/{len(algos)}] {algo}")
            print("=" * 62)
        s = PerturbationSettings(
            algorithm=algo, total_rounds=args.rounds,
            perturbation_rounds=tuple(args.perturbation_rounds),
            p_perturb_dims=args.p_perturb_dims,
            baseline_window=tuple(args.baseline_window),
            seed=args.seed, settings_seed=args.settings_seed,
            n_settings=args.n_settings, settings_path=args.settings_path,
            switch_limit=args.switch_limit, max_changes=args.max_changes,
            n_cores=args.n_cores, resume=args.resume, output_dir=args.output_dir,
            trace=args.trace)
        if args.noise is not None:
            s.noise_levels = args.noise
        r = run_perturbation_experiment(s)
        results.append(r)
        if r["interrupted"]:
            print("\nInterrupted. Rerun the same command to resume.")
            break
        print()

    if len(results) > 1:
        done = sum(1 for r in results if r["complete"])
        print("=" * 62)
        print(f"{done} of {len(results)} algorithms complete.")
        if done < len(results):
            print("Rerun the same command to finish the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
