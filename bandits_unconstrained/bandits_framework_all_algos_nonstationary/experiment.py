"""Experiment harness — 500-setting protocol, parallel, resumable, traced.

Protocol (see sampling.py)
--------------------------
500 settings sampled once into a shared JSON file: each is
<n_bandits from {3,6,9}, arm counts 2-5, all dimensions 'ongoing'> plus one
initial_bias and one optimal_arm. Every algorithm loads that same file, so all
of them face identical problems.

One run = one (setting, noise level) pair. With 6 noise levels that is
500 x 6 = 3000 runs per algorithm. No repetitions.

What a finished sweep leaves behind
-----------------------------------
    <stem>_trace.parquet     one row per ROUND of every run: what the
                             algorithm wanted, what it was allowed to play,
                             what the constraint blocked, the reward, its
                             recommendation, and its internal diagnostics
    <stem>.parquet           the results table -- a six-column projection of
                             the trace, which is what you actually plot
    <stem>_manifest.json     how the run was produced: seeds, settings digest,
                             switch limit, versions, wall time

The trace is the source of truth; the results file exists because reading six
columns out of a nineteen-column file on every plot is wasteful.

Execution
---------
Tasks are spread over `--n-cores` processes with a tqdm progress bar. Each
finished run writes one Parquet part, atomically; the set of parts present is
the resume checkpoint. Task seeding is independent of core count and of
completion order, so 1 core, 16 cores, and an interrupted-then-resumed run all
produce identical output.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import parallel, sampling, tracing
from parallel import (
    Checkpoint, derive_streams, seed_globals, task_id, task_seed,
)
from algorithms import (
    get_algorithm, ProblemConfig,
)
from environment import (
    NO_LIMIT, DEFAULT_MAX_CHANGES, SWITCH_LIMIT_MODES, SwitchLimiter,
    TeamRewardEnvironment,
)

DEFAULT_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ExperimentSettings:
    algorithm: str = "dreamteam"
    total_rounds: int = 100
    noise_levels: List[float] = field(default_factory=lambda: list(DEFAULT_NOISE_LEVELS))
    seed: int = 42
    settings_seed: int = sampling.DEFAULT_SETTINGS_SEED
    n_settings: int = sampling.N_SETTINGS
    settings_path: Optional[str] = None
    output_dir: str = "Global results"
    switch_limit: str = NO_LIMIT
    #: Absolute number of dimensions, not a fraction -- so it constrains a
    #: 9-dimension setting more than a 3-dimension one. Deliberate.
    max_changes: float = DEFAULT_MAX_CHANGES
    n_cores: Optional[int] = None
    resume: bool = True
    trace: str = tracing.DEFAULT_TRACE_LEVEL

    def validate(self):
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if self.switch_limit not in SWITCH_LIMIT_MODES:
            raise ValueError(
                f"switch_limit must be one of {', '.join(SWITCH_LIMIT_MODES)}, "
                f"got '{self.switch_limit}'")
        if self.max_changes < 0:
            raise ValueError("max_changes must be >= 0")
        if self.n_settings <= 0:
            raise ValueError("n_settings must be positive")
        if self.trace not in tracing.TRACE_LEVELS:
            raise ValueError(
                f"trace must be one of {', '.join(tracing.TRACE_LEVELS)}, "
                f"got '{self.trace}'")

    def make_limiter(self, rng=None) -> SwitchLimiter:
        return SwitchLimiter(mode=self.switch_limit, max_changes=self.max_changes,
                             total_rounds=self.total_rounds, rng=rng)

    def limit_tag(self) -> str:
        return self.make_limiter().label()


# ── one run ──────────────────────────────────────────────────────────────────

def run_single(algo_cls, config: ProblemConfig, initial_bias: List[int],
               optimal_arm: List[int], total_rounds: int, noise: float,
               limiter: Optional[SwitchLimiter] = None, env_rng=None,
               trace: Optional[tracing.TraceBuffer] = None,
               row_base: Optional[dict] = None) -> List[float]:
    """One run: fresh algorithm vs fresh environment.

    The switching limit is enforced at the single point every algorithm passes
    through:

        requested -> environment.apply_switch_limit(...) -> played

    The algorithm is then told what was actually played, via notify_played()
    and as update()'s arms_chosen argument, so every model trains on the team
    that was really fielded rather than the one it asked for.

    When `trace` is supplied, one row per round is recorded, including both
    teams and exactly which changes the environment blocked.
    """
    algorithm = algo_cls(config, initial_bias, total_rounds)
    environment = TeamRewardEnvironment(optimal_arm, noise,
                                        switch_limiter=limiter, rng=env_rng)
    # A row is ALWAYS recorded -- the results table is a projection of these.
    # `detailed` only decides whether the per-round detail columns come too.
    detailed = trace is not None and trace.detailed
    want_diag = trace is not None and trace.wants_diagnostics
    row_base = row_base or {}

    performances = []
    played = list(initial_bias)
    cum_changes = 0
    for round_num in range(1, total_rounds + 1):
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
        row.update({"round": round_num, "performance": float(perf)})

        if detailed:
            n_req = sum(1 for d in range(len(requested)) if requested[d] != prev[d])
            n_played = sum(1 for d in range(len(played)) if played[d] != prev[d])
            blocked = [d for d in range(len(requested))
                       if requested[d] != prev[d] and played[d] == prev[d]]
            cum_changes += n_played
            # Read what project() actually used. Calling budget()/allowance()
            # again would redraw the randomized rounding and desynchronise the
            # limiter's RNG, making traced and untraced runs diverge.
            budget = limiter.last_budget if limiter is not None else None
            allowance = limiter.last_allowance if limiter is not None else None
            row.update({
                "requested_team": list(requested),
                "played_team": list(played),
                "n_requested_changes": n_req,
                "n_played_changes": n_played,
                "blocked_dims": blocked,
                # None reads as "no cap in force this round"
                "budget": None if budget is None else float(budget),
                "allowance": None if allowance is None else int(allowance),
                "reward": float(reward),
                "predict_best": list(best),
                "n_correct": int(correct),
                "hamming_to_optimum": sum(1 for a, b in zip(played, optimal_arm) if a != b),
                "cum_changes": cum_changes,
            })

        trace.add(row, algorithm.diagnostics() if want_diag else None)

    return performances


# ── the parallel worker ──────────────────────────────────────────────────────

def run_task(task):
    """Execute one (setting, noise) task. Top-level and picklable."""
    (algo_name, setting, noise, total_rounds, master_seed,
     switch_limit, max_changes, run_id, trace_level) = task

    seed = task_seed(master_seed, setting["setting_id"], noise)
    algo_seed, env_rng, limiter_rng = derive_streams(seed)
    seed_globals(algo_seed)

    config = ProblemConfig(arm_counts=list(setting["arm_counts"]),
                           bandit_types=list(setting["bandit_types"]))
    limiter = SwitchLimiter(mode=switch_limit, max_changes=max_changes,
                            total_rounds=total_rounds, rng=limiter_rng)

    buf = tracing.TraceBuffer(level=trace_level)
    row_base = {
        "run_id": run_id,
        "setting_id": setting["setting_id"],
        "n_bandits": setting["n_bandits"],
        "noise_level": float(noise),
        "noise_pct": int(round(noise * 100)),
    }

    run_single(get_algorithm(algo_name), config, list(setting["initial_bias"]),
               list(setting["optimal_arm"]), total_rounds, noise,
               limiter=limiter, env_rng=env_rng, trace=buf, row_base=row_base)

    return task_id(setting["setting_id"], noise), buf.to_table()


# ── the sweep ────────────────────────────────────────────────────────────────

def build_tasks(settings: ExperimentSettings, problem_settings: List[Dict]) -> List:
    tasks, run_id = [], 0
    for noise in settings.noise_levels:
        for ps in problem_settings:
            tasks.append((settings.algorithm, ps, noise, settings.total_rounds,
                          settings.seed, settings.switch_limit,
                          settings.max_changes, run_id, settings.trace))
            run_id += 1
    return tasks


def output_paths(settings: ExperimentSettings, base_dir: Optional[str] = None):
    """Deterministic in the run's configuration -- no timestamp -- because a
    resumed run has to find its own parts again."""
    base_dir = base_dir or HERE
    save_dir = os.path.join(base_dir, settings.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    stem = (f"{settings.algorithm}_limit-{settings.limit_tag()}"
            f"_rounds{settings.total_rounds}"
            f"_settings{settings.n_settings}_seed{settings.seed}")
    return {
        "results": os.path.join(save_dir, stem + ".parquet"),
        "trace": os.path.join(save_dir, stem + "_trace.parquet"),
        "manifest": os.path.join(save_dir, stem + "_manifest.json"),
        "parts": os.path.join(save_dir, stem + "_parts"),
        "stem": stem,
    }


def run_experiment(settings: ExperimentSettings, base_dir: Optional[str] = None,
                   verbose: bool = True):
    settings.validate()
    base_dir = base_dir or HERE
    started = time.time()

    doc = sampling.ensure_settings(base_dir, seed=settings.settings_seed,
                                   n_settings=settings.n_settings,
                                   path=settings.settings_path)
    # An explicit --settings-path supplies its own count, which overrides
    # --n-settings. Adopt it before any filename or manifest is built, or the
    # output would claim a setting count the file does not have.
    settings.n_settings = doc["n_settings"]
    settings.settings_seed = doc["seed"]
    paths = output_paths(settings, base_dir)
    tasks = build_tasks(settings, doc["settings"])
    total = len(tasks)

    # Expected ids for THIS configuration. A parts directory may also hold ids
    # from a different one (n_settings or the noise list changed); intersecting
    # means those are ignored rather than counted towards completion.
    expected = {task_id(t[1]["setting_id"], t[2]) for t in tasks}

    checkpoint = Checkpoint(paths["parts"], settings.total_rounds)
    checkpoint.sweep()                       # clear .tmp files from a hard kill
    if not settings.resume:
        checkpoint.discard()
        checkpoint = Checkpoint(paths["parts"], settings.total_rounds)
        done = set()
    else:
        done = checkpoint.completed() & expected

    remaining = [t for t in tasks if task_id(t[1]["setting_id"], t[2]) not in done]
    n_cores = parallel.resolve_n_cores(settings.n_cores)

    if verbose:
        print(f"Algorithm:      {settings.algorithm}")
        print(f"Settings file:  {os.path.basename(doc['path'])}")
        print(f"                digest {doc['digest'][:16]}...  "
              f"({doc['n_settings']} settings, seed {doc['seed']})")
        print(settings.make_limiter().describe())
        print(f"Trace level:    {settings.trace}")
        print(f"Noise levels:   {settings.noise_levels}")
        print(f"Rounds per run: {settings.total_rounds}")
        print(f"Total runs:     {total}")
        print(f"Cores:          {n_cores}")
        if done:
            print(f"Resuming:       {len(done)} of {total} runs done, "
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
    result = {
        "algorithm": settings.algorithm, "total_tasks": total,
        "completed_tasks": len(finished), "complete": complete,
        "interrupted": interrupted, "settings_digest": doc["digest"],
        **paths,
    }

    if complete:
        # At level 'off' the parts hold only the results columns, so the
        # compacted file IS the results table and no trace file is written.
        detailed = settings.trace != tracing.TRACE_OFF
        target = paths["trace"] if detailed else paths["results"]
        n_rows = checkpoint.compact(target, keep=expected)
        n_res = (tracing.derive_results(paths["trace"], paths["results"])
                 if detailed else n_rows)
        tracing.write_manifest(paths["manifest"], {
            "algorithm": settings.algorithm,
            "algorithm_constants": _algorithm_constants(settings.algorithm),
            "total_rounds": settings.total_rounds,
            "noise_levels": settings.noise_levels,
            "n_settings": settings.n_settings,
            "master_seed": settings.seed,
            "settings_seed": doc["seed"],
            "settings_digest": doc["digest"],
            "settings_file": os.path.basename(doc["path"]),
            "switch_limit": settings.switch_limit,
            "max_changes": settings.max_changes,
            "trace_level": settings.trace,
            "n_runs": total,
            "n_trace_rows": n_rows,
            "n_cores": n_cores,
            "wall_seconds": round(time.time() - started, 1),
        })
        result.update(rows=n_rows, results_rows=n_res)
        checkpoint.discard()
        if verbose:
            print(f"\nComplete in {time.time()-started:.0f}s.")
            if detailed:
                print(f"  trace    {n_rows:,} rows  -> {os.path.basename(paths['trace'])} "
                      f"({os.path.getsize(paths['trace'])/1e6:.1f} MB)")
            print(f"  results  {n_res:,} rows  -> {os.path.basename(paths['results'])} "
                  f"({os.path.getsize(paths['results'])/1e6:.1f} MB)")
            print(f"  manifest -> {os.path.basename(paths['manifest'])}")
    elif verbose:
        print(f"\nStopped with {len(finished)} of {total} runs done.")
        print(f"Progress saved in {os.path.basename(paths['parts'])}/ — "
              f"rerun the same command to resume.")

    return result


def _algorithm_constants(name: str) -> dict:
    """The algorithm's tunable class constants, recorded so a result can be
    attributed to the exact configuration that produced it."""
    cls = get_algorithm(name)
    out = {}
    for k in dir(cls):
        if k.isupper() and not k.startswith("_"):
            v = getattr(cls, k)
            if isinstance(v, (int, float, str, bool, tuple, list)):
                out[k] = v
    return out
