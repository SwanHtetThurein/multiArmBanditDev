"""Parallel execution with checkpointing, shared by both drivers.

Three things live here.

1. TASK SEEDING -- and why it is not the obvious thing.

   A task is one (setting, noise level) pair. Its seed is derived from
   (master_seed, setting_id, noise_level) and deliberately NOT from the
   algorithm name or from anything about execution order. Two consequences:

     - Every algorithm faces the identical noise on the identical problem,
       so comparisons are paired.
     - The result does not depend on how tasks were distributed across
       processes, so 1 core and 16 cores produce byte-identical output, and
       so does a run that was interrupted and resumed.

   Within a task, three independent streams are derived from that one seed:
   the algorithm's (the global `random` / `numpy.random`, which is what the
   plug-ins use), the environment's reward noise, and the switch limiter's.
   Keeping them separate is what stops a change in one from shifting the
   others -- e.g. turning on a switching limit must not alter the noise
   sequence, or the constrained and unconstrained conditions would not be
   comparable run-for-run.

2. CHECKPOINTING -- resume after an interrupt.

   Each finished task writes ONE Parquet part file, named from its task id,
   into a `parts/` directory: written to a temp name, then renamed. Rename is
   atomic, so a part either exists complete or does not exist at all. There is
   no such thing as a half-written part, which is what makes the parts
   themselves the resume checkpoint: on restart, the task ids already on disk
   are simply skipped.

   This replaced an append-to-CSV checkpoint, and is strictly better. Appending
   could leave a torn final row that had to be detected and discarded, and it
   meant the trace and the results were two files that could disagree about
   what had finished. Now there is one artifact per task, and the results table
   is derived from it at the end, so they cannot disagree.

   When every task is present the parts are compacted into one sorted file and
   deleted. Sorting is what keeps output independent of completion order.

3. THE POOL. Workers ignore SIGINT so that Ctrl-C is handled once, by the
   parent, which then shuts the pool down and leaves a clean partial file.
"""

from __future__ import annotations

import hashlib
import os
import signal
from typing import Callable, List, Optional, Tuple

import tracing

# tqdm is required for the progress bar; fall back to a silent stub so the
# harness still runs on a machine without it.
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:                                            # pragma: no cover
    HAVE_TQDM = False

    class tqdm:                                                # type: ignore
        def __init__(self, iterable=None, **kw):
            self.iterable = iterable
            self.n = kw.get("initial", 0)
            self.total = kw.get("total")
        def __iter__(self):
            for x in (self.iterable or []):
                yield x
        def update(self, n=1): self.n += n
        def set_postfix_str(self, *a, **k): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False


# ── task identity and seeding ────────────────────────────────────────────────

def task_id(setting_id: int, noise: float) -> str:
    """Stable, human-readable, filename-safe id for one task."""
    return f"s{setting_id:05d}_n{noise:.4f}"


def task_seed(master_seed: int, setting_id: int, noise: float) -> int:
    """A deterministic 63-bit seed for one task.

    Hash-based rather than arithmetic (e.g. master*1000+id) so that nearby
    tasks get unrelated streams instead of adjacent ones, which some RNGs
    correlate on.
    """
    key = f"{int(master_seed)}|{int(setting_id)}|{noise:.6f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


def derive_streams(seed: int):
    """Three independent streams for one task: (algorithm, environment, limiter).

    Returns (algo_seed, env_rng, limiter_rng). The caller seeds the global
    RNGs with algo_seed, because the algorithm plug-ins use `random` and
    `numpy.random` directly.
    """
    import random as _random
    import numpy as _np
    ss = _np.random.SeedSequence(seed)
    a, e, l = ss.spawn(3)
    algo_seed = int(a.generate_state(1, dtype="uint32")[0])
    env_rng = _np.random.default_rng(e)
    limiter_rng = _random.Random(int(l.generate_state(2, dtype="uint32").sum()))
    return algo_seed, env_rng, limiter_rng


def seed_globals(algo_seed: int) -> None:
    """Seed the streams the algorithm plug-ins actually draw from."""
    import random as _random
    import numpy as _np
    _random.seed(algo_seed)
    _np.random.seed(algo_seed % (2 ** 32))


# ── checkpoint ───────────────────────────────────────────────────────────────

class Checkpoint:
    """Per-task Parquet parts, doubling as the resume checkpoint.

    `parts_dir` holds one `part-<task_id>.parquet` per finished task. The set
    of files present IS the record of what is done -- there is no separate
    ledger to fall out of sync, and no partially written row to detect.
    """

    def __init__(self, parts_dir: str, rows_per_task: int = 0):
        self.parts_dir = parts_dir
        self.rows_per_task = int(rows_per_task)
        os.makedirs(parts_dir, exist_ok=True)

    # -- recovery ------------------------------------------------------------

    def completed(self) -> set:
        """task_ids with a complete part on disk."""
        return tracing.completed_parts(self.parts_dir)

    def sweep(self) -> int:
        """Delete `.tmp` leftovers from a kill during a write."""
        return tracing.sweep_tmp_files(self.parts_dir)

    # -- writing -------------------------------------------------------------

    def write_task(self, task_id: str, table) -> None:
        tracing.write_part(table, tracing.part_path(self.parts_dir, task_id))

    # -- finishing -----------------------------------------------------------

    def compact(self, trace_path: str, keep: Optional[set] = None) -> int:
        return tracing.compact(self.parts_dir, trace_path, keep=keep)

    def discard(self) -> None:
        tracing.remove_parts(self.parts_dir)

    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── the pool ─────────────────────────────────────────────────────────────────

def _init_worker():
    """Workers ignore SIGINT; the parent handles Ctrl-C once."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def resolve_n_cores(requested: Optional[int]) -> int:
    import multiprocessing as mp
    avail = mp.cpu_count()
    if requested is None or requested <= 0:
        return max(1, avail - 1)      # leave one core for the machine
    return max(1, min(int(requested), avail))


def run_tasks(worker: Callable, tasks: List, checkpoint: Checkpoint,
              n_cores: int, desc: str = "runs",
              already_done: int = 0, total: Optional[int] = None) -> Tuple[int, bool]:
    """Run `tasks` through `worker`, checkpointing each result.

    `worker(task)` must return (task_id, arrow_table). Returns (completed_now,
    interrupted). On Ctrl-C the pool is shut down; every part already on disk
    is complete, so rerunning the same command resumes.
    """
    import multiprocessing as mp

    total = total if total is not None else already_done + len(tasks)
    done_now = 0
    interrupted = False

    bar = tqdm(total=total, initial=already_done, desc=desc,
               unit="run", dynamic_ncols=True, smoothing=0.05)
    try:
        if n_cores == 1:
            # No pool at all: easier to debug, and identical results.
            for t in tasks:
                tid, table = worker(t)
                checkpoint.write_task(tid, table)
                done_now += 1
                bar.update(1)
        else:
            ctx = mp.get_context("spawn")   # safe with numpy/BLAS threads
            pool = ctx.Pool(processes=n_cores, initializer=_init_worker)
            try:
                for tid, table in pool.imap_unordered(worker, tasks, chunksize=1):
                    checkpoint.write_task(tid, table)
                    done_now += 1
                    bar.update(1)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                interrupted = True
                pool.terminate()
                pool.join()
            except Exception:
                pool.terminate()
                pool.join()
                raise
    except KeyboardInterrupt:
        interrupted = True
    finally:
        bar.close()

    return done_now, interrupted


def limit_blas_threads(n_cores: int) -> None:
    """Keep numpy/BLAS to one thread per worker.

    Without this, every worker process spawns its own BLAS thread pool and
    oversubscribes the machine -- the classic symptom is that raising
    --n-cores makes the sweep SLOWER. Must run before numpy is imported,
    which is why the drivers call it first thing.
    """
    if n_cores <= 1:
        return
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
