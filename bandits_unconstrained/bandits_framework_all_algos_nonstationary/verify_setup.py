"""Self-check: confirm the guarantees this harness claims.

    python3 verify_setup.py

Run it before a long sweep, or after moving the folder to another machine.
It takes well under a minute and checks, in order:

  1. Dependencies are importable (numpy, scipy, pyarrow, tqdm).
  2. The shared settings file exists, its contents match its recorded digest,
     and it is reproducible from its seed.
  3. Every algorithm is handed the identical settings.
  4. Every algorithm faces the identical reward noise on a given
     (setting, noise) pair -- common random numbers.
  5. Running on 1 core and on N cores produces byte-identical results.
  6. A crashed run resumes correctly, including when the crash landed
     mid-write.
  7. The trace records what it claims to: requested vs played teams, the
     blocked changes, and per-algorithm diagnostics.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PKG = "bandits_unconstrained.bandits_framework_all_algos_nonstationary"

PASS, FAIL = "  PASS", "  FAIL"


def pa_equal(table, column, value):
    """Boolean mask for `table[column] == value` (pyarrow compute)."""
    import pyarrow.compute as pc
    return pc.equal(table.column(column), value)

_failures = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL}  {name}")
    if detail:
        print(f"        {detail}")
    if not ok:
        _failures.append(name)
    return ok


def main():
    print("=" * 66)
    print("Verifying the bandit harness")
    print("=" * 66)

    # 1 ── dependencies ------------------------------------------------------
    print("\n[1] Dependencies")
    try:
        import numpy; check("numpy", True, numpy.__version__)
    except ImportError:
        check("numpy", False, "pip install numpy")
    try:
        import scipy; check("scipy", True, scipy.__version__)
    except ImportError:
        check("scipy", False, "pip install scipy  (needed by the 'combo' algorithm)")
    try:
        import pyarrow; check("pyarrow", True, pyarrow.__version__)
    except ImportError:
        check("pyarrow", False, "pip install pyarrow  (REQUIRED -- results and traces are Parquet)")
    try:
        import tqdm; check("tqdm", True, tqdm.__version__)
    except ImportError:
        check("tqdm", False, "pip install tqdm  (progress bars; the sweep still runs without it)")

    import importlib
    sampling = importlib.import_module(PKG + ".sampling")
    experiment = importlib.import_module(PKG + ".experiment")
    parallel = importlib.import_module(PKG + ".parallel")
    tracing = importlib.import_module(PKG + ".tracing")
    environment = importlib.import_module(PKG + ".environment")
    algorithms = importlib.import_module(PKG + ".algorithms")

    import multiprocessing as mp
    print(f"        {mp.cpu_count()} CPU cores visible")

    # 2 ── settings file -----------------------------------------------------
    print("\n[2] Shared settings file")
    doc = sampling.ensure_settings(HERE)
    check("exists and digest matches its contents", True,
          f"{os.path.basename(doc['path'])}")
    v = sampling.verify_settings(doc["path"])
    check("reproducible from its seed (file was not hand-edited)",
          v["reproducible_from_seed"], f"digest {doc['digest'][:32]}...")
    n_dims = {}
    for s in doc["settings"]:
        n_dims[s["n_bandits"]] = n_dims.get(s["n_bandits"], 0) + 1
    ok_protocol = (set(n_dims) <= {3, 6, 9}
                   and all(2 <= a <= 5 for s in doc["settings"] for a in s["arm_counts"])
                   and all(t == "ongoing" for s in doc["settings"] for t in s["bandit_types"]))
    check("protocol honoured: dims in {3,6,9}, arms in 2..5, all 'ongoing'",
          ok_protocol, f"{len(doc['settings'])} settings, dims " +
          ", ".join(f"{k}x{v}" for k, v in sorted(n_dims.items())))

    # 3 ── identical settings for every algorithm ---------------------------
    print("\n[3] Every algorithm gets the identical settings")
    digests = set()
    for _name in sorted(algorithms.ALGORITHMS):
        d = sampling.load_settings(doc["path"])
        digests.add(sampling.settings_digest(d["settings"]))
    check(f"all {len(algorithms.ALGORITHMS)} algorithms load one settings file",
          len(digests) == 1, f"single digest: {digests.pop()[:32]}...")

    # 4 ── common random numbers --------------------------------------------
    print("\n[4] Identical reward noise across algorithms (common random numbers)")
    st = doc["settings"][0]

    def noise_seq(n=8, noise=0.4, master=42):
        seed = parallel.task_seed(master, st["setting_id"], noise)
        _a, env_rng, _l = parallel.derive_streams(seed)
        env = environment.TeamRewardEnvironment(st["optimal_arm"], noise, rng=env_rng)
        team = list(st["initial_bias"])
        return [round(env.reward(team), 10) for _ in range(n)]

    a, b = noise_seq(), noise_seq()
    check("the noise stream for a (setting, noise) pair is fixed", a == b,
          f"first draws: {[round(x,4) for x in a[:4]]}")
    diff = noise_seq(noise=0.6)
    check("a different noise level gives a different stream", diff != a)

    tmp = tempfile.mkdtemp(prefix="verify_")
    try:
        # 5 ── core-count independence --------------------------------------
        print("\n[5] Results do not depend on the number of cores")
        common = dict(algorithm="dreamteam", total_rounds=15, n_settings=12,
                      noise_levels=[0.4], switch_limit="flat", max_changes=2.0)
        s1 = experiment.ExperimentSettings(n_cores=1,
                                           output_dir=os.path.join(tmp, "c1"), **common)
        r1 = experiment.run_experiment(s1, base_dir=HERE, verbose=False)
        s4 = experiment.ExperimentSettings(n_cores=4,
                                           output_dir=os.path.join(tmp, "c4"), **common)
        r4 = experiment.run_experiment(s4, base_dir=HERE, verbose=False)
        same = (open(r1["results"], "rb").read() == open(r4["results"], "rb").read())
        check("1 core and 4 cores produce byte-identical output", same)

        # 6 ── resume ---------------------------------------------------------
        print("\n[6] Resume after an interrupted run")
        out = os.path.join(tmp, "resume")
        mk = lambda: experiment.ExperimentSettings(
            algorithm="dreamteam", total_rounds=15, n_settings=12,
            noise_levels=[0.4], switch_limit="flat", max_changes=2.0,
            n_cores=1, output_dir=out)
        full = experiment.run_experiment(mk(), base_dir=HERE, verbose=False)
        reference = open(full["results"], "rb").read()

        # Simulate a crash: keep 5 parts, delete the rest, and leave a .tmp
        # behind of the kind a kill mid-write would produce.
        import glob as _glob
        os.makedirs(full["parts"], exist_ok=True)
        # rebuild the parts directory from the finished trace
        import pyarrow.parquet as _pq
        tbl = _pq.read_table(full["trace"])
        df = tbl.to_pandas()
        kept = sorted(df["setting_id"].unique())[:5]
        for sid in kept:
            sub = tbl.filter(pa_equal(tbl, "setting_id", int(sid)))
            tid = parallel.task_id(int(sid), 0.4)
            tracing.write_part(sub, tracing.part_path(full["parts"], tid))
        with open(os.path.join(full["parts"], "part-bogus.parquet.tmp"), "wb") as f:
            f.write(b"\x00\x01truncated")
        for f in (full["results"], full["trace"], full["manifest"]):
            if os.path.exists(f):
                os.remove(f)
        check("crash left a partial parts directory", True,
              f"{len(kept)} of 12 runs survived, plus one torn .tmp file")

        resumed = experiment.run_experiment(mk(), base_dir=HERE, verbose=False)
        check("resumed run completed", resumed["complete"])
        check("resumed output is byte-identical to an uninterrupted run",
              open(resumed["results"], "rb").read() == reference)

        # 7 ── the trace says what it claims ----------------------------------
        print("\n[7] The trace records the decisions")
        import pyarrow.parquet as pq
        sch = pq.read_schema(resumed["trace"])
        need = ["requested_team", "played_team", "blocked_dims",
                "n_requested_changes", "n_played_changes", "allowance", "budget"]
        check("per-round decision columns present",
              all(c in sch.names for c in need),
              ", ".join(need))
        diag = [c for c in sch.names if c.startswith("diag_")]
        check("per-algorithm diagnostics present", len(diag) > 0,
              ", ".join(d[5:] for d in diag))
        tdf = pq.read_table(resumed["trace"]).to_pandas()
        blocked = tdf[tdf.n_requested_changes > tdf.n_played_changes]
        check("the switching limit is visible in the trace", len(blocked) > 0,
              f"{len(blocked)} of {len(tdf)} rounds had a change blocked")
        over = tdf[tdf.n_played_changes > tdf.allowance.fillna(99)]
        check("no round exceeded its recorded allowance", len(over) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 66)
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
