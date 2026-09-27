# REQUIREMENTS — what to install, and what breaks without it (non-stationary)

Everything here is plain Python. There is no build step, no compiler, no
database, no configuration file to edit, and nothing to run as a service. Copy
the folder anywhere, install the libraries below, and run it.

**The folder is self-contained.** It does not need to sit in any particular
directory, and it does not need a parent folder with a particular name. Unzip
it wherever you like — `D:\work\`, `~/projects/`, a scratch directory on a
cluster — and run the commands from inside it. Nothing outside the folder is
referenced.

Contents: [quick install](#quick-install) · [the libraries](#the-libraries) ·
[Python version](#python-version) · [hardware](#hardware) ·
[verifying](#verifying-the-install) · [gotchas](#environment-gotchas) ·
[offline / HPC](#offline-and-cluster-installs)

---

## Quick install

```bash
pip install numpy scipy pyarrow tqdm pandas
```

That covers running a sweep and reading the results. **Add `matplotlib`** if
you want `graph_perturbation.py` to produce the recovery plots — this folder's
pipeline ends in plots, so most people will want it:

```bash
pip install numpy scipy pyarrow tqdm pandas matplotlib
```

Add `duckdb` if you want to query results with SQL.

Then confirm it works:

```bash
python3 verify_setup.py
```

Under a minute, and it fails loudly if anything is missing or misbehaving.

---

## The libraries

| library | required? | what needs it | what happens without it |
|---|---|---|---|
| **numpy** | **required** | every algorithm, the environment, the RNG streams | nothing runs |
| **pyarrow** | **required** | all output — results, traces, checkpoints | nothing runs; every output file is Parquet |
| **pandas** | **required in practice** | `tracing.read()`, the recovery summary, the grapher | the sweep itself runs, but you cannot read results conveniently and the non-stationary summary fails |
| **scipy** | **required for one algorithm** | `combo` only, for `scipy.optimize` | the other 25 run fine; importing the registry fails, so in practice install it |
| **matplotlib** | needed for the plots | `graph_perturbation.py`, the last step of `run_full_sweep.sh` | the sweep and the aggregator finish; only the plotting step fails |
| **tqdm** | optional | the progress bar | the sweep runs silently, with a printed note. Nothing else changes |
| **duckdb** | optional | SQL queries over the Parquet files | use pandas instead |

### Why each one

**numpy** does all the linear algebra: Gaussian process Cholesky factorisations,
the Bayesian linear posteriors, the hand-written neural networks. There is no
deep-learning framework anywhere in this project — the MLPs in `neurallinear`
and `bootnn` are plain numpy with manual backpropagation.

**pyarrow** writes and reads Parquet. It is not optional because it is also the
*checkpoint* format: each finished run is one small Parquet file, and the set of
those files is what lets an interrupted sweep resume.

**pandas** is used for the recovery-metric grouping and by `tracing.read()`
(via `Table.to_pandas()`). `scipy` is used by exactly one algorithm, but because
`algorithms/__init__.py` imports the whole registry at once, a missing scipy
stops everything — so treat it as required.

### Version minimums

| library | minimum | why |
|---|---|---|
| Python | **3.8** | dataclasses and f-strings. No 3.9+ or 3.10+ syntax is used — verified by AST scan |
| pyarrow | **14.0** | `concat_tables(promote_options=...)`; earlier versions spell it `promote=` |
| numpy | **1.17** | `np.random.SeedSequence` and `default_rng`, which the per-run RNG streams depend on |
| pandas | 1.0 | nothing exotic |
| scipy | 1.4 | `scipy.optimize.minimize` |
| matplotlib | 3.0 | nothing exotic |

Tested on Python 3.10.12 with numpy 2.2.6, pyarrow 25.0.1, scipy 1.15.3,
pandas 2.3.3, matplotlib 3.10.9. Newer is fine; the upper bounds are open.

Two things worth knowing about pyarrow specifically. It must be built with
**zstd** support, which every official wheel from PyPI and conda-forge is —
`verify_setup.py` checks. And pyarrow wheels are large (~100 MB); on a
constrained machine that is the one install that may need patience.

---

## Python version

**3.8 or newer.** The code deliberately avoids newer syntax so it runs on older
cluster Pythons: no `match` statements, no `X | Y` unions, no walrus operators,
no lowercase builtin generics outside annotations. An AST scan across all 40
source files confirms zero occurrences of each.

Python 2 is not supported and never was.

---

## Hardware

Nothing exotic. No GPU, no special instruction sets. It runs on a laptop; it
just runs faster on more cores.

### CPU

The sweep is **embarrassingly parallel** — each run is independent — so runtime
falls close to linearly with core count. `--n-cores` defaults to every core but
one, which keeps the machine usable.

### RAM

Measured, not estimated:

| what | how much |
|---|---|
| one worker process | ~100 MB (Python + numpy + pyarrow + scipy) |
| one algorithm's whole trace, in memory during compaction | ~80 MB at 300 rounds |

**The worker pool dominates.** Budget roughly:

| `--n-cores` | approximate RAM for the pool |
|---|---|
| 4 | 0.4 GB |
| 8 | 0.8 GB |
| 16 | 1.6 GB |

So **4 GB of RAM is comfortable at 16 cores**, and 2 GB is enough at 8. If you
are memory-constrained, lower `--n-cores` — that is the knob, not anything
internal.

### Disk

| | per algorithm | all 26 |
|---|---|---|
| trace (300 rounds) | ~27 MB | ~0.69 GB |
| results + summary | a few MB | negligible |

Peak usage while running is roughly **twice** the final size, because the
checkpoint files and the compacted output briefly coexist. Budget about **2 GB**
per switch-limit condition. The stationary runner in this folder produces
roughly a third of that.

Traces compress well — 30 bytes per row on disk against 88 in memory — because
a team that changes two roles per round is enormously repetitive and Parquet
exploits that.

### Network

None. Nothing downloads anything at run time. Once the libraries are
installed the folder is fully self-contained and runs offline.

---

## Verifying the install

```bash
python3 verify_setup.py
```

Checks, in order: every dependency imports and reports its version; the shared
settings file matches its digest and is reproducible from its seed; all 26
algorithms load it; the noise stream is fixed per problem; 1 core and 4 cores
produce byte-identical output; a crashed run resumes to a byte-identical
result; and the trace records what it claims to.

Exit code 0 means everything passed. Run it after copying the folder to a new
machine — it is much faster than discovering a problem three hours into a sweep.

---

## Environment gotchas

**BLAS threads are capped automatically.** Each worker is pinned to one BLAS
thread by setting `OMP_NUM_THREADS` and friends before numpy is imported.
Without that, every worker spawns its own thread pool, they fight each other for
cores, and *raising* `--n-cores` makes the sweep slower. If you set those
variables yourself, the code respects your values rather than overriding them.

**Multiprocessing uses the `spawn` start method**, which is safe with numpy and
BLAS and is the default on macOS anyway. One consequence matters if you write
your own driver script:

```python
# your_script.py
if __name__ == "__main__":      # <-- this guard is REQUIRED
    run_experiment(settings)
```

`spawn` re-imports the main module in each worker. Without the guard the sweep
restarts inside every worker. For the same reason you **cannot drive a
multi-core sweep from `python3 -` (stdin), a REPL, or a notebook cell** —
workers try to re-import `<stdin>` and fail. Use a real `.py` file, or set
`--n-cores 1`. The shipped entry points already have the guard.

**macOS.** Works out of the box. `spawn` is already the default, so the guard
requirement applies there regardless. Install pyarrow from PyPI; Apple Silicon
wheels are published.

**Windows.** The Python is portable and should run, but it has not been tested.
`run_full_sweep.sh` is a bash script — use WSL, Git Bash, or call
`run_perturbation_experiment.py --all` directly, which does the same sweep
without the aggregation and plotting steps. The `spawn` guard matters
even more on Windows, where it is the only start method.

**Virtual environments** are recommended but not required:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pyarrow tqdm pandas matplotlib
```

**Conda** works equally well:

```bash
conda create -n bandits python=3.11 numpy scipy pyarrow pandas matplotlib tqdm -c conda-forge
conda activate bandits
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'bandits_unconstrained'`**

An old copy. Early versions of this folder imported through a parent package
and only worked while nested inside a directory of that name, so unzipping it
anywhere else failed. The folder is now self-contained — replace it with a
current copy and the error goes away. Nothing else changes.

**`error: unrecognized arguments: --something`**

argparse rejects unknown flags rather than ignoring them. Check `INPUTS.md` or
run `python3 run_perturbation_experiment.py --help`. Note there is **no `--bandits` flag**: team size is
a property of each sampled problem, not of a run (see "Fixing the team size" in
`INPUTS.md`).

**`ModuleNotFoundError: No module named 'pyarrow'`** (or numpy, pandas, scipy)

Install it — see the table above. `python3 verify_setup.py` reports every
missing dependency at once rather than one per run.

**The sweep seems to hang with no progress bar**

`tqdm` is probably missing; the sweep runs but prints nothing. The startup
banner says so explicitly. Install `tqdm`, or watch the `_parts/` directory
fill up.

**A run restarts from scratch instead of resuming**

Resume matches on the exact configuration, which is encoded in the filename. If
you changed `--rounds`, `--n-settings`, `--seed`, `--switch-limit` or
`--max-changes`, that is a different run and starts fresh by design.

---

## Offline and cluster installs

If the machine has no internet, download the wheels elsewhere and carry them:

```bash
# on a connected machine, matching the target's Python and platform
pip download numpy scipy pyarrow tqdm pandas matplotlib -d wheels/

# on the target
pip install --no-index --find-links=wheels/ numpy scipy pyarrow tqdm pandas matplotlib
```

On a shared cluster, prefer a module load plus a virtual environment over
installing into the system Python:

```bash
module load python/3.11
python3 -m venv ~/bandit-env && source ~/bandit-env/bin/activate
pip install numpy scipy pyarrow tqdm pandas matplotlib
```

Two notes for batch schedulers. Set `--n-cores` to the cores your job was
actually allocated, not the machine's total — the default of "all but one" will
oversubscribe a shared node. And the sweep is safely restartable, so a job that
hits a wall-clock limit can simply be resubmitted with the same command and will
continue from where it stopped.

---

## Summary

| | |
|---|---|
| Python | 3.8+ |
| Install | `pip install numpy scipy pyarrow tqdm pandas` (+ `matplotlib` for plots) |
| Check | `python3 verify_setup.py` |
| RAM | ~100 MB per worker; 4 GB comfortable at 16 cores |
| Disk | ~2 GB per full suite, including working space |
| Network | none needed at run time |
| GPU | none |
