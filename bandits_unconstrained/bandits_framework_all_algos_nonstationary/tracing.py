"""Per-round trace recording, in Parquet.

WHAT IS RECORDED
================
The results file answers "how well did it score". The trace answers "what did
it actually do". One row per round of every run, with:

  identity      algorithm, setting_id, n_bandits, noise_level, round
  the decision  requested_team  -- the team the algorithm ASKED for
                played_team     -- the team it was ALLOWED to play
  the constraint n_requested_changes, n_played_changes, blocked_dims,
                budget, allowance
  the feedback  reward
  the guess     predict_best, n_correct, performance, hamming_to_optimum
  internals     diag_* columns, one per algorithm-specific quantity

`requested_team` vs `played_team` is the point of the whole file: under a
switching limit the two differ, and `blocked_dims` says exactly which changes
the environment refused. With no limit in force they are always equal and the
columns compress to nearly nothing.

TRACE LEVELS
============
  off    only the columns the results table needs. No teams, no constraint
         detail, no diagnostics, and no separate trace file is written.
  basic  the full per-round record above, minus the diag_* columns
  full   (default) also calls each algorithm's diagnostics() every round

Note that `off` does NOT mean "record nothing": the results table is a
projection of these rows, so the per-round score is always recorded. Only the
detail is dropped. The three levels produce IDENTICAL results -- the level
changes what is kept, never what is run.

`diagnostics()` returns a flat dict of scalars, so each algorithm's trace file
gets its own `diag_<key>` columns. Schemas differ BETWEEN algorithms, which is
fine because each algorithm writes its own file.

There is deliberately no level that dumps full model state (a posterior weight
vector, a covariance matrix) every round. For the BOCS-family surrogates that
is ~470 floats per round, which at 300,000 rounds is over half a gigabyte per
algorithm for a single field. If that is ever wanted it belongs behind an
explicit flag with a stride, not in the default path.

WHY PARQUET
===========
Columnar and compressed. Measured on 300k realistic rows: 5 MB against 24 MB
for the same data as CSV, and reading three columns is ~29x faster because it
only touches those columns' bytes. For a file you will query repeatedly and
never re-generate, that is the whole ballgame.

Note on floats: noise levels are stored as float64 AND as an integer
`noise_pct` (0..100). float64 holds the standard levels exactly, so
`noise_level == 0.4` does match -- but float equality is fragile in general
(a narrowed float32 column, or a level produced by arithmetic, matches nothing
and returns zero rows rather than an error). `noise_pct` exists so filtering
never depends on that; prefer it.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

TRACE_OFF = "off"
TRACE_BASIC = "basic"
TRACE_FULL = "full"
TRACE_LEVELS = (TRACE_OFF, TRACE_BASIC, TRACE_FULL)
DEFAULT_TRACE_LEVEL = TRACE_FULL

COMPRESSION = "zstd"
COMPRESSION_LEVEL = 9


# ── schema ───────────────────────────────────────────────────────────────────

#: Columns every trace has, in order. Extra diag_* columns are appended.
BASE_FIELDS = [
    ("run_id", pa.int32()),
    ("setting_id", pa.int32()),
    ("n_bandits", pa.int8()),
    ("noise_level", pa.float64()),
    ("noise_pct", pa.int16()),
    ("round", pa.int32()),
    ("requested_team", pa.list_(pa.int8())),
    ("played_team", pa.list_(pa.int8())),
    ("n_requested_changes", pa.int8()),
    ("n_played_changes", pa.int8()),
    ("blocked_dims", pa.list_(pa.int8())),
    ("budget", pa.float32()),
    ("allowance", pa.int16()),
    ("reward", pa.float64()),
    ("predict_best", pa.list_(pa.int8())),
    ("n_correct", pa.int8()),
    ("performance", pa.float64()),
    ("hamming_to_optimum", pa.int8()),
    ("cum_changes", pa.int32()),
]

#: The subset the results table needs. At trace level `off` only these are
#: kept, so a sweep still produces results without the per-round detail.
RESULTS_FIELDS = [
    ("run_id", pa.int32()),
    ("setting_id", pa.int32()),
    ("n_bandits", pa.int8()),
    ("noise_level", pa.float64()),
    ("noise_pct", pa.int16()),
    ("round", pa.int32()),
    ("performance", pa.float64()),
]

#: Extra columns the non-stationary driver adds.
PERTURBATION_FIELDS = [
    ("phase", pa.string()),
    ("baseline", pa.float64()),
    ("optimum_version", pa.int8()),
]


def _arrow_type_for(values) -> pa.DataType:
    """Pick a column type for a diagnostic, from the values actually seen."""
    seen = [v for v in values if v is not None]
    if not seen:
        return pa.float64()
    if all(isinstance(v, bool) for v in seen):
        return pa.bool_()
    if all(isinstance(v, (int,)) and not isinstance(v, bool) for v in seen):
        return pa.int64()
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seen):
        return pa.float64()
    return pa.string()


class TraceBuffer:
    """Collects one task's rows, then converts them to an Arrow table.

    A task is one (setting, noise) run, i.e. `total_rounds` rows. Buffering a
    whole task in memory is fine -- 300 rows of small values -- and lets the
    part file be written in a single atomic operation.
    """

    def __init__(self, level: str = DEFAULT_TRACE_LEVEL,
                 extra_fields: Sequence = ()):
        if level not in TRACE_LEVELS:
            raise ValueError(f"trace level must be one of {TRACE_LEVELS}, got '{level}'")
        self.level = level
        base = RESULTS_FIELDS if level == TRACE_OFF else BASE_FIELDS
        self.fields = list(base) + list(extra_fields)
        self.rows: List[dict] = []
        self._diag_keys: List[str] = []

    @property
    def detailed(self) -> bool:
        """True when the per-round detail columns are being kept."""
        return self.level != TRACE_OFF

    @property
    def wants_diagnostics(self) -> bool:
        return self.level == TRACE_FULL

    def add(self, row: dict, diagnostics: Optional[dict] = None) -> None:
        if self.level == TRACE_FULL and diagnostics:
            for k, v in diagnostics.items():
                key = f"diag_{k}"
                if key not in row:
                    row[key] = v
                if key not in self._diag_keys:
                    self._diag_keys.append(key)
        self.rows.append(row)

    def to_table(self) -> pa.Table:
        cols, schema = {}, []
        for name, typ in self.fields:
            cols[name] = pa.array([r.get(name) for r in self.rows], typ)
            schema.append(pa.field(name, typ))
        for key in self._diag_keys:
            vals = [r.get(key) for r in self.rows]
            typ = _arrow_type_for(vals)
            if typ == pa.string():
                vals = [None if v is None else str(v) for v in vals]
            cols[key] = pa.array(vals, typ)
            schema.append(pa.field(key, typ))
        return pa.table(cols, schema=pa.schema(schema))


# ── writing parts ────────────────────────────────────────────────────────────

def write_part(table: pa.Table, path: str) -> None:
    """Write one part atomically: temp file, then rename.

    A crash therefore leaves either a complete part or no part at all -- never
    a half-written one. This is what makes the trace usable as the resume
    checkpoint.
    """
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression=COMPRESSION,
                   compression_level=COMPRESSION_LEVEL)
    os.replace(tmp, path)


def part_path(parts_dir: str, task_id: str) -> str:
    return os.path.join(parts_dir, f"part-{task_id}.parquet")


def completed_parts(parts_dir: str) -> set:
    """task_ids with a complete part on disk."""
    if not os.path.isdir(parts_dir):
        return set()
    out = set()
    for fn in os.listdir(parts_dir):
        if fn.startswith("part-") and fn.endswith(".parquet"):
            out.add(fn[len("part-"):-len(".parquet")])
    return out


def sweep_tmp_files(parts_dir: str) -> int:
    """Remove .tmp leftovers from a crash mid-write."""
    if not os.path.isdir(parts_dir):
        return 0
    n = 0
    for fn in os.listdir(parts_dir):
        if fn.endswith(".tmp"):
            try:
                os.remove(os.path.join(parts_dir, fn)); n += 1
            except OSError:
                pass
    return n


# ── compaction ───────────────────────────────────────────────────────────────

SORT_KEYS = [("noise_pct", "ascending"), ("setting_id", "ascending"),
             ("round", "ascending")]


def compact(parts_dir: str, out_path: str, keep: Optional[set] = None,
            batch_size: int = 200) -> int:
    """Merge every part into one Parquet file, sorted deterministically.

    Sorting matters: without it the row order would depend on which worker
    finished first, and two identical runs would produce different files.

    Parts are read in batches rather than all at once, so peak memory stays
    bounded even for a 900,000-row trace.
    """
    names = sorted(completed_parts(parts_dir) if keep is None else keep)
    if not names:
        return 0

    tables = []
    for i in range(0, len(names), batch_size):
        chunk = [pq.read_table(part_path(parts_dir, n)) for n in names[i:i + batch_size]]
        tables.append(pa.concat_tables(chunk, promote_options="default"))
    table = pa.concat_tables(tables, promote_options="default")
    table = table.sort_by(SORT_KEYS)

    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression=COMPRESSION,
                   compression_level=COMPRESSION_LEVEL)
    os.replace(tmp, out_path)
    return table.num_rows


def remove_parts(parts_dir: str) -> None:
    if not os.path.isdir(parts_dir):
        return
    for fn in os.listdir(parts_dir):
        try:
            os.remove(os.path.join(parts_dir, fn))
        except OSError:
            pass
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass


# ── the results projection ───────────────────────────────────────────────────

#: `noise_pct` is carried through deliberately: OUTPUTS.md tells people to
#: filter on the integer rather than the float, and that advice has to work on
#: the results file as well as on the trace.
RESULTS_COLUMNS = ["run_id", "setting_id", "n_bandits", "noise_level", "noise_pct",
                   "round", "performance"]
PERTURBATION_RESULTS_COLUMNS = RESULTS_COLUMNS + ["phase", "baseline"]


def derive_results(trace_path: str, out_path: str,
                   columns: Optional[Sequence[str]] = None) -> int:
    """Write the small results table as a projection of the trace.

    The results file holds nothing the trace does not; it exists because it is
    the thing you actually plot, and reading six columns out of a 19-column
    file every time is wasteful.
    """
    columns = list(columns or RESULTS_COLUMNS)
    table = pq.read_table(trace_path, columns=columns)
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression=COMPRESSION,
                   compression_level=COMPRESSION_LEVEL)
    os.replace(tmp, out_path)
    return table.num_rows


# ── run manifest ─────────────────────────────────────────────────────────────

def write_manifest(path: str, payload: Dict) -> None:
    """Record how a run was produced, so a CSV found months later can be
    attributed without guesswork."""
    doc = dict(payload)
    doc.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    doc.setdefault("host", socket.gethostname())
    doc.setdefault("platform", platform.platform())
    doc.setdefault("python", sys.version.split()[0])
    try:
        import numpy
        doc.setdefault("numpy", numpy.__version__)
    except ImportError:
        pass
    doc.setdefault("pyarrow", pa.__version__)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, path)


# ── reading, for analysis ────────────────────────────────────────────────────

def read(path: str, columns: Optional[Sequence[str]] = None,
         noise_pct: Optional[int] = None, setting_id: Optional[int] = None):
    """Load a trace or results file as a pandas DataFrame.

        from tracing import read
        df = read("Global results/bocs_...trace.parquet",
                  columns=["round", "performance"], noise_pct=40)

    Filter on `noise_pct` (an integer) rather than on `noise_level`: float64
    holds the standard levels exactly, but the integer column cannot be tripped
    up by a narrowed dtype or a computed level.
    """
    filters = []
    if noise_pct is not None:
        filters.append(("noise_pct", "=", int(noise_pct)))
    if setting_id is not None:
        filters.append(("setting_id", "=", int(setting_id)))
    table = pq.read_table(path, columns=list(columns) if columns else None,
                          filters=filters or None)
    return table.to_pandas()


def summarize(path: str) -> str:
    md = pq.read_metadata(path)
    sch = pq.read_schema(path)
    diag = [n for n in sch.names if n.startswith("diag_")]
    return (f"{os.path.basename(path)}\n"
            f"  {md.num_rows:,} rows x {len(sch.names)} columns, "
            f"{os.path.getsize(path)/1e6:.1f} MB\n"
            f"  diagnostics: {', '.join(d[5:] for d in diag) if diag else '(none)'}")
