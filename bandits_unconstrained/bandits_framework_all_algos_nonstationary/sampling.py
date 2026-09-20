"""The sampling protocol: 500 bandit settings, shared by every algorithm.

The protocol
------------
Each *setting* is drawn independently:

  1. n_bandits is picked uniformly from {3, 6, 9}.
  2. Each bandit's arm count is picked uniformly from 2..5.
  3. Every bandit is 'ongoing'. There is no early/late split: the published
     DreamTeam algorithm's temporal schedule gives an 'ongoing' dimension a
     discount of d = 1, which makes its renormalization the identity, so the
     schedule is simply not applied (see algorithms/dreamteam.py).

A setting also carries the two vectors a run needs, drawn at sampling time so
they are fixed and shared like everything else:

  4. initial_bias  -- the team the algorithm starts from, one arm per bandit.
  5. optimal_arm   -- the hidden answer, one arm per bandit.

500 settings are drawn. One run = one (setting, noise level) pair, so the full
sweep is 500 x 6 = 3000 runs per algorithm. There are no repetitions: the 500
independent settings supply more instance diversity than the old design's
9 tests x 20 repetitions = 180 instances ever did, and they do it without
re-measuring the same instance.

Why the settings are written to a file
--------------------------------------
Every algorithm must face the identical settings. Regenerating them from a
seed in each process would *probably* work, but it silently depends on the
Python version's RNG internals and on nothing ever reordering a draw. Instead
the settings are generated once, written to JSON, and loaded by every run.

That makes the guarantee checkable rather than assumed: the file has a SHA-256
digest recorded inside it, every results CSV records that digest, and
`verify_settings()` re-derives it. If two result sets carry the same digest,
they faced the same problems -- no reasoning about RNG required.

The file is deterministic given the seed, so both the stationary and the
non-stationary folder produce byte-identical settings at the same seed, and
results from the two can be compared directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Sequence

#: The protocol's constants.
BANDIT_CHOICES: Sequence[int] = (3, 6, 9)
MIN_ARMS = 2
MAX_ARMS = 5
N_SETTINGS = 500
ONGOING = "ongoing"

#: Default seed for the settings file. Changing this changes the benchmark.
DEFAULT_SETTINGS_SEED = 20240501

SETTINGS_VERSION = 1


def sample_settings(n_settings: int = N_SETTINGS,
                    seed: int = DEFAULT_SETTINGS_SEED,
                    bandit_choices: Sequence[int] = BANDIT_CHOICES,
                    min_arms: int = MIN_ARMS,
                    max_arms: int = MAX_ARMS) -> List[Dict]:
    """Draw the settings. Uses its own Random instance, so it cannot be
    perturbed by anything else that touches the global RNG."""
    rng = random.Random(seed)
    settings = []
    for i in range(n_settings):
        n_bandits = rng.choice(list(bandit_choices))
        arm_counts = [rng.randint(min_arms, max_arms) for _ in range(n_bandits)]
        initial_bias = [rng.randrange(n) for n in arm_counts]
        optimal_arm = [rng.randrange(n) for n in arm_counts]
        settings.append({
            "setting_id": i,
            "n_bandits": n_bandits,
            "arm_counts": arm_counts,
            "bandit_types": [ONGOING] * n_bandits,
            "initial_bias": initial_bias,
            "optimal_arm": optimal_arm,
        })
    return settings


def settings_digest(settings: List[Dict]) -> str:
    """SHA-256 over the settings' content. Two runs quoting the same digest
    provably faced the same problems."""
    payload = json.dumps(
        [[s["setting_id"], s["n_bandits"], s["arm_counts"],
          s["initial_bias"], s["optimal_arm"]] for s in settings],
        separators=(",", ":"), sort_keys=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_settings_path(base_dir: str, seed: int = DEFAULT_SETTINGS_SEED,
                          n_settings: int = N_SETTINGS) -> str:
    return os.path.join(base_dir, f"bandit_settings_n{n_settings}_seed{seed}.json")


def write_settings(path: str, settings: List[Dict], seed: int) -> str:
    doc = {
        "version": SETTINGS_VERSION,
        "seed": seed,
        "n_settings": len(settings),
        "protocol": {
            "bandit_choices": list(BANDIT_CHOICES),
            "arm_range": [MIN_ARMS, MAX_ARMS],
            "bandit_types": ONGOING,
            "note": "one initial_bias and optimal_arm per setting; no repetitions",
        },
        "digest": settings_digest(settings),
        "settings": settings,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)          # atomic: never leave a half-written file
    return doc["digest"]


def load_settings(path: str) -> Dict:
    with open(path, "r") as f:
        doc = json.load(f)
    recomputed = settings_digest(doc["settings"])
    if recomputed != doc.get("digest"):
        raise ValueError(
            f"settings file {path} is corrupt: digest {doc.get('digest')} "
            f"does not match its contents ({recomputed})"
        )
    return doc


def ensure_settings(base_dir: str, seed: int = DEFAULT_SETTINGS_SEED,
                    n_settings: int = N_SETTINGS,
                    path: Optional[str] = None) -> Dict:
    """Load the settings file, creating it first if it does not exist.

    Every driver calls this, so the first run writes the file and every later
    run -- and every other algorithm, and a resumed run -- loads that same
    file rather than re-drawing.
    """
    if path is None:
        path = default_settings_path(base_dir, seed, n_settings)
    if not os.path.exists(path):
        settings = sample_settings(n_settings=n_settings, seed=seed)
        write_settings(path, settings, seed)
    doc = load_settings(path)
    doc["path"] = path
    return doc


def verify_settings(path: str, seed: Optional[int] = None) -> Dict:
    """Re-derive the settings from the recorded seed and confirm the file on
    disk matches. This is what proves the file was not hand-edited."""
    doc = load_settings(path)
    seed = doc["seed"] if seed is None else seed
    fresh = sample_settings(n_settings=doc["n_settings"], seed=seed)
    ok = settings_digest(fresh) == doc["digest"]
    return {
        "path": path,
        "seed": seed,
        "n_settings": doc["n_settings"],
        "digest": doc["digest"],
        "reproducible_from_seed": ok,
    }


def summarize(settings: List[Dict]) -> str:
    from collections import Counter
    dims = Counter(s["n_bandits"] for s in settings)
    arms = Counter(a for s in settings for a in s["arm_counts"])
    total_dims = sum(s["n_bandits"] for s in settings)
    lines = [
        f"{len(settings)} settings",
        "  dimensions: " + ", ".join(f"{k}->{v}" for k, v in sorted(dims.items())),
        "  arm counts: " + ", ".join(f"{k}->{v}" for k, v in sorted(arms.items())),
        f"  mean dimensions per setting: {total_dims/len(settings):.2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate or inspect the shared settings file.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SETTINGS_SEED)
    ap.add_argument("--n-settings", type=int, default=N_SETTINGS)
    ap.add_argument("--verify", action="store_true",
                    help="check the file on disk still matches its seed")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    doc = ensure_settings(here, seed=a.seed, n_settings=a.n_settings)
    print(f"settings file: {doc['path']}")
    print(f"digest:        {doc['digest']}")
    print(summarize(doc["settings"]))
    if a.verify:
        print(verify_settings(doc["path"]))
