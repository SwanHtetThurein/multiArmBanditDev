"""Generate a settings file whose every setting has 9 dimensions.

The standard protocol samples n_bandits from {3, 6, 9}, so the default file is
a mix. Run this once if you want a sweep over 9-dimension problems only:

    python3 make_9dim_settings.py

then point runs at the file it writes:

    python3 run_experiment.py --all --settings-path bandit_settings_n500_seed20240501_dims9.json

Everything else about the protocol is unchanged: arm counts are still drawn
uniformly from 2-5, every dimension is still 'ongoing', and each setting still
carries its own initial_bias and optimal_arm.

This is a DIFFERENT benchmark from the standard file -- different problems, so
a different digest. Do not compare results across the two.
"""

import argparse
import os

from sampling import (DEFAULT_SETTINGS_SEED, N_SETTINGS, sample_settings,
                      settings_digest, write_settings, summarize)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dims", type=int, default=9,
                    help="dimensions every setting should have (default: 9)")
    ap.add_argument("--n-settings", type=int, default=N_SETTINGS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SETTINGS_SEED)
    a = ap.parse_args()

    settings = sample_settings(n_settings=a.n_settings, seed=a.seed,
                               bandit_choices=(a.dims,))
    path = os.path.join(
        HERE, f"bandit_settings_n{a.n_settings}_seed{a.seed}_dims{a.dims}.json")
    digest = write_settings(path, settings, a.seed)
    print(f"wrote {os.path.basename(path)}")
    print(f"digest: {digest}")
    print(summarize(settings))
    print(f"\nUse it with:\n  --settings-path {os.path.basename(path)}")


if __name__ == "__main__":
    main()
