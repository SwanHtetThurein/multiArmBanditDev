# Perturbation Experiment — Cross-Algorithm Summary

Numbers below are means across the 180 runs per (algorithm, noise_level).
Recovery thresholds are relative to perturbation_round (round 101).
`recovered_pct` is the fraction of runs that hit the recovery threshold by round 300.

| algorithm | noise | n | baseline (mean) | recovery_rounds (mean ± std) | time_to_baseline (mean ± std) | max_post_dip (mean) | recovered_pct |
|---|---|---|---|---|---|---|---|
| dreamteam | 0.0 | 180 | 0.516 | 130.0 ± 39.2 | 22.4 ± 36.7 | 0.531 | 3.9% |
| dreamteam | 0.2 | 180 | 0.577 | 135.1 ± 37.3 | 31.1 ± 43.2 | 0.480 | 7.8% |
| dreamteam | 0.4 | 180 | 0.604 | 133.1 ± 43.2 | 37.7 ± 45.5 | 0.511 | 16.1% |
| dreamteam | 0.6 | 180 | 0.569 | 105.7 ± 42.2 | 25.3 ± 34.2 | 0.538 | 11.1% |
| dreamteam | 0.8 | 180 | 0.570 | 115.3 ± 53.8 | 23.0 ± 34.0 | 0.564 | 10.0% |
| dreamteam | 1.0 | 180 | 0.568 | 86.9 ± 33.9 | 23.5 ± 34.6 | 0.573 | 6.1% |
| random | 0.0 | 180 | 0.414 | 123.5 ± 59.3 | 2.2 ± 1.6 | 0.947 | 6.1% |
| random | 0.2 | 180 | 0.408 | 77.3 ± 71.8 | 2.2 ± 1.9 | 0.951 | 3.9% |
| random | 0.4 | 180 | 0.411 | 83.5 ± 91.8 | 2.1 ± 1.7 | 0.943 | 2.2% |
| random | 0.6 | 180 | 0.413 | 83.9 ± 60.4 | 2.3 ± 2.8 | 0.957 | 3.9% |
| random | 0.8 | 180 | 0.412 | 50.4 ± 42.4 | 2.3 ± 1.8 | 0.949 | 2.8% |
| random | 1.0 | 180 | 0.417 | 99.8 ± 46.9 | 2.2 ± 2.6 | 0.955 | 6.7% |
