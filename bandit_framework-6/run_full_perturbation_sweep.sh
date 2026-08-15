#!/bin/bash
# Run the full 12-algorithm perturbation sweep in the background.
# One CSV per algorithm is written to "Perturbation results/" as each
# algorithm finishes, so we have incremental results even if interrupted.

set -e
cd "$(dirname "$0")"

SEED=42

ALGOS=(
  dreamteam
  random
  bocs
  combo
  neurallinear
  neuralucb
  neuralts
  bocs_constrained
  combo_constrained
  neurallinear_constrained
  neuralucb_constrained
  neuralts_constrained
)

mkdir -p "Perturbation results"

for algo in "${ALGOS[@]}"; do
  echo "=== Running $algo ==="
  date
  python3 run_perturbation_experiment.py \
    --algorithm "$algo" \
    --bandits 9 \
    --rounds 300 \
    --perturbation_round 101 \
    --p_perturb_dims 1 \
    --tests 9 \
    --runs 20 \
    --noise 0.0 0.2 0.4 0.6 0.8 1.0 \
    --seed "$SEED"
  echo "=== Finished $algo ==="
  date
done

echo "=== Aggregating ==="
python3 aggregate_perturbation.py --results_dir "Perturbation results"
echo "=== All done ==="
