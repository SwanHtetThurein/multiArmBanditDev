#!/bin/bash
# Run the 5 constrained variants in parallel, 2 at a time (to leave room
# for the main sweep's COMBO). Each writes its own CSV.
set -e
cd "$(dirname "$0")"

mkdir -p "Perturbation results"

SEED=42

# 2-process pipeline: launch constrained_bocs + constrained_combo in
# foreground (since combo is the slowest, runs them serially), and the
# remaining 3 neural constrained variants in the background in pairs.
# Use --seed offsets so each parallel run sees a different RNG sequence
# (still deterministic).

run_one() {
  local algo="$1"
  local seed="$2"
  python3 run_perturbation_experiment.py \
    --algorithm "$algo" \
    --bandits 9 \
    --rounds 300 \
    --perturbation_round 101 \
    --p_perturb_dims 1 \
    --tests 9 \
    --runs 20 \
    --noise 0.0 0.2 0.4 0.6 0.8 1.0 \
    --seed "$seed"
}

# 4-way parallel: combo_constrained (slow) + 3 constrained neural methods.
# bocs_constrained is left to the main sweep to avoid duplicate work.
(run_one combo_constrained 43) > "Perturbation results/combo_constrained.log" 2>&1 &
PID1=$!

sleep 5
(run_one neurallinear_constrained 44) > "Perturbation results/neurallinear_constrained.log" 2>&1 &
PID2=$!

sleep 5
(run_one neuralucb_constrained 45) > "Perturbation results/neuralucb_constrained.log" 2>&1 &
PID3=$!

sleep 5
(run_one neuralts_constrained 46) > "Perturbation results/neuralts_constrained.log" 2>&1 &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "constrained parallel batch done"
