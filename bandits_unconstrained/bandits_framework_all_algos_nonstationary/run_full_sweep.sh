#!/bin/bash
# Run the full 27-algorithm NON-STATIONARY sweep, then aggregate + plot.
#
# The hidden optimal arm is swapped after round 101 and again after round 201,
# each time flipping 1 of the 9 dimensions. Algorithm state is never reset, so
# what is measured is how fast each method notices and recovers.
#
# Each algorithm writes its own per-round CSV and per-run recovery-summary CSV.
#
# Usage:
#   bash run_full_sweep.sh                 # full sweep, default seed
#   SEED=123 bash run_full_sweep.sh        # different seed
#   ALGOS="dreamteam bocs kg" bash run_full_sweep.sh   # subset
#
# WARNING ON RUNTIME. This is 300 rounds x 1080 runs per algorithm — three
# times the stationary sweep — across 27 algorithms. Expect this to run for
# many hours; `bocs` and `smac` alone are on the order of a couple of hours
# each at 300 rounds. Run a subset first (see ALGOS above), or overnight.
#
# After the sweep the aggregator and plotter run automatically; to redo just
# those:
#   python3 aggregate_perturbation.py --results_dir "Unconstrained Perturbation Results"
#   python3 graph_perturbation.py \
#       --results_dir "Unconstrained Perturbation Results" \
#       --output_dir "Graphs" --n_bandits 9 --p_perturb_dims 1

set -e
cd "$(dirname "$0")"

SEED=${SEED:-42}
OUTPUT_DIR="Unconstrained Perturbation Results"
mkdir -p "$OUTPUT_DIR"

# Ordered cheapest-first, so a partial run still yields a usable spread of
# results if you stop it early.
DEFAULT_ALGOS="\
random \
sa \
ols \
regevo \
cucb \
cts \
dreamteam \
dreamteam_orig \
cocabo \
purexp \
gp_onehot \
casmopolitan \
glm_fpl \
bayesgap \
gp_ts \
gp_ucb \
neurallinear \
kg \
bocs_hs \
bootnn \
gp_nei \
combo \
combo_slice \
linucb \
bocs \
sts \
smac"

read -r -a ALGO_LIST <<< "${ALGOS:-$DEFAULT_ALGOS}"

run_one() {
  local algo="$1"
  local seed="$2"
  echo "============================================================"
  echo "Running $algo (seed=$seed)"
  echo "============================================================"
  python3 run_perturbation_experiment.py \
    --algorithm "$algo" \
    --bandits 9 \
    --rounds 300 \
    --perturbation_rounds 101 201 \
    --p_perturb_dims 1 \
    --tests 9 \
    --runs 20 \
    --noise 0.0 0.2 0.4 0.6 0.8 1.0 \
    --seed "$seed"
}

for algo in "${ALGO_LIST[@]}"; do
  run_one "$algo" "$SEED" 2>&1 | tee "$OUTPUT_DIR/${algo}.log"
done

echo ""
echo "Sweep complete. Aggregating..."
python3 aggregate_perturbation.py --results_dir "$OUTPUT_DIR" --p_perturb_dims 1

echo ""
echo "Generating graphs..."
mkdir -p Graphs
python3 graph_perturbation.py --results_dir "$OUTPUT_DIR" --output_dir Graphs \
    --n_bandits 9 --p_perturb_dims 1

echo ""
echo "Done. Per-algorithm plots + recovery_summary.png are in Graphs/."
