#!/bin/bash
# Run the 7 unconstrained algorithms (dreamteam, random, bocs, combo,
# neurallinear, neuralucb, neuralts) sequentially, then aggregate + plot.
# Each algorithm writes its own full CSV and per-run summary CSV.
#
# Usage:
#   bash run_full_sweep.sh                # full sweep, default seed
#   SEED=123 bash run_full_sweep.sh       # different seed
#
# After the sweep, run the aggregator + plotter:
#   python3 aggregate_perturbation.py --results_dir "Unconstrained Perturbation Results"
#   python3 graph_perturbation.py \
#       --results_dir "Unconstrained Perturbation Results" \
#       --output_dir "Graphs"

set -e
cd "$(dirname "$0")"

SEED=${SEED:-42}
OUTPUT_DIR="Unconstrained Perturbation Results"
mkdir -p "$OUTPUT_DIR"

ALGOS=(
  "dreamteam"
  "random"
  "bocs"
  "combo"
  "neurallinear"
  "neuralucb"
  "neuralts"
)

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

# Sequential is fine — the slowest is combo, and we want to keep the machine
# usable while this runs. If you have a beefy box, parallelize by replacing
# this loop with `&` and a `wait`.
for algo in "${ALGOS[@]}"; do
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
