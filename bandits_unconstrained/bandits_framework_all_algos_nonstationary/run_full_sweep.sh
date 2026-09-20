#!/bin/bash
# Run the full 26-algorithm NON-STATIONARY sweep, then aggregate + plot.
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
#   SWITCH_LIMIT=flat MAX_CHANGES=2 bash run_full_sweep.sh   # under a switching limit
#   N_CORES=8 bash run_full_sweep.sh       # use 8 worker processes
#
# RESUMING. Press Ctrl-C at any time. Every run is checkpointed, so rerunning
# the exact same command continues from where it stopped rather than starting
# over. That applies within an algorithm as well as across them.
#
# SWITCH_LIMIT is none (default), flat, or parabolic: a cap on how many roles
# may change per round, applied by the environment to every algorithm equally.
# MAX_CHANGES (default 2) is the cap for flat, or the mid-run peak for
# parabolic.
#
# NOTE ON `parabolic` IN THE NON-STATIONARY SETTING. The parabolic allowance is
# zero at round 1, peaks at T/2 and returns to zero at round T. The optimum
# moves at rounds 101 and 201, and the budget knows nothing about that, so the
# second perturbation has to be recovered from on a shrinking allowance. That
# confounds a comparison of recovery times BETWEEN the two perturbations. Use
# `flat` for recovery work unless that interaction is what you are studying.
#
# Results are tagged with the mode in the filename (e.g.
# bocs_limit-flat2_perturbation_...), so conditions never overwrite each other,
# and the aggregator reports them as separate rows ("bocs [flat2]").
#
# WARNING ON RUNTIME. 500 settings x 6 noise levels = 3000 runs per algorithm,
# at 300 rounds each, across 26 algorithms. This is a long job even in
# parallel: set N_CORES to the number of cores you can spare and expect it to
# run for many hours. It is safe to stop and restart -- see RESUMING above.
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
SWITCH_LIMIT=${SWITCH_LIMIT:-none}
N_CORES=${N_CORES:-0}          # 0 = all cores but one
N_SETTINGS=${N_SETTINGS:-500}
MAX_CHANGES=${MAX_CHANGES:-2}
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
  echo "Running $algo (seed=$seed, switch-limit=$SWITCH_LIMIT/$MAX_CHANGES, cores=$N_CORES)"
  echo "============================================================"
  python3 run_perturbation_experiment.py \
    --algorithm "$algo" \
    --rounds 300 \
    --perturbation_rounds 101 201 \
    --p_perturb_dims 1 \
    --n-settings "$N_SETTINGS" \
    --noise 0.0 0.2 0.4 0.6 0.8 1.0 \
    --seed "$seed" \
    --switch-limit "$SWITCH_LIMIT" \
    --max-changes "$MAX_CHANGES" \
    --n-cores "$N_CORES"
}

for algo in "${ALGO_LIST[@]}"; do
  run_one "$algo" "$SEED" 2>&1 | tee "$OUTPUT_DIR/${algo}_limit-${SWITCH_LIMIT}.log"
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
