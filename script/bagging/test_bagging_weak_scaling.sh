#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_weak_scaling.sh
#
# Bagging Weak-scaling test: Measures execution time and MSE for varying dataset sizes
# (scaled with thread count) on the same base data.
# =============================================================================

# 1) Project root path and executable location
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh" # Source OpenMP thread configuration

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "Error: Executable not found at $EXECUTABLE"
  exit 1
fi

# 2) Fixed parameters
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA" ]]; then
  echo "Error: Data file not found at $DATA"
  exit 1
fi

NUM_TREES=20
SAMPLE_RATIO=1.0
MAX_DEPTH=10
MIN_LEAF=2
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
PRUNER_PARAM=0.01 # Example pruner parameter (e.g., for cost_complexity or mingain)
SEED=42 # Random seed for reproducibility

# 3) Total rows and base rows per thread
total_rows=$(( $(wc -l < "$DATA") - 1 )) # Count lines excluding header
if (( total_rows < OMP_NUM_THREADS )); then
  echo "Warning: Data rows ($total_rows) are less than physical cores ($OMP_NUM_THREADS), exiting script."
  exit 1
fi
BASE=$(( total_rows / OMP_NUM_THREADS )) # Base rows per physical core
echo "Total rows (excluding header): $total_rows, Physical cores: $OMP_NUM_THREADS, Base rows BASE=$BASE"

# 4) Generate thread list: 1, 2, 4, ..., MAX_CORES; ensure MAX_CORES is included
threads=(1)
while (( threads[-1]*2 <= OMP_NUM_THREADS )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != OMP_NUM_THREADS )) && threads+=( $OMP_NUM_THREADS )

# 5) Print table header
echo "==============================================="
echo "    Bagging Weak Scaling Performance Test     "
echo "==============================================="
echo "Fixed Parameters (per thread):"
echo "  Trees: $NUM_TREES | Sample Ratio: $SAMPLE_RATIO"
echo "  Max Depth: $MAX_DEPTH | Min Leaf: $MIN_LEAF"
echo "  Criterion: $CRITERION | Finder: $FINDER"
echo "  Base rows per thread: $BASE"
echo ""
echo "Threads | SubsetRows | Elapsed(ms) | TestMSE    | TestMAE    | OOB_MSE    | Efficiency"
echo "--------|------------|-------------|------------|------------|------------|----------"

baseline_time=0 # Initialize baseline time for efficiency calculation
for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t # Set OMP_NUM_THREADS for the current test run

  # Calculate the subset size for this iteration
  chunk_size=$(( t * BASE ))
  if (( chunk_size > total_rows )); then
    chunk_size=$total_rows # Do not exceed total available rows
  fi
  lines_to_take=$(( chunk_size + 1 )) # Include the header row for 'head' command

  # Create a temporary subset file
  tmpfile="$PROJECT_ROOT/data/data_clean/tmp_bagging_t${t}_$$.csv"
  head -n "$lines_to_take" "$DATA" > "$tmpfile"

  start_ts=$(date +%s%3N) # Start timestamp (milliseconds)
  # Execute the Bagging trainer
  output=$("$EXECUTABLE" bagging "$tmpfile" \
      $NUM_TREES $SAMPLE_RATIO $MAX_DEPTH $MIN_LEAF \
      "$CRITERION" "$FINDER" "$PRUNER" $PRUNER_PARAM $SEED)
  end_ts=$(date +%s%3N)   # End timestamp (milliseconds)
  elapsed=$(( end_ts - start_ts )) # Calculate elapsed time

  rm -f "$tmpfile" # Clean up the temporary file

  # Extract performance metrics from the output using grep and sed/awk
  test_mse=$(echo "$output" | grep -E "Test MSE:" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  test_mae=$(echo "$output" | grep -E "Test MAE:" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
  oob_mse=$(echo "$output" | grep -E "OOB MSE:" | sed -n 's/.*OOB MSE: *\([0-9.-]*\).*/\1/p' | tail -1)

  # Assign "ERROR" if metrics are not found in the output
  [[ -z "$test_mse" ]] && test_mse="ERROR"
  [[ -z "$test_mae" ]] && test_mae="ERROR"
  [[ -z "$oob_mse" ]] && oob_mse="ERROR"

  # Calculate efficiency
  if (( t == 1 )); then
    baseline_time=$elapsed # Set baseline time for the single-thread run
    efficiency="1.00"
  else
    if (( baseline_time > 0 && elapsed > 0 )); then
      efficiency=$(echo "scale=2; $baseline_time / $elapsed" | bc -l) # Efficiency = Baseline Time / Current Time
    else
      efficiency="N/A" # Handle cases where time is zero or error
    fi
  fi

  # Print formatted results
  printf "%7d | %10d | %11d | %-10s | %-10s | %-10s | %s\n" \
         "$t" "$chunk_size" "$elapsed" "$test_mse" "$test_mae" "$oob_mse" "$efficiency"
done

echo ""
echo "==============================================="
echo "Weak Scaling Analysis:"
echo "- Ideal: Time remains constant as threads and data increase."
echo "- Efficiency = (Single-thread time / Current time)"
echo "- Focus: Efficiency close to 1.0 indicates good scaling."
echo "==============================================="

exit 0