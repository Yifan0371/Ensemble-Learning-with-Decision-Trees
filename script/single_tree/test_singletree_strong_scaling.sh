#!/bin/bash
set -euo pipefail

# =============================================================================
# script/single_tree/test_singletree_strong_scaling.sh
#
# Strong-scaling test: Fixed dataset size, measures execution time and MSE
# with varying thread counts.
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

MAX_DEPTH=20
MIN_LEAF=5
CRITERION="mse"
FINDER="exhaustive"
PRUNER="none"
VAL_SPLIT=0.2

# 3) Generate thread list: 1, 2, 4, ..., MAX_CORES; ensure MAX_CORES is included
MAX_CORES=$OMP_NUM_THREADS
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 4) Print table header
echo "Threads | Elapsed(ms) | MSE"
echo "------------------------------------"

# Loop through each thread count
for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t # Set OMP_NUM_THREADS for the current test
  start_ts=$(date +%s%3N) # Start timestamp (milliseconds)

  # Execute the tree trainer with specified parameters on the full dataset
  output=$("$EXECUTABLE" single "$DATA" \
      $MAX_DEPTH $MIN_LEAF "$CRITERION" "$FINDER" \
      "$PRUNER" 0 "$VAL_SPLIT")

  end_ts=$(date +%s%3N) # End timestamp (milliseconds)
  elapsed=$(( end_ts - start_ts )) # Calculate elapsed time

  # Extract "MSE: <num>" line and then the numerical value
  mse=$(echo "$output" | grep -Eo "^MSE:\s*[0-9.+-eE]+" | awk -F'[: ]+' '{print $2}' | tail -1)
  [[ -z "$mse" ]] && mse="N/A" # Set to N/A if MSE is not found

  # Print results in a formatted table row
  printf "%7d | %11d | %s\n" "$t" "$elapsed" "$mse"
done

exit 0