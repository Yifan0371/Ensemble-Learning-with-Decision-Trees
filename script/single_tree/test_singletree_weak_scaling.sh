#!/bin/bash
set -euo pipefail

# =============================================================================
# script/single_tree/test_singletree_weak_scaling.sh
#
# Weak-scaling test: Measures execution time and MSE for varying dataset sizes
# (scaled with thread count) on the same base data.
# =============================================================================

# 1) Project root path and executable location
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh" # Source OpenMP thread config

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

# 3) Calculate total rows (excluding header) and base rows per thread
total_rows=$(( $(wc -l < "$DATA") - 1 ))
if (( total_rows < OMP_NUM_THREADS )); then
  echo "Warning: Data rows ($total_rows) are less than physical cores ($OMP_NUM_THREADS), exiting script."
  exit 1
fi
BASE=$(( total_rows / OMP_NUM_THREADS ))

# 4) Generate thread list: 1, 2, 4, ..., MAX_CORES; ensure MAX_CORES is included
MAX_CORES=$OMP_NUM_THREADS
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
  threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 5) Print table header
echo "Threads | SubsetRows | Elapsed(ms) | MSE"
echo "---------------------------------------------"

# Loop through each thread count
for t in "${threads[@]}"; do
  export OMP_NUM_THREADS=$t # Set OMP_NUM_THREADS for the current test

  # Calculate the subset size for this run
  chunk_size=$(( t * BASE ))
  if (( chunk_size > total_rows )); then
    chunk_size=$total_rows # Do not exceed total rows
  fi
  lines_to_take=$(( chunk_size + 1 ))  # Include the header row

  tmpfile="$PROJECT_ROOT/data/data_clean/tmp_singletree_t${t}_$$.csv"
  head -n "$lines_to_take" "$DATA" > "$tmpfile" # Create a temporary subset file

  start_ts=$(date +%s%3N) # Start timestamp (milliseconds)
  # Execute the tree trainer with specified parameters
  output=$("$EXECUTABLE" single "$tmpfile" \
      $MAX_DEPTH $MIN_LEAF "$CRITERION" "$FINDER" \
      "$PRUNER" 0 "$VAL_SPLIT")
  end_ts=$(date +%s%3N)   # End timestamp (milliseconds)
  elapsed=$(( end_ts - start_ts )) # Calculate elapsed time

  rm -f "$tmpfile" # Clean up temporary file

  # Extract MSE from the output (tail -1 to get the final MSE if multiple are printed)
  mse=$(echo "$output" | grep -Eo "^MSE:\s*[0-9.+-eE]+" | awk -F'[: ]+' '{print $2}' | tail -1)
  [[ -z "$mse" ]] && mse="N/A" # Set to N/A if MSE is not found

  # Print results in a formatted table row
  printf "%7d | %10d | %11d | %s\n" "$t" "$chunk_size" "$elapsed" "$mse"
done

exit 0