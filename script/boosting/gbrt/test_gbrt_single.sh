、#!/bin/bash
# =============================================================================
# script/boosting/gbrt/test_gbrt_single.sh
# GBRT (base without DART) Single-Thread Comprehensive Test Script
# Run from project root: bash script/boosting/gbrt/test_gbrt_single.sh
# =============================================================================

export OMP_NUM_THREADS=1 # Ensure single-thread operation

PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt"
OUTFILE="$RESULTS_DIR/gbrt_single_results.txt"
mkdir -p "$RESULTS_DIR" # Create results directory if it doesn't exist

echo "Config | TestMSE | TrainTime(ms) | TotalTime(ms) | Trees" > "$OUTFILE" # Write header to results file

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
  echo "Error: Executable not found at $EXECUTABLE"
  exit 1
fi
# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
  echo "Error: Data file not found at $DATA_PATH"
  exit 1
fi

# Function to run the executable and parse its output
run_and_parse() {
  local config_desc="$1" # Configuration description for logging
  shift
  local args=( "$@" )    # Remaining arguments passed to the executable

  local start_time=$(date +%s%3N) # Start timestamp in milliseconds
  local raw                  # Variable to capture raw output
  raw=$("$EXECUTABLE" "${args[@]}" 2>&1) # Execute the GBRT program and capture stdout/stderr
  local end_time=$(date +%s%3N)     # End timestamp in milliseconds
  local wall_time=$((end_time - start_time)) # Calculate wall clock time

  # Extract key metrics from the raw output using grep and sed/tail
  local test_mse=$(echo "$raw" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
  local train_time=$(echo "$raw" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
  local trees=$(echo "$raw" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

  # Handle missing values by setting them to "N/A"
  [ -z "$test_mse" ] && test_mse="N/A"
  [ -z "$train_time" ] && train_time="N/A"
  [ -z "$trees" ] && trees="N/A"

  # Append results to the output file
  echo "${config_desc} | ${test_mse} | ${train_time} | ${wall_time} | ${trees}" >> "$OUTFILE"
}

echo "=== GBRT Single-Thread Comprehensive Test ==="
echo "Data File: $DATA_PATH"
echo "Results Saved To: $OUTFILE"
echo ""

# 1. Loss Function Test
echo "1. Loss Function Test..."
run_and_parse "Loss=squared,Baseline" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Loss=huber,Robust Loss" "$DATA_PATH" "huber" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Loss=absolute,L1 Loss" "$DATA_PATH" "absolute" 30 0.1 4 1 "mae" "exhaustive" 1.0

# 2. Iterations Test
echo "2. Iterations Test..."
run_and_parse "Iters=10,Fast Training" "$DATA_PATH" "squared" 10 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Iters=25,Standard Training" "$DATA_PATH" "squared" 25 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Iters=50,Full Training" "$DATA_PATH" "squared" 50 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Iters=100,Deep Training" "$DATA_PATH" "squared" 100 0.1 4 1 "mse" "exhaustive" 1.0

# 3. Learning Rate Test
echo "3. Learning Rate Test..."
run_and_parse "LR=0.01,Conservative Learning" "$DATA_PATH" "squared" 30 0.01 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.05,Steady Learning" "$DATA_PATH" "squared" 30 0.05 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.1,Standard Learning" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.2,Fast Learning" "$DATA_PATH" "squared" 30 0.2 4 1 "mse" "exhaustive" 1.0
run_and_parse "LR=0.5,Aggressive Learning" "$DATA_PATH" "squared" 30 0.5 4 1 "mse" "exhaustive" 1.0

# 4. Tree Depth Test
echo "4. Tree Depth Test..."
run_and_parse "Depth=2,Shallow Tree" "$DATA_PATH" "squared" 30 0.1 2 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=3,Balanced Tree" "$DATA_PATH" "squared" 30 0.1 3 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=4,Standard Tree" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=6,Deep Tree" "$DATA_PATH" "squared" 30 0.1 6 1 "mse" "exhaustive" 1.0
run_and_parse "Depth=8,Very Deep Tree" "$DATA_PATH" "squared" 30 0.1 8 1 "mse" "exhaustive" 1.0

# 5. Minimum Samples per Leaf Test
echo "5. Minimum Samples per Leaf Test..."
run_and_parse "MinLeaf=1,Fine Splits" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "MinLeaf=2,Standard Splits" "$DATA_PATH" "squared" 30 0.1 4 2 "mse" "exhaustive" 1.0
run_and_parse "MinLeaf=5,Robust Splits" "$DATA_PATH" "squared" 30 0.1 4 5 "mse" "exhaustive" 1.0
run_and_parse "MinLeaf=10,Conservative Splits" "$DATA_PATH" "squared" 30 0.1 4 10 "mse" "exhaustive" 1.0

# 6. Split Method Test
echo "6. Split Method Test..."
run_and_parse "Split=exhaustive,Exhaustive Search" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0
run_and_parse "Split=random,Random Search" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "random" 1.0
run_and_parse "Split=histogram_ew,Equal-Width Histogram" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "histogram_ew" 1.0
run_and_parse "Split=histogram_eq,Equal-Frequency Histogram" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "histogram_eq" 1.0

# 7. Subsampling Test
echo "7. Subsampling Test..."
run_and_parse "Subsample=0.5,50% Sampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.5
run_and_parse "Subsample=0.7,70% Sampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.7
run_and_parse "Subsample=0.8,80% Sampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.8
run_and_parse "Subsample=1.0,No Sampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0

# 8. Combined Optimization Test
echo "8. Combined Optimization Test..."
run_and_parse "Optimized1:Deep Training" "$DATA_PATH" "squared" 50 0.1 6 1 "mse" "exhaustive" 0.8
run_and_parse "Optimized2:Robust Training" "$DATA_PATH" "huber" 50 0.05 4 2 "mse" "exhaustive" 0.9
run_and_parse "FastCombination:Prototype" "$DATA_PATH" "squared" 20 0.2 3 1 "mse" "random" 1.0
run_and_parse "BalancedCombination:Production" "$DATA_PATH" "squared" 30 0.15 4 2 "mse" "histogram_ew" 0.85

echo ""
echo "=== GBRT Single-Thread Test Complete ==="
echo "Detailed results saved in: $OUTFILE"