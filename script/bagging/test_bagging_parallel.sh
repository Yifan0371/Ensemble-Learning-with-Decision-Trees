#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_parallel.sh
# Comprehensive Bagging Parallel Performance Test Script
# =============================================================================

# 1) Project root path & set thread count
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh" # Sources OMP_NUM_THREADS

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: Executable not found at $EXECUTABLE. Please compile first."
  exit 1
fi

# 2) Data path & results file
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: Data file $DATA_PATH does not exist."
  exit 1
fi

RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR" # Create results directory if it doesn't exist
RESULTS_FILE="$RESULTS_DIR/parallel_performance_results.txt"
> "$RESULTS_FILE" # Clear/create the results file

echo "=========================================="
echo "    Bagging Parallel Performance Test     "
echo "=========================================="
echo "Physical Cores: $OMP_NUM_THREADS"
echo "Data File: $(basename "$DATA_PATH")"
echo "Results Saved To: $RESULTS_FILE"
echo "Date: $(date)"
echo ""

# 3) Generate thread list (powers of 2 up to OMP_NUM_THREADS, plus OMP_NUM_THREADS itself)
threads_list=(1)
cur=1
while (( cur*2 <= OMP_NUM_THREADS )); do
  cur=$((cur*2))
  threads_list+=( $cur )
done
if (( cur != OMP_NUM_THREADS )); then
  threads_list+=( $OMP_NUM_THREADS )
fi
echo "Testing Thread Sequence: ${threads_list[*]}"
echo ""

# 4) Results file header
{
  echo "# Bagging Parallel Performance Test Results"
  echo "# Date: $(date)"
  echo "# Max Cores: $OMP_NUM_THREADS"
  echo "# Data: $(basename "$DATA_PATH")"
  echo "# Format: Config,Threads,TestMSE,TestMAE,OOB_MSE,TrainTime(ms),TotalTime(ms),Speedup,Efficiency"
} >> "$RESULTS_FILE"

declare -A baseline_times # Associative array to store baseline (single-thread) training times

# Function to extract results from the executable's output
extract_results() {
  local out="$1"
  local test_mse test_mae oob_mse train_time total_time
  # Use `tail -1` to ensure we get the last occurrence if the executable prints multiple times
  test_mse=$(echo "$out" | grep -E "Test MSE:" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  test_mae=$(echo "$out" | grep -E "Test MAE:" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
  oob_mse=$(echo "$out" | grep -E "OOB MSE:" | sed -n 's/.*OOB MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
  train_time=$(echo "$out" | grep -E "Train Time:" | sed -n 's/.*Train Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
  total_time=$(echo "$out" | grep -E "Total Time:" | sed -n 's/.*Total Time: *\([0-9]*\)ms.*/\1/p' | tail -1)

  # Assign "ERROR" or "0" if values are not found
  [[ -z "$test_mse" ]] && test_mse="ERROR"
  [[ -z "$test_mae" ]] && test_mae="ERROR"
  [[ -z "$oob_mse" ]] && oob_mse="ERROR"
  [[ -z "$train_time" ]] && train_time="0"
  [[ -z "$total_time" ]] && total_time="0"

  echo "$test_mse,$test_mae,$oob_mse,$train_time,$total_time"
}

# Function to run a single test configuration
run_test() {
  local config_name="$1" # Name of the configuration (e.g., "Standard")
  local threads="$2"     # Number of threads for this run
  shift 2
  local params=( "$@" )  # Remaining parameters for the executable

  export OMP_NUM_THREADS=$threads # Set OMP_NUM_THREADS for this test
  echo -n "  Testing $config_name ($threads threads)... "

  local start_ts=$(date +%s%3N) # Start timestamp in milliseconds
  local out # Variable to capture executable output
  out=$("$EXECUTABLE" bagging "$DATA_PATH" "${params[@]}" 2>&1) # Run the executable
  local exit_code=$? # Capture exit code
  local end_ts=$(date +%s%3N)   # End timestamp in milliseconds
  local wall_time=$(( end_ts - start_ts )) # Wall clock time

  if (( exit_code != 0 )); then
    echo "FAILED"
    echo "$config_name,$threads,ERROR,ERROR,ERROR,0,$wall_time,ERROR,ERROR" >> "$RESULTS_FILE"
    return
  fi

  local results # Variable to store extracted results
  results=$(extract_results "$out")
  local test_mse test_mae oob_mse train_time total_time
  IFS=',' read -r test_mse test_mae oob_mse train_time total_time <<< "$results" # Parse results string
  echo "Done (TrainTime=${train_time}ms)"

  local speedup="N/A"
  local efficiency="N/A"
  if (( threads == 1 )); then
    baseline_times["$config_name"]="$train_time" # Store baseline time for single-thread run
    speedup="1.00"
    efficiency="1.00"
  else
    local base="${baseline_times[$config_name]:-0}" # Get baseline time, default to 0 if not found
    if (( base > 0 && train_time > 0 )); then
      speedup=$(echo "scale=2; $base / $train_time" | bc -l) # Speedup = BaselineTime / CurrentTrainTime
      efficiency=$(echo "scale=2; $speedup / $threads" | bc -l) # Efficiency = Speedup / Threads
    fi
  fi

  echo "$config_name,$threads,$test_mse,$test_mae,$oob_mse,$train_time,$wall_time,$speedup,$efficiency" \
    >> "$RESULTS_FILE" # Append results to file
}

# 5) Define test configurations
declare -A test_configs # Associative array to hold parameters for each config
test_configs["FastSmall"]="10 1.0 8 2 mse exhaustive none 0.01 42"        # Quick test
test_configs["Standard"]="20 1.0 10 2 mse exhaustive none 0.01 42"        # Standard config
test_configs["DeepTrees"]="15 1.0 15 1 mse exhaustive none 0.01 42"       # Deeper trees, min_leaf=1
test_configs["RandomSplit"]="20 1.0 10 2 mse random none 0.01 42"         # Random split finder
test_configs["HistogramSplit"]="20 1.0 10 2 mse histogram_ew:64 none 0.01 42" # Histogram split finder

echo "Starting performance tests..."
echo ""

# Loop through each defined configuration and run tests for all thread counts
for config_name in "FastSmall" "Standard" "DeepTrees" "RandomSplit" "HistogramSplit"; do
  echo "=== Configuration: $config_name ==="
  IFS=' ' read -r -a params <<< "${test_configs[$config_name]}" # Read parameters into an array
  for threads in "${threads_list[@]}"; do
    run_test "$config_name" "$threads" "${params[@]}" # Call the run_test function
  done
  echo "" # Add a blank line between configurations for readability
done

echo "=========================================="
echo "Tests complete! Results saved to: $RESULTS_FILE"
echo "=========================================="