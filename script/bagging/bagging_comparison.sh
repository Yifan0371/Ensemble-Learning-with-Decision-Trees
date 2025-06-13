#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/test_bagging_performance_comparison.sh
#
# Compares the performance of Exhaustive vs. Random Split Finders in the Bagging module.
# =============================================================================

# 1) Project root path & set thread count
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh" # Sources OMP_NUM_THREADS

EXECUTABLE="$PROJECT_ROOT/build/DecisionTreeMain"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: Executable not found at $EXECUTABLE. Please compile first."
  exit 1
fi

# 2) Data path & output files
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: Data file not found at $DATA_PATH."
  exit 1
fi

RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"

EXHAUSTIVE_RESULTS="$RESULTS_DIR/exhaustive_results.txt"
RANDOM_RESULTS="$RESULTS_DIR/random_results.txt"

# Clear old results and write headers
{
  echo "# Exhaustive Finder Results - $(date)"
  echo "Trees,Train_Time(ms),Test_MSE,Test_MAE,OOB_MSE"
} > "$EXHAUSTIVE_RESULTS"

{
  echo "# Random Finder Results - $(date)"
  echo "Trees,Train_Time(ms),Test_MSE,Test_MAE,OOB_MSE"
} > "$RANDOM_RESULTS"

TREE_COUNTS=(5 10 20 50 100) # Number of trees to test

echo "=========================================="
echo "Bagging Performance Comparison"
echo "=========================================="
echo "Comparing: Exhaustive vs. Random Finders"
echo "Configuration: MSE criterion, No pruner"
echo "Date: $(date)"
echo ""
echo "Testing configurations: Trees = ${TREE_COUNTS[*]}"
echo "Data: $DATA_PATH"
echo "Timeout: 300 seconds per test"
echo ""

# Helper function to extract results from log file
extract_results() {
  local log_file="$1"
  local train_time test_mse test_mae oob_mse
  # Use awk to extract the numeric part after the specific string, and sed to remove "ms"
  train_time=$(grep -E "Train Time:" "$log_file" | awk '{print $3}' | sed 's/ms//g' | tail -1)
  test_mse=$(grep -E "Test MSE:" "$log_file" | awk '{print $3}' | tail -1)
  test_mae=$(grep -E "Test MAE:" "$log_file" | awk '{print $6}' | tail -1)
  oob_mse=$(grep -E "OOB MSE:" "$log_file" | awk '{print $3}' | tail -1)
  echo "$train_time,$test_mse,$test_mae,$oob_mse"
}

# Loop through each tree count
for trees in "${TREE_COUNTS[@]}"; do
  echo "Testing with $trees trees..."

  # --- Exhaustive Finder Test ---
  echo "  Running Exhaustive finder..."
  temp_ex="temp_exhaustive_${trees}.log"
  timeout 300 "$EXECUTABLE" bagging "$DATA_PATH" \
    "$trees" 1.0 30 2 mse exhaustive none 0.01 42 \
    > "$temp_ex" 2>&1

  # Check exit code of the timeout command
  if [[ $? -eq 0 ]]; then
    results=$(extract_results "$temp_ex")
    echo "$trees,$results" >> "$EXHAUSTIVE_RESULTS"
    echo "    Exhaustive completed: $results"
  else
    echo "    Exhaustive failed or timed out"
    echo "$trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$EXHAUSTIVE_RESULTS"
  fi
  rm -f "$temp_ex" # Clean up temporary log file

  # --- Random Finder Test ---
  echo "  Running Random finder..."
  temp_rnd="temp_random_${trees}.log"
  timeout 300 "$EXECUTABLE" bagging "$DATA_PATH" \
    "$trees" 1.0 30 2 mse random none 0.01 42 \
    > "$temp_rnd" 2>&1

  # Check exit code of the timeout command
  if [[ $? -eq 0 ]]; then
    results=$(extract_results "$temp_rnd")
    echo "$trees,$results" >> "$RANDOM_RESULTS"
    echo "    Random completed: $results"
  else
    echo "    Random failed or timed out"
    echo "$trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$RANDOM_RESULTS"
  fi
  rm -f "$temp_rnd" # Clean up temporary log file

  echo "  Completed $trees trees test"
  echo ""
done

echo "=========================================="
echo "Summary Report"
echo "=========================================="
echo ""
echo "Exhaustive Finder Results:"
echo "Trees | Train_Time | Test_MSE | Test_MAE | OOB_MSE"
echo "------|------------|----------|----------|--------"
# Skip header rows (first 2 lines) and format output
tail -n +3 "$EXHAUSTIVE_RESULTS" | while IFS=',' read -r t tt mse mae oob; do
  printf "%5s | %10s | %8s | %8s | %7s\n" "$t" "$tt" "$mse" "$mae" "$oob"
done

echo ""
echo "Random Finder Results:"
echo "Trees | Train_Time | Test_MSE | Test_MAE | OOB_MSE"
echo "------|------------|----------|----------|--------"
# Skip header rows (first 2 lines) and format output
tail -n +3 "$RANDOM_RESULTS" | while IFS=',' read -r t tt mse mae oob; do
  printf "%5s | %10s | %8s | %8s | %7s\n" "$t" "$tt" "$mse" "$mae" "$oob"
done

echo ""
echo "Results saved to:"
echo "  - $EXHAUSTIVE_RESULTS"
echo "  - $RANDOM_RESULTS"
echo ""
echo "Performance comparison completed."
echo "=========================================="