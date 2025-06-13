#!/bin/bash
# =============================================================================
# script/boosting/gbrt_dart/test_gbrt_dart_single.sh
# GBRT DART (with DART) Single-Thread Comprehensive Test Script
# Run from project root: bash script/boosting/gbrt_dart/test_gbrt_dart_single.sh
# =============================================================================

export OMP_NUM_THREADS=1 # Ensure single-thread operation

PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt_dart"
OUTFILE="$RESULTS_DIR/gbrt_dart_single_results.txt"
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

echo "=== GBRT DART Single-Thread Comprehensive Test ==="
echo "Data File: $DATA_PATH"
echo "Results Saved To: $OUTFILE"
echo ""

---

### 1. DART vs. Standard GBRT Baseline Comparison

`echo` "1. DART vs. Standard GBRT Baseline Comparison..."
run_and_parse "Standard GBRT, Baseline" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "false" 0.0 "false" "false"
run_and_parse "DART 0% dropout, Consistency" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.0 "false" "false"

---

### 2. DART Dropout Rate Sensitivity Test

`echo` "2. DART Dropout Rate Test..."
run_and_parse "DART 5% dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.05 "false" "false"
run_and_parse "DART 10% dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.10 "false" "false"
run_and_parse "DART 15% dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.15 "false" "false"
run_and_parse "DART 20% dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART 25% dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.25 "false" "false"
run_and_parse "DART 30% dropout" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.30 "false" "false"

---

### 3. DART Weight Normalization Test

`echo` "3. DART Weight Normalization Test..."
run_and_parse "DART norm=true, Normalization Enabled" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "true" "false"
run_and_parse "DART norm=false, Normalization Disabled" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

---

### 4. DART Prediction Mode Test

`echo` "4. DART Prediction Mode Test..."
run_and_parse "DART skip=false, Training Mode" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART skip=true, Prediction Mode" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "true"

---

### 5. DART with Different Loss Functions

`echo` "5. DART with Loss Function Combinations..."
run_and_parse "DART+Squared Loss" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+Huber Loss" "$DATA_PATH" "huber" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+Absolute Loss" "$DATA_PATH" "absolute" 30 0.1 4 1 "mae" "exhaustive" 1.0 "true" 0.20 "false" "false"

---

### 6. DART with Different Tree Depths

`echo` "6. DART with Tree Depth Combinations..."
run_and_parse "DART depth=2, Shallow Tree" "$DATA_PATH" "squared" 30 0.1 2 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART depth=3, Balanced Tree" "$DATA_PATH" "squared" 30 0.1 3 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART depth=4, Standard Tree" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART depth=6, Deep Tree" "$DATA_PATH" "squared" 30 0.1 6 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

---

### 7. DART with Different Learning Rates

`echo` "7. DART with Learning Rate Combinations..."
run_and_parse "DART lr=0.05, Conservative Learning" "$DATA_PATH" "squared" 40 0.05 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART lr=0.1, Standard Learning" "$DATA_PATH" "squared" 40 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART lr=0.2, Fast Learning" "$DATA_PATH" "squared" 40 0.2 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART lr=0.3, Aggressive Learning" "$DATA_PATH" "squared" 40 0.3 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

---

### 8. DART with Split Method Combinations

`echo` "8. DART with Split Method Combinations..."
run_and_parse "DART+Exhaustive Split" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+Random Split" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "random" 1.0 "true" 0.20 "false" "false"
run_and_parse "DART+Equal-Width Histogram" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "histogram_ew" 1.0 "true" 0.20 "false" "false"

---

### 9. DART with Subsampling Combinations

`echo` "9. DART with Subsampling Combinations..."
run_and_parse "DART+50% Subsampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.5 "true" 0.20 "false" "false"
run_and_parse "DART+70% Subsampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.7 "true" 0.20 "false" "false"
run_and_parse "DART+90% Subsampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 0.9 "true" 0.20 "false" "false"
run_and_parse "DART+No Subsampling" "$DATA_PATH" "squared" 30 0.1 4 1 "mse" "exhaustive" 1.0 "true" 0.20 "false" "false"

---

### 10. DART Optimal Configuration Exploration

`echo` "10. DART Optimal Configuration Exploration..."
run_and_parse "Conservative DART Config" "$DATA_PATH" "squared" 40 0.08 4 2 "mse" "exhaustive" 0.9 "true" 0.15 "false" "false"
run_and_parse "Aggressive DART Config" "$DATA_PATH" "squared" 35 0.15 6 1 "mse" "exhaustive" 0.8 "true" 0.35 "false" "false"
run_and_parse "Balanced DART Config" "$DATA_PATH" "squared" 35 0.1 4 1 "mse" "exhaustive" 0.85 "true" 0.25 "false" "false"
run_and_parse "Robust DART Config" "$DATA_PATH" "huber" 35 0.1 4 3 "mse" "exhaustive" 0.9 "true" 0.20 "false" "false"

echo ""
echo "=== GBRT DART Single-Thread Test Complete ==="
echo "Detailed results saved in: $OUTFILE"
echo ""
echo "DART Test Key Points:"
echo "1. DART with 0% dropout should be equivalent to standard GBRT."
echo "2. Appropriate dropout rates (10%-30%) usually yield the best performance."
echo "3. The impact of weight normalization varies by data."
echo "4. The choice of prediction mode affects final performance."