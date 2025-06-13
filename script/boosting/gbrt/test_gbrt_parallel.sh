#!/bin/bash

# =============================================================================
# script/boosting/gbrt/test_gbrt_parallel.sh
# GBRT Parallel Performance Comprehensive Test Script
# Tests performance across different core counts, including various configurations.
# =============================================================================

# Project root path and executable
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# Source env_config.sh to get physical core count and automatically build (if configured)
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt"
mkdir -p "$RESULTS_DIR"

# Results file
RESULTS_FILE="$RESULTS_DIR/gbrt_parallel_performance_results.txt"
> "$RESULTS_FILE" # Clears the file if it exists, or creates it

echo "=========================================="
echo "    GBRT Parallel Performance Test     "
echo "=========================================="
echo "Physical Cores: $MAX_CORES"
echo "Data File: $(basename $DATA_PATH)"
echo "Results Saved To: $RESULTS_FILE"
echo "Date: $(date)"
echo ""

# Check for file existence
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: $EXECUTABLE not found!"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file $DATA_PATH not found!"
    # Try alternative data files
    for alt_data in "$PROJECT_ROOT/data/data_clean/cleaned_sample_400_rows.csv" "$PROJECT_ROOT/data/data_clean/cleaned_15k_random.csv"; do
        if [ -f "$alt_data" ]; then
            echo "Using alternative data file: $(basename $alt_data)"
            DATA_PATH="$alt_data"
            break
        fi
    done
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "No suitable data file found."
        exit 1
    fi
fi

# Generate thread count list
threads_list=(1)
current=1
while (( current * 2 <= MAX_CORES )); do
    current=$((current * 2))
    threads_list+=($current)
done
# If the last generated value is not MAX_CORES, add MAX_CORES
if (( current != MAX_CORES )); then
    threads_list+=($MAX_CORES)
fi

echo "Testing Thread Sequence: ${threads_list[*]}"
echo ""

# Write results file header
{
    echo "# GBRT Parallel Performance Test Results"
    echo "# Date: $(date)"
    echo "# Max Cores: $MAX_CORES"
    echo "# Data: $(basename $DATA_PATH)"
    echo "# Format: Config | Threads | TestMSE | TrainTime(ms) | TotalTime(ms) | Speedup | Efficiency"
} >> "$RESULTS_FILE"

# Function to extract results from output
extract_results() {
    local output="$1"
    # Extract Test MSE, Train Time, and Trees from output
    local test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    local train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    local trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)
    
    # Handle empty values
    [ -z "$test_mse" ] && test_mse="ERROR"
    [ -z "$train_time" ] && train_time="0"
    [ -z "$trees" ] && trees="0"
    
    echo "$test_mse,$train_time,$trees"
}

# Function to run a single test
run_test() {
    local config_name="$1"
    local threads="$2"
    local loss="$3"
    local iterations="$4" 
    local learning_rate="$5"
    local max_depth="$6"
    local min_leaf="$7"
    local criterion="$8"
    local split_method="$9"
    local subsample="${10}"
    
    export OMP_NUM_THREADS=$threads # Set OpenMP threads for this run
    
    echo -n "  Testing $config_name (${threads} threads)... "
    
    local start_time=$(date +%s%3N) # Start timestamp in milliseconds
    local output # Capture executable output
    output=$($EXECUTABLE "$DATA_PATH" \
        "$loss" $iterations $learning_rate $max_depth $min_leaf \
        "$criterion" "$split_method" $subsample 2>&1) # Execute the GBRT program
    local exit_code=$? # Capture exit code
    local end_time=$(date +%s%3N)   # End timestamp in milliseconds
    local wall_time=$((end_time - start_time)) # Wall clock time
    
    if [ $exit_code -ne 0 ]; then
        echo "FAILED"
        echo "$config_name,$threads,ERROR,0,$wall_time,ERROR,ERROR" >> "$RESULTS_FILE"
        return
    fi
    
    local results=$(extract_results "$output")
    local test_mse=$(echo "$results" | cut -d',' -f1)
    local train_time=$(echo "$results" | cut -d',' -f2)
    local trees=$(echo "$results" | cut -d',' -f3) # Extracted trees count
    
    echo "Done (${train_time}ms)"
    
    # Calculate speedup and efficiency
    local speedup="N/A"
    local efficiency="N/A"
    if [ "$threads" -eq 1 ]; then
        # Store baseline time for single-thread run
        baseline_times["$config_name"]="$train_time"
        speedup="1.00"
        efficiency="1.00"
    else
        # Get baseline time
        local baseline_time="${baseline_times[$config_name]:-0}"
        if [ "$baseline_time" -gt 0 ] && [ "$train_time" -gt 0 ]; then
            speedup=$(echo "scale=2; $baseline_time / $train_time" | bc -l 2>/dev/null || echo "N/A")
            efficiency=$(echo "scale=2; $speedup / $threads" | bc -l 2>/dev/null || echo "N/A")
        fi
    fi
    
    # Write results to the file
    echo "$config_name,$threads,$test_mse,$train_time,$wall_time,$speedup,$efficiency" >> "$RESULTS_FILE"
}

# Test configurations definition
declare -A test_configs
declare -A baseline_times # Already declared globally above, but good practice to show intent

test_configs["FastSmall"]="squared 20 0.1 3 2 mse exhaustive 1.0"
test_configs["Standard"]="squared 30 0.1 4 1 mse exhaustive 1.0" 
test_configs["DeepTrees"]="squared 25 0.1 6 1 mse exhaustive 1.0"
test_configs["RandomSplit"]="squared 30 0.1 4 1 mse random 1.0"
test_configs["HistogramSplit"]="squared 30 0.1 4 1 mse histogram_ew 1.0"
test_configs["HuberLoss"]="huber 30 0.1 4 1 mse exhaustive 1.0"
test_configs["Subsample"]="squared 30 0.1 4 1 mse exhaustive 0.8"

echo "Starting performance tests..."
echo ""

# Main test loop
for config_name in "FastSmall" "Standard" "DeepTrees" "RandomSplit" "HistogramSplit" "HuberLoss" "Subsample"; do
    echo "=== Configuration: $config_name ==="
    
    # Parse configuration parameters into an array
    IFS=' ' read -ra params <<< "${test_configs[$config_name]}"
    
    # Run test for each thread count in the list
    for threads in "${threads_list[@]}"; do
        run_test "$config_name" "$threads" "${params[@]}"
    done
    
    echo "" # Newline for readability between configs
done

echo "=========================================="
echo "Tests complete! Generating results summary..."
echo ""

# Generate summary report within the results file
{
    echo ""
    echo "===== PERFORMANCE SUMMARY ====="
    echo ""
    
    # Group results by configuration
    for config_name in "FastSmall" "Standard" "DeepTrees" "RandomSplit" "HistogramSplit" "HuberLoss" "Subsample"; do
        echo "=== $config_name ==="
        echo "Threads | TestMSE    | TrainTime  | Speedup | Efficiency"
        echo "--------|------------|------------|---------|----------"
        
        # Read and format results for the current configuration
        grep "^$config_name," "$RESULTS_FILE" | while IFS=',' read -r cfg threads mse train_time wall_time speedup efficiency; do
            printf "%-7s | %-10s | %-10s | %-7s | %s\n" "$threads" "$mse" "${train_time}ms" "$speedup" "$efficiency"
        done
        echo ""
    done
    
    echo "===== PARALLEL SCALING ANALYSIS ====="
    echo ""
    echo "Ideal parallel performance metrics:"
    echo "- Speedup: Close to the number of threads indicates good strong scaling."
    echo "- Efficiency: Close to 1.0 indicates high thread utilization."
    echo "- Linear Scaling: Efficiency > 0.8 is generally considered good."
    echo ""
    
    echo "GBRT Parallel Characteristics Analysis:"
    echo "- Gradient Calculation: Highly parallel, efficiency close to 1.0."
    echo "- Tree Building: Moderately parallel, affected by split finding."
    echo "- Prediction Update: Highly parallel, efficiency close to 1.0."
    echo "- Overall: Expected efficiency 0.7-0.9, depending on data size and tree depth."
    echo ""
    
} >> "$RESULTS_FILE"

echo "Detailed results saved to: $RESULTS_FILE"
echo ""

# Display brief summary on console
echo "=== Brief Performance Summary ==="
printf "%-15s | %-7s | %-10s | %-7s | %s\n" "Config" "Threads" "Train Time" "Speedup" "Efficiency"
echo "----------------|---------|------------|---------|----------"

for config_name in "Standard" "DeepTrees" "RandomSplit"; do
    # Display comparison for 1 core and MAX_CORES
    baseline_result=$(grep "^$config_name,1," "$RESULTS_FILE" | cut -d',' -f4)
    maxcore_line=$(grep "^$config_name,$MAX_CORES," "$RESULTS_FILE")
    
    if [ -n "$maxcore_line" ]; then
        maxcore_time=$(echo "$maxcore_line" | cut -d',' -f4)
        maxcore_speedup=$(echo "$maxcore_line" | cut -d',' -f6)
        maxcore_efficiency=$(echo "$maxcore_line" | cut -d',' -f7)
        
        printf "%-15s | %-7s | %-10s | %-7s | %s\n" \
               "$config_name" "1" "${baseline_result}ms" "1.00" "1.00"
        printf "%-15s | %-7s | %-10s | %-7s | %s\n" \
               "" "$MAX_CORES" "${maxcore_time}ms" "$maxcore_speedup" "$maxcore_efficiency"
        echo "----------------|---------|------------|---------|----------"
    fi
done

echo ""
echo "GBRT parallel tests complete!"
echo "Detailed results: $RESULTS_FILE"
echo "Recommended to review the full report for analysis of parallel scalability across different configurations."