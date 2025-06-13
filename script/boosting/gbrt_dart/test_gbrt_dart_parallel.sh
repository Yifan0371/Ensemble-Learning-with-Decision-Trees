#!/bin/bash

# =============================================================================
# script/boosting/gbrt_dart/test_gbrt_parallel.sh
# GBRT DART Parallel Performance Comprehensive Test Script
# Tests performance across different core counts, including various DART configurations.
# =============================================================================

# Project root path and executable
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# Source env_config.sh to get physical core count and automatically build
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"
DATA_PATH="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/script/boosting/gbrt_dart"
mkdir -p "$RESULTS_DIR"

# Results file
RESULTS_FILE="$RESULTS_DIR/gbrt_dart_parallel_performance_results.txt"
> "$RESULTS_FILE"

echo "=========================================="
echo "   GBRT DART Parallel Performance Test   "
echo "=========================================="
echo "Physical Cores: $MAX_CORES"
echo "Data File: $(basename $DATA_PATH)"
echo "Results Saved To: $RESULTS_FILE"
echo "Time: $(date)"
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
# Add MAX_CORES if not already included
if (( current != MAX_CORES )); then
    threads_list+=($MAX_CORES)
fi

echo "Testing Thread Sequence: ${threads_list[*]}"
echo ""

# Write results file header
{
    echo "# GBRT DART Parallel Performance Test Results"
    echo "# Date: $(date)"
    echo "# Max Cores: $MAX_CORES"
    echo "# Data: $(basename $DATA_PATH)"
    echo "# Format: Config | Threads | TestMSE | TrainTime(ms) | TotalTime(ms) | Speedup | Efficiency"
} >> "$RESULTS_FILE"

# Function to extract results from output
extract_results() {
    local output="$1"
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
    local enable_dart="${11}"
    local dart_drop_rate="${12}"
    local dart_normalize="${13}"
    local dart_skip_drop="${14}"
    
    export OMP_NUM_THREADS=$threads
    
    echo -n "  Testing $config_name (${threads} threads)... "
    
    local start_time=$(date +%s%3N)
    local output
    output=$($EXECUTABLE "$DATA_PATH" \
        "$loss" $iterations $learning_rate $max_depth $min_leaf \
        "$criterion" "$split_method" $subsample \
        "$enable_dart" $dart_drop_rate "$dart_normalize" "$dart_skip_drop" 2>&1)
    local exit_code=$?
    local end_time=$(date +%s%3N)
    local wall_time=$((end_time - start_time))
    
    if [ $exit_code -ne 0 ]; then
        echo "FAILED"
        echo "$config_name,$threads,ERROR,0,$wall_time,ERROR,ERROR" >> "$RESULTS_FILE"
        return
    fi
    
    local results=$(extract_results "$output")
    local test_mse=$(echo "$results" | cut -d',' -f1)
    local train_time=$(echo "$results" | cut -d',' -f2)
    local trees=$(echo "$results" | cut -d',' -f3)
    
    echo "Done (${train_time}ms)"
    
    # Calculate speedup and efficiency
    local speedup="N/A"
    local efficiency="N/A"
    if [ "$threads" -eq 1 ]; then
        # Store baseline time
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
    
    # Write results
    echo "$config_name,$threads,$test_mse,$train_time,$wall_time,$speedup,$efficiency" >> "$RESULTS_FILE"
}

# Test configurations definition
declare -A test_configs
declare -A baseline_times

# Config format: loss iterations lr depth minleaf criterion split subsample enable_dart drop_rate normalize skip_drop
test_configs["DART_Standard"]="squared 30 0.1 4 1 mse exhaustive 1.0 true 0.15 false false"
test_configs["DART_Conservative"]="squared 30 0.1 4 2 mse exhaustive 0.9 true 0.10 false false"
test_configs["DART_Aggressive"]="squared 30 0.15 6 1 mse exhaustive 0.8 true 0.25 false false"
test_configs["DART_Normalized"]="squared 30 0.1 4 1 mse exhaustive 1.0 true 0.20 true false"
test_configs["DART_SkipDrop"]="squared 30 0.1 4 1 mse exhaustive 1.0 true 0.20 false true"
test_configs["DART_Random"]="squared 30 0.1 4 1 mse random 1.0 true 0.15 false false"
test_configs["DART_Huber"]="huber 30 0.1 4 1 mse exhaustive 1.0 true 0.15 false false"
test_configs["Standard_GBRT"]="squared 30 0.1 4 1 mse exhaustive 1.0 false 0.0 false false"

echo "Starting performance tests..."
echo ""

# Main test loop
for config_name in "Standard_GBRT" "DART_Standard" "DART_Conservative" "DART_Aggressive" "DART_Normalized" "DART_SkipDrop" "DART_Random" "DART_Huber"; do
    echo "=== Configuration: $config_name ==="
    
    # Parse configuration parameters
    IFS=' ' read -ra params <<< "${test_configs[$config_name]}"
    
    for threads in "${threads_list[@]}"; do
        run_test "$config_name" "$threads" "${params[@]}"
    done
    
    echo ""
done

echo "=========================================="
echo "Tests complete! Generating results summary..."
echo ""

# Generate summary report
{
    echo ""
    echo "===== PERFORMANCE SUMMARY ====="
    echo ""
    
    # Group results by configuration
    for config_name in "Standard_GBRT" "DART_Standard" "DART_Conservative" "DART_Aggressive" "DART_Normalized" "DART_SkipDrop" "DART_Random" "DART_Huber"; do
        echo "=== $config_name ==="
        echo "Threads | TestMSE    | TrainTime  | Speedup | Efficiency"
        echo "--------|------------|------------|---------|----------"
        
        grep "^$config_name," "$RESULTS_FILE" | while IFS=',' read -r cfg threads mse train_time wall_time speedup efficiency; do
            printf "%-7s | %-10s | %-10s | %-7s | %s\n" "$threads" "$mse" "${train_time}ms" "$speedup" "$efficiency"
        done
        echo ""
    done
    
    echo "===== DART PARALLEL SCALING ANALYSIS ====="
    echo ""
    echo "DART vs. Standard GBRT Parallel Performance Comparison:"
    echo "- Standard GBRT: Limited tree building parallelism, expected efficiency 0.7-0.9."
    echo "- DART: Additional dropout and weight update overhead."
    echo "- Conservative Configuration: Lower dropout rate, better parallelism."
    echo "- Aggressive Configuration: Higher dropout rate, may reduce parallel efficiency."
    echo "- Weight Normalization: May introduce synchronization points, affecting parallelism."
    echo ""
    
    echo "Optimization Suggestions:"
    echo "- For multi-core systems, conservative or standard DART configurations are recommended."
    echo "- Avoid excessively high dropout rates (>30%) in parallel environments."
    echo "- The parallel overhead of weight normalization needs to be evaluated."
    echo "- Consider using random split to improve parallelism."
    echo ""
    
} >> "$RESULTS_FILE"

echo "Detailed results saved to: $RESULTS_FILE"
echo ""

# Display brief summary
echo "=== Brief Performance Summary ==="
printf "%-20s | %-7s | %-10s | %-7s | %s\n" "Config" "Threads" "Train Time" "Speedup" "Efficiency"
echo "--------------------|---------|------------|---------|----------"

for config_name in "Standard_GBRT" "DART_Standard" "DART_Aggressive"; do
    # Display comparison for 1 core and max cores
    baseline_result=$(grep "^$config_name,1," "$RESULTS_FILE" | cut -d',' -f4)
    maxcore_line=$(grep "^$config_name,$MAX_CORES," "$RESULTS_FILE")
    
    if [ -n "$maxcore_line" ]; then
        maxcore_time=$(echo "$maxcore_line" | cut -d',' -f4)
        maxcore_speedup=$(echo "$maxcore_line" | cut -d',' -f6)
        maxcore_efficiency=$(echo "$maxcore_line" | cut -d',' -f7)
        
        printf "%-20s | %-7s | %-10s | %-7s | %s\n" \
               "$config_name" "1" "${baseline_result}ms" "1.00" "1.00"
        printf "%-20s | %-7s | %-10s | %-7s | %s\n" \
               "" "$MAX_CORES" "${maxcore_time}ms" "$maxcore_speedup" "$maxcore_efficiency"
        echo "--------------------|---------|------------|---------|----------"
    fi
done

echo ""
echo "GBRT DART parallel tests complete!"
echo "Detailed results: $RESULTS_FILE"
echo "It's recommended to review the full report to analyze DART's performance characteristics in a parallel environment."