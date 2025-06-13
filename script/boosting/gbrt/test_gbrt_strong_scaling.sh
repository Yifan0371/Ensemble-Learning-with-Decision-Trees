、#!/bin/bash
set -euo pipefail

# ==============================================================================
# script/boosting/gbrt/test_gbrt_strong_scaling.sh
# GBRT Strong-scaling test: Fixed dataset size, measures execution time and MSE
# with varying thread counts.
# ==============================================================================

# 1) Locate project root directory & log directory
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/boosting/gbrt"
mkdir -p "$SCRIPT_DIR"

# Define and create log file (distinguish each run with a timestamp)
LOGFILE="$SCRIPT_DIR/gbrt_strong_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "GBRT Strong Scaling Performance Test Log" >> "$LOGFILE"
echo "Run Time: $(date)" >> "$LOGFILE"
echo "Project Root: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# Redirect all subsequent output to both terminal and log file
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

# 2) Source env_config.sh to automatically set OMP_NUM_THREADS and compile if necessary
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) Fixed parameters
DATA="$PROJECT_ROOT/data/data_clean/cleaned_data.csv"
LOSS="squared"
NUM_ITERATIONS=30
LEARNING_RATE=0.1
MAX_DEPTH=4
MIN_LEAF=1
CRITERION="mse"
SPLIT_METHOD="exhaustive"
SUBSAMPLE=1.0

# 4) Confirm executable and data file existence
[[ -f "$EXECUTABLE" ]] || { echo "ERROR: $EXECUTABLE does not exist."; exit 1; }
[[ -f "$DATA"       ]] || { echo "ERROR: $DATA does not exist."; exit 1; }

# 5) Generate thread list: 1, 2, 4, ..., MAX_CORES, and add MAX_CORES if not already included
threads=(1)
while (( threads[-1]*2 <= MAX_CORES )); do
    threads+=( $(( threads[-1]*2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 6) Print table header
echo "==============================================="
echo "    GBRT Strong Scaling Performance Test    "
echo "==============================================="
echo "Fixed Parameters:"
echo "  Loss: $LOSS | Iterations: $NUM_ITERATIONS"
echo "  Learning Rate: $LEARNING_RATE | Max Depth: $MAX_DEPTH"
echo "  Min Leaf: $MIN_LEAF | Criterion: $CRITERION"
echo "  Split Method: $SPLIT_METHOD | Subsample: $SUBSAMPLE"
echo "  Data: $(basename "$DATA")"
echo ""
echo "Threads | Elapsed(ms) | TestMSE    | TrainTime  | Trees/sec"
echo "--------|-------------|------------|------------|----------"

for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t

    # 6.1) Start timing
    start_ts=$(date +%s%3N)

    # 6.2) Execute GBRT and capture full stdout
    output=$("$EXECUTABLE" "$DATA" \
        "$LOSS" $NUM_ITERATIONS $LEARNING_RATE $MAX_DEPTH $MIN_LEAF \
        "$CRITERION" "$SPLIT_METHOD" $SUBSAMPLE 2>/dev/null) # Redirect stderr to null

    # 6.3) End timing
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 6.4) Extract key metrics
    test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)
    trees=$(echo "$output" | grep "Trees:" | sed -n 's/.*Trees: \([0-9]*\).*/\1/p' | tail -1)

    # Calculate throughput: Trees/sec
    if [[ -n "$trees" && "$elapsed" -gt 0 ]]; then
        trees_per_sec=$(echo "scale=2; $trees * 1000 / $elapsed" | bc -l 2>/dev/null || echo "N/A")
    else
        trees_per_sec="N/A"
    fi

    # Handle empty values
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$train_time" ]] && train_time="0" # Default to 0 if not found for calculation
    [[ -z "$trees" ]] && trees="0"           # Default to 0 if not found for calculation

    # 6.5) Print and log current result row
    printf "%7d | %11d | %-10s | %-10s | %s\n" \
           "$t" "$elapsed" "$test_mse" "${train_time}ms" "$trees_per_sec"
done

echo ""
echo "==============================================="
echo "Strong Scaling Analysis:"
echo "- Ideal: Linear speedup, time inversely proportional to thread count."
echo "- Focus points: TestMSE should remain stable, Elapsed time should decrease."
echo "- Efficiency = (Serial Time / Parallel Time) / Number of Threads."
echo "- GBRT Characteristics: Tree building has data dependencies, limiting parallel efficiency."
echo "==============================================="

exit 0