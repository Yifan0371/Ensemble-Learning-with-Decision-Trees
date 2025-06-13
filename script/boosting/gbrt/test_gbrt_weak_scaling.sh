、#!/bin/bash
set -euo pipefail

# ==============================================================================
# script/boosting/gbrt/test_gbrt_weak_scaling.sh
# GBRT Weak-scaling test (automatically subsets data from a large file)
# Scales problem size linearly with thread count, testing time and MSE.
# ==============================================================================

# 1) Locate project root directory and executable
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/script/boosting/gbrt"
mkdir -p "$SCRIPT_DIR" # Create script directory if it doesn't exist

# Define and create log file (distinguish each run with a timestamp)
LOGFILE="$SCRIPT_DIR/gbrt_weak_scaling_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOGFILE"
echo "===============================================" >> "$LOGFILE"
echo "GBRT Weak Scaling Performance Test Log" >> "$LOGFILE"
echo "Run Time: $(date)" >> "$LOGFILE"
echo "Project Root: $PROJECT_ROOT" >> "$LOGFILE"
echo "===============================================" >> "$LOGFILE"

# Redirect all subsequent output to both terminal and log file
exec > >(tee -a "$LOGFILE") 2>&1

echo "PROJECT_ROOT=$PROJECT_ROOT"
EXECUTABLE="$PROJECT_ROOT/build/RegressionBoostingMain"

# 2) Source env_config.sh: automatically sets OMP_NUM_THREADS and compiles if necessary
source "$PROJECT_ROOT/script/env_config.sh"
MAX_CORES=$OMP_NUM_THREADS

# 3) Fixed original data file: cleaned_data.csv
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

# 5) Calculate total rows in original data (excluding header) and base rows per core
total_rows=$(( $(wc -l < "$DATA") - 1 ))
if (( total_rows < MAX_CORES )); then
    echo "WARNING: Data rows ($total_rows) are less than physical cores ($MAX_CORES), exiting script."
    exit 1
fi
BASE=$(( total_rows / MAX_CORES )) # Base rows per physical core
echo "Total rows (excluding header): $total_rows, Physical cores: $MAX_CORES, Base rows BASE=$BASE"

# 6) Generate thread list: 1, 2, 4, ..., MAX_CORES. Add MAX_CORES if not already included.
threads=(1)
while (( threads[-1] * 2 <= MAX_CORES )); do
    threads+=( $(( threads[-1] * 2 )) )
done
(( threads[-1] != MAX_CORES )) && threads+=( $MAX_CORES )

# 7) Print results table header
echo "==============================================="
echo "    GBRT Weak Scaling Performance Test     "
echo "==============================================="
echo "Fixed Parameters (per thread):"
echo "  Loss: $LOSS | Iterations: $NUM_ITERATIONS"
echo "  Learning Rate: $LEARNING_RATE | Max Depth: $MAX_DEPTH"
echo "  Min Leaf: $MIN_LEAF | Criterion: $CRITERION"
echo "  Split Method: $SPLIT_METHOD | Subsample: $SUBSAMPLE"
echo "  Base rows per thread: $BASE"
echo ""
echo "Threads | SubsetRows | Elapsed(ms) | TestMSE    | TrainTime  | Efficiency"
echo "--------|------------|-------------|------------|------------|----------"

# Record single-thread time as baseline for efficiency calculation
baseline_time=0

# 8) For each thread count 't':
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t # Set OpenMP threads for current run

    # 8.1) Calculate number of sample rows to subset: chunk_size = t * BASE
    chunk_size=$(( t * BASE ))
    # If chunk_size exceeds total_rows, take all available rows
    if (( chunk_size > total_rows )); then
        chunk_size=$total_rows
    fi
    # Number of lines to take for 'head' command = 1 header row + chunk_size sample rows
    lines_to_take=$(( chunk_size + 1 ))

    # 8.2) Generate a temporary file with header + the first 'lines_to_take - 1' samples
    tmpfile="$PROJECT_ROOT/data/data_clean/tmp_gbrt_chunk_t${t}.csv"
    head -n "$lines_to_take" "$DATA" > "$tmpfile"

    # 8.3) Start timing
    start_ts=$(date +%s%3N)

    # 8.4) Run GBRT and capture full stdout
    output=$("$EXECUTABLE" "$tmpfile" \
        "$LOSS" $NUM_ITERATIONS $LEARNING_RATE $MAX_DEPTH $MIN_LEAF \
        "$CRITERION" "$SPLIT_METHOD" $SUBSAMPLE 2>/dev/null) # Redirect stderr to null

    # 8.5) End timing
    end_ts=$(date +%s%3N)
    elapsed=$(( end_ts - start_ts ))

    # 8.6) Extract key metrics
    test_mse=$(echo "$output" | grep "Test MSE:" | sed -n 's/.*Test MSE: \([0-9.-]*\).*/\1/p' | tail -1)
    train_time=$(echo "$output" | grep "Train Time:" | sed -n 's/.*Train Time: \([0-9]*\)ms.*/\1/p' | tail -1)

    # Handle empty values
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$train_time" ]] && train_time="0" # Default to 0 if not found for calculation

    # 8.7) Calculate efficiency (relative to ideal weak scaling)
    if (( t == 1 )); then
        baseline_time=$elapsed # Set baseline for single-thread run
        efficiency="1.00"
    else
        if (( elapsed > 0 )); then
            efficiency=$(echo "scale=2; $baseline_time / $elapsed" | bc -l 2>/dev/null || echo "N/A")
        else
            efficiency="N/A"
        fi
    fi

    # 8.8) Print and log current result row
    printf "%7d | %10d | %11d | %-10s | %-10s | %s\n" \
           "$t" "$chunk_size" "$elapsed" "$test_mse" "${train_time}ms" "$efficiency"

    # 8.9) Delete temporary file
    rm -f "$tmpfile"
done

echo ""
echo "==============================================="
echo "Weak Scaling Analysis:"
echo "- Ideal: Time remains constant as threads and data increase."
echo "- Efficiency = Single-thread time / Current time."
echo "- Focus point: Efficiency close to 1.0 indicates good weak scaling."
echo "- Data size increases linearly with thread count, and computational complexity also increases linearly."
echo "- GBRT Characteristics: Sequential tree building limits weak scaling effectiveness."
echo "==============================================="

exit 0