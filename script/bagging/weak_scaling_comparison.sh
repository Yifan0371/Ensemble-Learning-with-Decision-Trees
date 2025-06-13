#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/weak_scaling_comparison.sh
# 
# Weak Scaling Test: Compares MPI+OpenMP hybrid version vs. Pure OpenMP version.
# Maintains constant workload per processor while scaling problem size linearly
# with the number of processors.
# =============================================================================

# Project root path
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh"

# Ensure running in the build directory
BUILD_DIR="$PROJECT_ROOT/build"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: Build directory not found: $BUILD_DIR"
    exit 1
fi

echo "Changing to build directory: $BUILD_DIR"
cd "$BUILD_DIR"

# Executable paths (relative to build directory)
MPI_EXECUTABLE="./MPIBaggingMain"
OPENMP_EXECUTABLE="./DecisionTreeMain"

# Check executables
if [[ ! -x "$MPI_EXECUTABLE" ]]; then
    echo "ERROR: MPI executable not found: $MPI_EXECUTABLE"
    echo "Please build with: cmake -DENABLE_MPI=ON .. && make"
    exit 1
fi

if [[ ! -x "$OPENMP_EXECUTABLE" ]]; then
    echo "ERROR: OpenMP executable not found: $OPENMP_EXECUTABLE"
    echo "Please build the project first"
    exit 1
fi

# Data path - using relative path
DATA_PATH="../data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# Results directory (back to project root's script/bagging)
RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"

# Result files
MPI_RESULTS="$RESULTS_DIR/weak_scaling_mpi_results.csv"
OPENMP_RESULTS="$RESULTS_DIR/weak_scaling_openmp_results.csv"
COMPARISON_RESULTS="$RESULTS_DIR/weak_scaling_comparison.csv"

# Weak scaling base parameters
BASE_TREES_PER_CORE=25       # Base number of trees per core
FIXED_SAMPLE_RATIO=1.0       # Fixed sample ratio
FIXED_MAX_DEPTH=10           # Fixed maximum tree depth
FIXED_MIN_SAMPLES_LEAF=2     # Fixed minimum samples per leaf
FIXED_CRITERION="mse"        # Fixed split criterion
FIXED_SPLIT_METHOD="random"  # Fixed split method
FIXED_PRUNER_TYPE="none"     # Fixed pruner type
# Note: MPI version requires 8 parameters

# CPU architecture related configuration
PHYSICAL_CORES=36  # Example: 2 sockets * 18 cores/socket
MAX_CORES=$PHYSICAL_CORES

# Test configurations: number of cores (for weak scaling)
TEST_CONFIGS=(1 2 4 6 9 12 18 24 36)

echo "=========================================="
echo "     Weak Scaling Comparison Test        "
echo "=========================================="
echo "CPU Architecture: Intel Xeon E5-2699 v3"
echo "Physical Cores: $PHYSICAL_CORES"
echo "Test Date: $(date)"
echo ""
echo "Weak Scaling Strategy:"
echo "  Base workload: $BASE_TREES_PER_CORE trees per core"
echo "  Scaling: Trees = Cores × $BASE_TREES_PER_CORE"
echo ""
echo "Fixed Parameters:"
echo "  Sample Ratio: $FIXED_SAMPLE_RATIO"
echo "  Max Depth: $FIXED_MAX_DEPTH"
echo "  Min Samples Leaf: $FIXED_MIN_SAMPLES_LEAF"
echo "  Criterion: $FIXED_CRITERION"
echo "  Split Method: $FIXED_SPLIT_METHOD"
echo "  Data: $(basename "$DATA_PATH")"
echo ""

# Create result file headers
{
    echo "# Weak Scaling Test Results - MPI+OpenMP Version"
    echo "# Date: $(date)"
    echo "# Base workload: $BASE_TREES_PER_CORE trees per core"
    echo "# Format: Cores,MPI_Processes,OpenMP_Threads,Total_Trees,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Efficiency"
} > "$MPI_RESULTS"

{
    echo "# Weak Scaling Test Results - Pure OpenMP Version"
    echo "# Date: $(date)"
    echo "# Base workload: $BASE_TREES_PER_CORE trees per core"
    echo "# Format: Cores,OpenMP_Threads,Total_Trees,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Efficiency"
} > "$OPENMP_RESULTS"

# Helper function: Extract results from log file
extract_mpi_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    wall_time=$(grep -E "Total time.*including communication" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Max training time across processes" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

extract_openmp_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    wall_time=$(grep -E "Total Time:" "$log_file" | sed -n 's/.*Total Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Train Time:" "$log_file" | sed -n 's/.*Train Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

# Baseline times (single core performance)
baseline_mpi_time=""
baseline_openmp_time=""

echo "=========================================="
echo "Starting Weak Scaling Tests..."
echo "=========================================="

# Test different configurations
for cores in "${TEST_CONFIGS[@]}"; do
    # Calculate total trees for current configuration (weak scaling: workload grows linearly with cores)
    total_trees=$((cores * BASE_TREES_PER_CORE))
    
    echo ""
    echo "Testing configuration: $cores cores, $total_trees trees"
    echo "----------------------------------------"
    
    # === MPI+OpenMP Hybrid Test ===
    echo "  [1/2] Testing MPI+OpenMP version..."
    
    # Calculate MPI processes and OpenMP threads per process
    # Simplified strategy: consistent with strong scaling test
    if (( cores <= 4 )); then
        mpi_procs=$cores     # 1-4 cores: one process per core
        omp_threads=1
    elif (( cores <= 12 )); then
        mpi_procs=$(( (cores + 1) / 2 ))   # 5-12 cores: approx. one process per two cores
        omp_threads=2
    else
        mpi_procs=$(( (cores + 3) / 4 ))   # >12 cores: approx. one process per four cores
        omp_threads=4
    fi
    
    # Ensure at least 1 process, and not more than cores processes
    mpi_procs=$(( mpi_procs > 0 ? mpi_procs : 1 ))
    mpi_procs=$(( mpi_procs > cores ? cores : mpi_procs ))
    
    # Set OpenMP environment variables
    export OMP_NUM_THREADS=$omp_threads
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    
    mpi_log_file="temp_mpi_weak_${cores}_cores.log"
    
    # Run MPI version (simplified command, consistent with successful example)
    timeout 900 mpirun -np $mpi_procs \
        "$MPI_EXECUTABLE" \
        "$DATA_PATH" \
        $total_trees \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        > "$mpi_log_file" 2>&1
    
    if [[ $? -eq 0 ]]; then
        mpi_results=$(extract_mpi_results "$mpi_log_file")
        IFS=',' read -r mpi_wall_time mpi_train_time mpi_test_mse mpi_test_mae <<< "$mpi_results"
        
        # Calculate weak scaling efficiency (ideally, time should remain constant)
        if [[ -z "$baseline_mpi_time" && "$mpi_wall_time" != "ERROR" ]]; then
            baseline_mpi_time=$mpi_wall_time
            mpi_efficiency="1.00"
        elif [[ "$mpi_wall_time" != "ERROR" && -n "$baseline_mpi_time" ]]; then
            # Weak scaling efficiency = BaselineTime / CurrentTime
            mpi_efficiency=$(echo "scale=3; $baseline_mpi_time / $mpi_wall_time" | bc -l)
        else
            mpi_efficiency="ERROR"
        fi
        
        echo "$cores,$mpi_procs,$omp_threads,$total_trees,$mpi_wall_time,$mpi_train_time,$mpi_test_mse,$mpi_test_mae,$mpi_efficiency" >> "$MPI_RESULTS"
        echo "    MPI+OpenMP: ${mpi_procs}P×${omp_threads}T, Trees: $total_trees, Time: ${mpi_wall_time}ms, Efficiency: ${mpi_efficiency}"
    else
        echo "    MPI+OpenMP: FAILED or TIMEOUT"
        echo "$cores,$mpi_procs,$omp_threads,$total_trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR" >> "$MPI_RESULTS"
    fi
    
    rm -f "$mpi_log_file"
    
    # === Pure OpenMP Test ===
    echo "  [2/2] Testing Pure OpenMP version..."
    
    # Set OpenMP environment variables
    export OMP_NUM_THREADS=$cores
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    
    openmp_log_file="temp_openmp_weak_${cores}_cores.log"
    
    # Run OpenMP version (includes prunerParam and seed)
    timeout 900 "$OPENMP_EXECUTABLE" bagging \
        "$DATA_PATH" \
        $total_trees \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        0.01 \
        42 \
        > "$openmp_log_file" 2>&1
    
    if [[ $? -eq 0 ]]; then
        openmp_results=$(extract_openmp_results "$openmp_log_file")
        IFS=',' read -r openmp_wall_time openmp_train_time openmp_test_mse openmp_test_mae <<< "$openmp_results"
        
        # Calculate weak scaling efficiency
        if [[ -z "$baseline_openmp_time" && "$openmp_wall_time" != "ERROR" ]]; then
            baseline_openmp_time=$openmp_wall_time
            openmp_efficiency="1.00"
        elif [[ "$openmp_wall_time" != "ERROR" && -n "$baseline_openmp_time" ]]; then
            openmp_efficiency=$(echo "scale=3; $baseline_openmp_time / $openmp_wall_time" | bc -l)
        else
            openmp_efficiency="ERROR"
        fi
        
        echo "$cores,$cores,$total_trees,$openmp_wall_time,$openmp_train_time,$openmp_test_mse,$openmp_test_mae,$openmp_efficiency" >> "$OPENMP_RESULTS"
        echo "    Pure OpenMP: ${cores}T, Trees: $total_trees, Time: ${openmp_wall_time}ms, Efficiency: ${openmp_efficiency}"
    else
        echo "    Pure OpenMP: FAILED or TIMEOUT"
        echo "$cores,$cores,$total_trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR" >> "$OPENMP_RESULTS"
    fi
    
    rm -f "$openmp_log_file"
done

# Generate comparison report
{
    echo "# Weak Scaling Comparison Report"
    echo "# Date: $(date)"
    echo "# Base workload: $BASE_TREES_PER_CORE trees per core"
    echo "# Format: Cores,Total_Trees,MPI_Time_ms,OpenMP_Time_ms,MPI_Efficiency,OpenMP_Efficiency,Time_Ratio"
} > "$COMPARISON_RESULTS"

echo ""
echo "=========================================="
echo "Weak Scaling Test Results Summary"
echo "=========================================="
echo ""
echo "Cores | Trees | MPI Time (ms) | OpenMP Time (ms) | MPI Efficiency | OpenMP Efficiency | Time Ratio"
echo "------|-------|---------------|------------------|----------------|-------------------|------------"

# Read results and generate comparison
for cores in "${TEST_CONFIGS[@]}"; do
    total_trees=$((cores * BASE_TREES_PER_CORE))
    
    # Read MPI results
    mpi_line=$(grep "^$cores," "$MPI_RESULTS" | tail -1)
    if [[ -n "$mpi_line" ]]; then
        IFS=',' read -r cores_col mpi_procs omp_threads trees_col mpi_time mpi_train mpi_mse mpi_mae mpi_efficiency <<< "$mpi_line"
    else
        mpi_time="N/A"
        mpi_efficiency="N/A"
    fi
    
    # Read OpenMP results
    openmp_line=$(grep "^$cores," "$OPENMP_RESULTS" | tail -1)
    if [[ -n "$openmp_line" ]]; then
        IFS=',' read -r cores_col threads_col trees_col openmp_time openmp_train openmp_mse openmp_mae openmp_efficiency <<< "$openmp_line"
    else
        openmp_time="N/A"
        openmp_efficiency="N/A"
    fi
    
    # Calculate time ratio
    if [[ "$mpi_time" != "N/A" && "$mpi_time" != "ERROR" && "$mpi_time" != "TIMEOUT" && \
          "$openmp_time" != "N/A" && "$openmp_time" != "ERROR" && "$openmp_time" != "TIMEOUT" ]]; then
        time_ratio=$(echo "scale=2; $openmp_time / $mpi_time" | bc -l)
    else
        time_ratio="N/A"
    fi
    
    # Output to comparison file
    echo "$cores,$total_trees,$mpi_time,$openmp_time,$mpi_efficiency,$openmp_efficiency,$time_ratio" >> "$COMPARISON_RESULTS"
    
    # Formatted output to console
    printf "%5s | %5s | %13s | %16s | %14s | %17s | %10s\n" \
           "$cores" "$total_trees" "$mpi_time" "$openmp_time" "$mpi_efficiency" "$openmp_efficiency" "$time_ratio"
done

echo ""
echo "=========================================="
echo "Weak Scaling Performance Analysis"
echo "=========================================="

# Analyze weak scaling trends
echo ""
echo "Efficiency Analysis:"
echo "-------------------"

# Calculate average efficiencies
mpi_efficiencies=()
openmp_efficiencies=()

for cores in "${TEST_CONFIGS[@]}"; do
    if (( cores == 1 )); then continue; fi  # Skip baseline (cores=1)
    
    mpi_line=$(grep "^$cores," "$MPI_RESULTS" | tail -1)
    if [[ -n "$mpi_line" ]]; then
        IFS=',' read -r _ _ _ _ _ _ _ _ mpi_eff <<< "$mpi_line"
        if [[ "$mpi_eff" != "ERROR" && "$mpi_eff" != "N/A" ]]; then
            mpi_efficiencies+=("$mpi_eff")
        fi
    fi
    
    openmp_line=$(grep "^$cores," "$OPENMP_RESULTS" | tail -1)
    if [[ -n "$openmp_line" ]]; then
        IFS=',' read -r _ _ _ _ _ _ _ openmp_eff <<< "$openmp_line"
        if [[ "$openmp_eff" != "ERROR" && "$openmp_eff" != "N/A" ]]; then
            openmp_efficiencies+=("$openmp_eff")
        fi
    fi
done

if [[ ${#mpi_efficiencies[@]} -gt 0 ]]; then
    # Use printf for precise calculation with floating point
    mpi_avg_eff=$(printf "%.3f" "$(echo "${mpi_efficiencies[@]}" | tr ' ' '+' | bc -l) / ${#mpi_efficiencies[@]}")
    echo "MPI+OpenMP Average Efficiency (cores > 1): $mpi_avg_eff"
fi

if [[ ${#openmp_efficiencies[@]} -gt 0 ]]; then
    # Use printf for precise calculation with floating point
    openmp_avg_eff=$(printf "%.3f" "$(echo "${openmp_efficiencies[@]}" | tr ' ' '+' | bc -l) / ${#openmp_efficiencies[@]}")
    echo "Pure OpenMP Average Efficiency (cores > 1): $openmp_avg_eff"
fi

echo ""
echo "Results saved to:"
echo "  MPI+OpenMP results: $MPI_RESULTS"
echo "  Pure OpenMP results: $OPENMP_RESULTS"
echo "  Comparison report: $COMPARISON_RESULTS"
echo ""
echo "Weak Scaling Analysis Notes:"
echo "- Efficiency = BaselineTime / CurrentTime"
echo "- Ideal weak scaling: Efficiency close to 1.0 (constant time)"
echo "- Time Ratio = OpenMP_Time / MPI_Time (>1 means MPI is faster)"
echo "- Good weak scaling: Efficiency > 0.8"
echo "- Problem size scales linearly with cores: Trees = Cores × $BASE_TREES_PER_CORE"
echo "=========================================="