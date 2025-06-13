#!/bin/bash
set -euo pipefail

# =============================================================================
# script/bagging/strong_scaling_comparison.sh
# 
# Strong Scaling Test: Compares MPI+OpenMP hybrid version vs. Pure OpenMP version.
# Fixed problem size, increases processor count, measures speedup and efficiency.
# =============================================================================

# Project root path
PROJECT_ROOT="$( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )"
source "$PROJECT_ROOT/script/env_config.sh" # Source OpenMP thread configuration

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

# Detailed check of executables and environment
echo "=== Environment Check ==="
echo "Checking MPI environment..."
if ! command -v mpirun &> /dev/null; then
    echo "ERROR: mpirun not found in PATH."
    echo "Please install MPI (e.g., sudo yum install openmpi openmpi-devel) or load the MPI module (e.g., module load mpi/openmpi)."
    exit 1
fi
echo "✓ mpirun found: $(which mpirun)"

echo "Checking executables..."
if [[ ! -f "$MPI_EXECUTABLE" ]]; then
    echo "ERROR: MPI executable not found: $MPI_EXECUTABLE."
    echo "Please build with: cmake -DENABLE_MPI=ON .. && make"
    echo "Available files in build/:"
    ls -la "$PROJECT_ROOT/build/" 2>/dev/null || echo "Build directory not found"
    exit 1
elif [[ ! -x "$MPI_EXECUTABLE" ]]; then
    echo "WARNING: MPI executable is not executable. Fixing permissions..."
    chmod +x "$MPI_EXECUTABLE"
fi
echo "✓ MPI executable: $MPI_EXECUTABLE"

if [[ ! -f "$OPENMP_EXECUTABLE" ]]; then
    echo "ERROR: OpenMP executable not found: $OPENMP_EXECUTABLE."
    echo "Please build the project first."
    exit 1
elif [[ ! -x "$OPENMP_EXECUTABLE" ]]; then
    echo "WARNING: OpenMP executable is not executable. Fixing permissions..."
    chmod +x "$OPENMP_EXECUTABLE"
fi
echo "✓ OpenMP executable: $OPENMP_EXECUTABLE"

# Data path - using relative path
DATA_PATH="../data/data_clean/cleaned_data.csv"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

# Output directory (back to project root's script/bagging)
RESULTS_DIR="$PROJECT_ROOT/script/bagging"
mkdir -p "$RESULTS_DIR"

# Result files
MPI_RESULTS="$RESULTS_DIR/strong_scaling_mpi_results.csv"
OPENMP_RESULTS="$RESULTS_DIR/strong_scaling_openmp_results.csv"
COMPARISON_RESULTS="$RESULTS_DIR/strong_scaling_comparison.csv"

# Fixed test parameters (strong scaling: fixed problem size)
FIXED_NUM_TREES=100          # Fixed number of trees
FIXED_SAMPLE_RATIO=1.0       # Fixed sample ratio
FIXED_MAX_DEPTH=10           # Fixed maximum tree depth
FIXED_MIN_SAMPLES_LEAF=2     # Fixed minimum samples per leaf
FIXED_CRITERION="mse"        # Fixed criterion
FIXED_SPLIT_METHOD="random"  # Fixed split method
FIXED_PRUNER_TYPE="none"     # Fixed pruner type
# Note: MPI version requires 8 parameters, OpenMP version 10 (including prunerParam and seed)

# CPU architecture related configuration
PHYSICAL_CORES=36  # Example: 2 sockets * 18 cores/socket
MAX_CORES=$PHYSICAL_CORES

# Test configurations: number of processes/threads
# Full 36-core test sequence
TEST_CONFIGS=(1 2 4 6 9 12 15 18 21 24 27 30 33 36)

echo "=========================================="
echo "    Strong Scaling Comparison Test       "
echo "=========================================="
echo "CPU Architecture: Intel Xeon E5-2699 v3"
echo "Physical Cores: $PHYSICAL_CORES (2 sockets × 18 cores/socket)"
echo "Test Date: $(date)"
echo ""
echo "Test Range: 1 to $PHYSICAL_CORES cores"
echo "Test Points: ${TEST_CONFIGS[*]}"
echo ""
echo "Fixed Parameters (Strong Scaling):"
echo "  Trees: $FIXED_NUM_TREES"
echo "  Sample Ratio: $FIXED_SAMPLE_RATIO"
echo "  Max Depth: $FIXED_MAX_DEPTH"
echo "  Min Samples Leaf: $FIXED_MIN_SAMPLES_LEAF"
echo "  Criterion: $FIXED_CRITERION"
echo "  Split Method: $FIXED_SPLIT_METHOD"
echo "  Pruner: $FIXED_PRUNER_TYPE"
echo "  Data: $(basename "$DATA_PATH")"
echo "  Note: MPI version uses 8 parameters, OpenMP version uses 10 parameters"
echo ""
echo "Expected Runtime: 2-3 hours (14 configurations × 2 versions × ~5min each)"
echo ""

# Create result file headers
{
    echo "# Strong Scaling Test Results - MPI+OpenMP Version"
    echo "# Date: $(date)"
    echo "# Fixed Problem Size: $FIXED_NUM_TREES trees"
    echo "# Format: MPI_Processes,OpenMP_Threads,Total_Cores,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Speedup,Efficiency"
} > "$MPI_RESULTS"

{
    echo "# Strong Scaling Test Results - Pure OpenMP Version"
    echo "# Date: $(date)"
    echo "# Fixed Problem Size: $FIXED_NUM_TREES trees"
    echo "# Format: OpenMP_Threads,Wall_Time_ms,Train_Time_ms,Test_MSE,Test_MAE,Speedup,Efficiency"
} > "$OPENMP_RESULTS"

# Helper function: Extract results from MPI log file
extract_mpi_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    # Extract time information from MPI output
    wall_time=$(grep -E "Total time.*including communication" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Max training time across processes" "$log_file" | sed -n 's/.*: \([0-9]*\)ms.*/\1/p' | tail -1)
    
    # Extract accuracy information
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    # Set default values if not found
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

# Helper function: Extract results from OpenMP log file
extract_openmp_results() {
    local log_file="$1"
    local wall_time train_time test_mse test_mae
    
    # Extract time information from OpenMP output
    wall_time=$(grep -E "Total Time:" "$log_file" | sed -n 's/.*Total Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    train_time=$(grep -E "Train Time:" "$log_file" | sed -n 's/.*Train Time: *\([0-9]*\)ms.*/\1/p' | tail -1)
    
    # Extract accuracy information
    test_mse=$(grep -E "Test MSE:" "$log_file" | sed -n 's/.*Test MSE: *\([0-9.-]*\).*/\1/p' | tail -1)
    test_mae=$(grep -E "Test MAE:" "$log_file" | sed -n 's/.*Test MAE: *\([0-9.-]*\).*/\1/p' | tail -1)
    
    # Set default values if not found
    [[ -z "$wall_time" ]] && wall_time="ERROR"
    [[ -z "$train_time" ]] && train_time="ERROR"
    [[ -z "$test_mse" ]] && test_mse="ERROR"
    [[ -z "$test_mae" ]] && test_mae="ERROR"
    
    echo "$wall_time,$train_time,$test_mse,$test_mae"
}

# Baseline times (single-core performance)
baseline_mpi_time=""
baseline_openmp_time=""

echo "=========================================="
echo "Starting Strong Scaling Tests..."
echo "=========================================="

# Test different configurations
for config in "${TEST_CONFIGS[@]}"; do
    echo ""
    echo "Testing configuration: $config cores"
    echo "----------------------------------------"
    
    # === MPI+OpenMP Hybrid Test ===
    echo "  [1/2] Testing MPI+OpenMP version..."
    
    # Calculate MPI processes and OpenMP threads per process
    # Intelligent allocation strategy: balance MPI processes and OpenMP threads
    if (( config == 1 )); then
        mpi_procs=1
        omp_threads=1
    elif (( config <= 6 )); then
        mpi_procs=$config    # Small scale: one process per core
        omp_threads=1
    elif (( config <= 18 )); then
        # Medium scale: approximately one process per 2-3 cores
        mpi_procs=$(( (config + 1) / 2 ))
        omp_threads=2
    else
        # Large scale: approximately one process per 3-4 cores, leveraging NUMA optimization
        mpi_procs=$(( (config + 2) / 3 ))
        omp_threads=3
        # Adjust omp_threads if it doesn't divide evenly
        if (( mpi_procs * omp_threads < config )); then
            omp_threads=$(( (config + mpi_procs - 1) / mpi_procs ))
        fi
    fi
    
    # Ensure number of processes does not exceed total cores
    mpi_procs=$(( mpi_procs > config ? config : mpi_procs ))
    mpi_procs=$(( mpi_procs < 1 ? 1 : mpi_procs ))
    
    # Set OpenMP environment variables
    export OMP_NUM_THREADS=$omp_threads
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    
    mpi_log_file="temp_mpi_${config}_cores.log"
    
    # Construct and display MPI command
    mpi_cmd="timeout 600 mpirun -np $mpi_procs $MPI_EXECUTABLE $DATA_PATH $FIXED_NUM_TREES $FIXED_SAMPLE_RATIO $FIXED_MAX_DEPTH $FIXED_MIN_SAMPLES_LEAF $FIXED_CRITERION $FIXED_SPLIT_METHOD $FIXED_PRUNER_TYPE"
    echo "    Command: $mpi_cmd"
    echo "    Environment: OMP_NUM_THREADS=$omp_threads"
    echo "    Running..."
    
    # Run MPI version
    timeout 600 mpirun -np $mpi_procs \
        "$MPI_EXECUTABLE" \
        "$DATA_PATH" \
        $FIXED_NUM_TREES \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        > "$mpi_log_file" 2>&1
    
    mpi_exit_code=$?
    echo "    Exit code: $mpi_exit_code"
    
    if [[ $mpi_exit_code -eq 0 ]]; then
        mpi_results=$(extract_mpi_results "$mpi_log_file")
        IFS=',' read -r mpi_wall_time mpi_train_time mpi_test_mse mpi_test_mae <<< "$mpi_results"
        
        if [[ -z "$baseline_mpi_time" && "$mpi_wall_time" != "ERROR" \
              && "$mpi_procs" -eq 1 && "$omp_threads" -eq 1 ]]; then
            baseline_mpi_time=$mpi_wall_time
            mpi_speedup="1.00"
            mpi_efficiency="1.00"
        elif [[ "$mpi_wall_time" != "ERROR" && -n "$baseline_mpi_time" ]]; then
            # Speedup for other configurations = Single-process single-thread time / Current configuration time
            mpi_speedup=$(echo "scale=2; $baseline_mpi_time / $mpi_wall_time" | bc -l)
            # Efficiency = Speedup / Total cores
            mpi_efficiency=$(echo "scale=3; $mpi_speedup / $config" | bc -l)
        else
            mpi_speedup="ERROR"
            mpi_efficiency="ERROR"
        fi
        echo "$mpi_procs,$omp_threads,$config,$mpi_wall_time,$mpi_train_time,$mpi_test_mse,$mpi_test_mae,$mpi_speedup,$mpi_efficiency" >> "$MPI_RESULTS"
        echo "    MPI+OpenMP: ${mpi_procs}P×${omp_threads}T, Time: ${mpi_wall_time}ms, Speedup: ${mpi_speedup}"
    elif [[ $mpi_exit_code -eq 124 ]]; then
        echo "    MPI+OpenMP: TIMEOUT (>600s)"
        echo "    Last 10 lines of output:"
        tail -10 "$mpi_log_file" 2>/dev/null || echo "    (no output)"
        echo "$mpi_procs,$omp_threads,$config,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR,ERROR" >> "$MPI_RESULTS"
    else
        echo "    MPI+OpenMP: FAILED (exit code: $mpi_exit_code)"
        echo "    Error output:"
        tail -20 "$mpi_log_file" 2>/dev/null || echo "    (no output)"
        echo "$mpi_procs,$omp_threads,$config,FAILED,FAILED,FAILED,FAILED,ERROR,ERROR" >> "$MPI_RESULTS"
    fi
    
    # Retain log file for debugging on failure, delete on success
    if [[ $mpi_exit_code -eq 0 ]]; then
        rm -f "$mpi_log_file"
    else
        echo "    Keeping log file for debugging: $mpi_log_file"
    fi
    
    # === Pure OpenMP Test ===
    echo "  [2/2] Testing Pure OpenMP version..."
    
    # Set OpenMP environment variables
    export OMP_NUM_THREADS=$config
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    
    openmp_log_file="temp_openmp_${config}_cores.log"
    
    # Construct and display OpenMP command (including all parameters)
    openmp_cmd="timeout 600 $OPENMP_EXECUTABLE bagging $DATA_PATH $FIXED_NUM_TREES $FIXED_SAMPLE_RATIO $FIXED_MAX_DEPTH $FIXED_MIN_SAMPLES_LEAF $FIXED_CRITERION $FIXED_SPLIT_METHOD $FIXED_PRUNER_TYPE 0.01 42"
    echo "    Command: $openmp_cmd"
    echo "    Environment: OMP_NUM_THREADS=$config"
    echo "    Running..."
    
    # Run OpenMP version (includes prunerParam and seed)
    timeout 600 "$OPENMP_EXECUTABLE" bagging \
        "$DATA_PATH" \
        $FIXED_NUM_TREES \
        $FIXED_SAMPLE_RATIO \
        $FIXED_MAX_DEPTH \
        $FIXED_MIN_SAMPLES_LEAF \
        "$FIXED_CRITERION" \
        "$FIXED_SPLIT_METHOD" \
        "$FIXED_PRUNER_TYPE" \
        0.01 \
        42 \
        > "$openmp_log_file" 2>&1
    
    openmp_exit_code=$?
    echo "    Exit code: $openmp_exit_code"
    
    if [[ $openmp_exit_code -eq 0 ]]; then
        openmp_results=$(extract_openmp_results "$openmp_log_file")
        IFS=',' read -r openmp_wall_time openmp_train_time openmp_test_mse openmp_test_mae <<< "$openmp_results"
        
        # Calculate speedup and efficiency for OpenMP version (based on single-thread OpenMP performance)
        if [[ -z "$baseline_openmp_time" && "$openmp_wall_time" != "ERROR" && "$config" -eq 1 ]]; then
            baseline_openmp_time=$openmp_wall_time
            openmp_speedup="1.00"
            openmp_efficiency="1.00"
        elif [[ "$openmp_wall_time" != "ERROR" && -n "$baseline_openmp_time" ]]; then
            openmp_speedup=$(echo "scale=2; $baseline_openmp_time / $openmp_wall_time" | bc -l)
            openmp_efficiency=$(echo "scale=3; $openmp_speedup / $config" | bc -l)
        else
            openmp_speedup="ERROR"
            openmp_efficiency="ERROR"
        fi
        
        echo "$config,$openmp_wall_time,$openmp_train_time,$openmp_test_mse,$openmp_test_mae,$openmp_speedup,$openmp_efficiency" >> "$OPENMP_RESULTS"
        echo "    Pure OpenMP: ${config}T, Time: ${openmp_wall_time}ms, Speedup: ${openmp_speedup}"
    elif [[ $openmp_exit_code -eq 124 ]]; then
        echo "    Pure OpenMP: TIMEOUT (>600s)"
        echo "    Last 10 lines of output:"
        tail -10 "$openmp_log_file" 2>/dev/null || echo "    (no output)"
        echo "$config,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,ERROR,ERROR" >> "$OPENMP_RESULTS"
    else
        echo "    Pure OpenMP: FAILED (exit code: $openmp_exit_code)"
        echo "    Error output:"
        tail -20 "$openmp_log_file" 2>/dev/null || echo "    (no output)"
        echo "$config,FAILED,FAILED,FAILED,FAILED,ERROR,ERROR" >> "$OPENMP_RESULTS"
    fi
    
    # Retain log file for debugging on failure, delete on success
    if [[ $openmp_exit_code -eq 0 ]]; then
        rm -f "$openmp_log_file"
    else
        echo "    Keeping log file for debugging: $openmp_log_file"
    fi
done

# Generate comparison report
{
    echo "# Strong Scaling Comparison Report"
    echo "# Date: $(date)"
    echo "# Fixed Problem Size: $FIXED_NUM_TREES trees"
    echo "# Format: Cores,MPI_Config,OpenMP_Config,MPI_Time_ms,OpenMP_Time_ms,MPI_Speedup,OpenMP_Speedup,MPI_Efficiency,OpenMP_Efficiency,Relative_Performance"
} > "$COMPARISON_RESULTS"

echo ""
echo "=========================================="
echo "Strong Scaling Test Results Summary"
echo "=========================================="
echo ""
echo "Cores | MPI Config | OpenMP Config | MPI Time | OpenMP Time | MPI Speedup | OpenMP Speedup | MPI Efficiency | OpenMP Efficiency"
echo "------|------------|---------------|----------|-------------|-------------|----------------|----------------|------------------"

# Read results and generate comparison
for config in "${TEST_CONFIGS[@]}"; do
    # Read MPI results
    mpi_line=$(grep ",$config," "$MPI_RESULTS" | tail -1)
    if [[ -n "$mpi_line" ]]; then
        IFS=',' read -r mpi_procs omp_threads cores mpi_time mpi_train mpi_mse mpi_mae mpi_speedup mpi_eff <<< "$mpi_line"
        mpi_config="${mpi_procs}P×${omp_threads}T"
    else
        mpi_time="N/A"
        mpi_speedup="N/A"
        mpi_eff="N/A"
        mpi_config="N/A"
    fi
    
    # Read OpenMP results
    openmp_line=$(grep "^$config," "$OPENMP_RESULTS" | tail -1)
    if [[ -n "$openmp_line" ]]; then
        IFS=',' read -r threads openmp_time openmp_train openmp_mse openmp_mae openmp_speedup openmp_eff <<< "$openmp_line"
        openmp_config="${threads}T"
    else
        openmp_time="N/A"
        openmp_speedup="N/A"
        openmp_eff="N/A"
        openmp_config="N/A"
    fi
    
    # Calculate relative performance
    if [[ "$mpi_time" != "N/A" && "$mpi_time" != "ERROR" && "$mpi_time" != "TIMEOUT" && \
          "$openmp_time" != "N/A" && "$openmp_time" != "ERROR" && "$openmp_time" != "TIMEOUT" ]]; then
        relative_perf=$(echo "scale=2; $openmp_time / $mpi_time" | bc -l)
    else
        relative_perf="N/A"
    fi
    
    # Output to comparison file
    echo "$config,$mpi_config,$openmp_config,$mpi_time,$openmp_time,$mpi_speedup,$openmp_speedup,$mpi_eff,$openmp_eff,$relative_perf" >> "$COMPARISON_RESULTS"
    
    # Formatted output
    printf "%5s | %10s | %13s | %8s | %11s | %11s | %14s | %14s | %17s\n" \
           "$config" "$mpi_config" "$openmp_config" "$mpi_time" "$openmp_time" "$mpi_speedup" "$openmp_speedup" "$mpi_eff" "$openmp_eff"
done

echo ""
echo "=========================================="
echo "Results saved to:"
echo "  MPI+OpenMP results: $MPI_RESULTS"
echo "  Pure OpenMP results: $OPENMP_RESULTS"
echo "  Comparison report: $COMPARISON_RESULTS"
echo ""
echo "Analysis Notes:"
echo "- MPI Speedup = MPI_SingleProcess_Time / MPI_Current_Time"
echo "- OpenMP Speedup = OpenMP_SingleThread_Time / OpenMP_Current_Time"
echo "- Efficiency = Speedup / NumberOfCores"
echo "- Relative Performance = OpenMP_Time / MPI_Time (>1 means MPI is faster)"
echo "- Ideal strong scaling: Linear speedup, Efficiency close to 1.0"
echo "- MPI Config: NP×NT means N processes × T threads per process"
echo "=========================================="