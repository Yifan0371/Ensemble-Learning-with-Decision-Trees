#!/bin/bash
# =============================================================================
# script/env_config.sh
# Functionality:
#   1) Detects physical CPU cores (excluding hyperthreading) and sets OMP_NUM_THREADS
#   2) No longer auto-detects or compiles executables; other scripts can source this file to get the correct OMP_NUM_THREADS.
# =============================================================================

# -------- 1. Detect and Set Physical Core Count (excluding hyperthreading) --------
if ! command -v lscpu &> /dev/null; then
    echo "Warning: lscpu not available, defaulting OMP_NUM_THREADS=1"
    export OMP_NUM_THREADS=1
else
    # First, try English fields: Socket(s) and Core(s) per socket
    SOCKETS=$(lscpu | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2}')
    CORES_PS=$(lscpu | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2}')
    # If English fields are not found, try Chinese fields
    if [[ -z "$SOCKETS" || -z "$CORES_PS" ]]; then
        SOCKETS=$(lscpu | awk -F: '/^ *座：/ {gsub(/ /,"",$2); print $2}')
        CORES_PS=$(lscpu | awk -F: '/^ *每个座的核数：/ {gsub(/ /,"",$2); print $2}')
    fi

    # Calculate physical cores based on obtained results
    if [[ -n "$SOCKETS" && -n "$CORES_PS" ]]; then
        PHYS=$(( SOCKETS * CORES_PS ))
    else
        # If neither English nor Chinese fields are found, fall back to total logical cores divided by 2
        LOGICAL=$(lscpu | awk -F: '/^CPU\(s\):/ {gsub(/ /,"",$2); print $2}')
        if [[ -z "$LOGICAL" ]]; then
            LOGICAL=$(lscpu | awk -F: '/^CPU：/ {gsub(/ /,"",$2); print $2}')
            if [[ -z "$LOGICAL" ]]; then
                LOGICAL=$(lscpu | awk -F: '/^CPU:/ {gsub(/ /,"",$2); print $2}')
            fi
        fi
        if [[ -n "$LOGICAL" ]]; then
            PHYS=$(( LOGICAL / 2 ))
        else
            PHYS=1
        fi
    fi

    # Ensure at least 1 core
    if [[ -z "$PHYS" || "$PHYS" -lt 1 ]]; then
        PHYS=1
    fi

    export OMP_NUM_THREADS=$PHYS
    echo "Physical cores (excluding hyperthreading): $PHYS, OMP_NUM_THREADS=$OMP_NUM_THREADS has been set"
fi
