#!/bin/bash

# Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=0-01:00
#SBATCH --job-name=SLoc_Profile
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie

# =-=-=-=-=-=-=-=-=-=-=-=-=-
source /project/rrg-plavoie/denggua1/pd_env.sh
set -euo pipefail

# Simple profiling script for source_loc_cpp
# Assumes:
#   - You have an allocation with at least 196 cores total
#   - Modules for gcc/openmpi/python/rarray are loaded
#   - source_loc_cpp*.so is built in this directory
#   - test_source_loc_cpp_only.py is present and executable

# Problem size for the test
NODES=5000
NFREQ=512
NTARGET=32
LOGFILE="log_profile_source_loc_${NODES}nodes_$(date +%Y%m%d).log"

# List of (RANKS THREADS) combinations; product should be 196
COMBOS=(
    "192 1"
    "96 2"
    "48 4"
    "32 6"
    "24 8"
    "16 12"
    "12 16"
    "8 24"
    "6 32"
    "4 48"
    "2 96"
    "1 192"
)

echo "=== Profiling started at $(date) ===" | tee -a "$LOGFILE"
echo "Using nodes=${NODES}, nfreq=${NFREQ}, ntarget=${NTARGET}" | tee -a "$LOGFILE"

for combo in "${COMBOS[@]}"; do
    read -r RANKS THREADS <<< "$combo"

    echo "" | tee -a "$LOGFILE"
    echo "=== RANKS=${RANKS} THREADS=${THREADS} ===" | tee -a "$LOGFILE"
    export OMP_NUM_THREADS="${THREADS}"
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread

    mpirun -np "${RANKS}" python3 ./profile_source_loc.py \
        --nodes "${NODES}" --nfreq "${NFREQ}" --ntarget "${NTARGET}" \
        2>&1 | tee -a "$LOGFILE"

    echo "" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
done

echo "=== Profiling finished at $(date) ===" | tee -a "$LOGFILE"