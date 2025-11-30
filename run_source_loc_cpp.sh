#!/bin/bash
#SBATCH --job-name=B10_SourceLoc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8        # 8 MPI ranks
#SBATCH --cpus-per-task=24         # 24 OMP threads per rank
#SBATCH --time=12:00:00
#SBATCH --account=rrg-plavoie
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL

# ===== Environment Setup =====
echo "Loading environment..."
source /project/rrg-plavoie/denggua1/pd_env.sh
set -euxo pipefail

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# ===== User Inputs =====
WORKING_DIR="./"
MESH_DIR="/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/MESH_Medium_Aug25"
MESH_FILE="Bombardier_10AOA_U50_Combine_Medium.mesh.h5"

FWH_DATA_DIR="/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/RUN_Medium/FWH_Airfoil/FWH_Data_TTG"

VAR="pressure"
OPTION=2
NSKIP=1
MIN_FILE=5000
MAX_FILE=10000
FREQ_SELECT=1500

FFT_METHOD="FFT"
COMPUTE_METHOD="C"  # C++ backend
SOURCE_LOCALIZATION="True"

SURFACE_PATCHES=(
  "Airfoil_Surface"
  "Airfoil_Trailing_Edge"
  "Airfoil_Side_LE"
  "Airfoil_Side_Mid"
  "Airfoil_Side_TE"
)

# ===== MPI + OpenMP Control =====
export OMP_NUM_THREADS=24
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

echo "MPI ranks: ${SLURM_NTASKS_PER_NODE:-8}, OMP threads: ${OMP_NUM_THREADS}"
echo "Running source localization (C++ backend)..."

# ===== Run Application =====
mpirun -np 8 python3 -m source_localization_core \
  --working-dir "${WORKING_DIR}" \
  --mesh-dir "${MESH_DIR}" \
  --mesh-file "${MESH_FILE}" \
  --FWH-data-dir "${FWH_DATA_DIR}" \
  --var "${VAR}" \
  --option "${OPTION}" \
  --nskip "${NSKIP}" \
  --min-file "${MIN_FILE}" \
  --max-file "${MAX_FILE}" \
  --freq-select "${FREQ_SELECT}" \
  --fft-method "${FFT_METHOD}" \
  --compute-method "${COMPUTE_METHOD}" \
  --surface-patches "${SURFACE_PATCHES[@]}" \
  --source-localization "${SOURCE_LOCALIZATION}" \
  --num-ranks "${SLURM_NTASKS_PER_NODE:-8}" \
  --num-threads "${OMP_NUM_THREADS}" \
  2>&1 | tee "log_SourceLoc_Cpp_${FREQ_SELECT}Hz_$(date +%Y%m%d%H%M).log"

echo "==== DONE ===="