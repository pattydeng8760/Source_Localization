# **Source Localization Framework for Turbulence-Induced Surface Sound**

### Implementation of the Method from:
**Delfs, J.W. & Ruck, R. (2024).  
“A quantity to identify turbulence related sound generation on surfaces.”**  
*Journal of Sound and Vibration, 586, 118490.*  
[https://doi.org/10.1016/j.jsv.2024.118490]

---

## **Overview**

This repository provides a Python implementation of the **aeroacoustic source localization method** proposed by Delfs & Ruck (2024), which directly identifies turbulence-related sound generation regions on aerodynamic surfaces using **surface pressure data** from scale-resolving simulations (SRS/LES/DNS).  

Unlike conventional beamforming or FW–H propagation, this approach operates **entirely in the nearfield**, filtering the fluctuating surface pressure to isolate the **acoustically active component** responsible for sound generation.

The central idea is based on the **mirror principle violation**:  
sound is generated where the incident nearfield pressure no longer satisfies the ideal mirror symmetry condition, i.e. where curvature or diffraction induces acoustic radiation.

---

## **Method Summary**

### 1. Mathematical Foundation

The method derives from the **Kirchhoff surface integral formulation** of the Ffowcs Williams–Hawkings (FW–H) equation, simplified for low Mach number flow with the assumption that the pressure gradient normal to the surface is negligible ($\partial\hat{p}/\partial n \approx 0$).

Given the total surface pressure ($$\hat{p}(\mathbf{x})$$) in the frequency domain, the governing **boundary integral equation** for an observer on the surface ($\mathbf{x} \in \partial V_B$) is (Eq. 10 in the article):

$$
\hat{p}(\mathbf{x}) = 2\hat{p}_{f}(\mathbf{x}) + \frac{1}{2\pi} \oint_{\partial V_B} \exp(-ikr) (ikr + 1) \frac{\mathbf{e}_r \cdot \mathbf{n}}{r^2} \hat{p}(\boldsymbol{\xi}) dS(\boldsymbol{\xi})
$$

Where:
* $\oint_{\partial V_B}$ denotes the **Cauchy principal value integral** ($\text{P.V.}$).
* $\hat{p}_{f}(\mathbf{x})$ is the **free field pressure** incident on the surface.
* $k = \omega / a_\infty$ is the **wavenumber**.
* $\mathbf{e}_r$ is the **unit vector** from the source point $\boldsymbol{\xi}$ to the observer point $\mathbf{x}$.
* $\mathbf{n}$ is the **local surface normal**.

The **Acoustic Surface Pressure** ($$\hat{p}_{S}(\mathbf{x})$$) is defined by rearranging the equation, based on the physical hypothesis of the mirror principle (Eq. 16 in the article):

$$
\hat{p}_{S}(\mathbf{x}) = \hat{p}(\mathbf{x}) - 2\hat{p}_{f}(\mathbf{x})
$$

This quantity, $$\hat{p}_{s}(\mathbf{x})$$, which is also called the **Surface Source Quantity** ($\hat{q}$ in the article), is mathematically equivalent to the **diffraction integral**:

$$
\hat{p}_{S}(\mathbf{x}) = \frac{1}{2\pi} \oint_{\partial V_B} \exp(-ikr) (ikr + 1) \frac{\mathbf{e}_r \cdot \mathbf{n}}{r^2} \hat{p}(\boldsymbol{\xi}) dS(\boldsymbol{\xi})
$$

This quantity ($$\hat{p}_S(\mathbf{x})$$) provides a **scalar, observer-independent map** of the local sound generation intensity, as it isolates the acoustically active component—the part of the surface pressure relevant for sound generation.

---

## **Repository Structure**

```
Source_Localization/
│
├── source_localization_core/                    # Main Python + C++ compute package
│   │
│   ├── __init__.py
│   ├── __main__.py                              # CLI entry point (supports MPI mode)
│   │
│   ├── SourceLocalization.py                    # Main workflow class (loading, execution, output)
│   ├── source_localization_func.py              # Full Python implementation (reference)
│   ├── source_localization_cpp.py               # Python wrapper for C++ MPI/OpenMP backend
│   ├── source_localization_block.py             # Python block-distribution utilities
│   ├── source_localization_utils.py             # Shared utilities for the SourceLocalization class with deprecated functions (timing, math helpers, printing)
│   ├── fft_surface.py                           # Fourier Transform computation, either via scipy.fft or manual compute
│   ├── extract.py                               # Data I/O: reads .h5 surface pressure datasets and extracts the airfoil surface mesh from the domain 
│   ├── utils.py                                 # Utilities for coordinate handling, geometry, normalization
│
├── source_localization_cpp/                     # C++ build folder (not importable)
│   ├── Makefile                                 # Makefile for C++ programme
│   ├── source_localization_worker.cpp           # The worker function to calculate the inner loop for a single observer position, accelerated via OpenMP
│   ├── source_localization_worker.h
│   ├── source_localization_mpi.cpp              # The main function to calculate all surface points for the integration, accelerated via MPI
│   └── source_localization_mpi.h
│
└── source_localization_profile/                 # C++ profiling folder
    ├── profile_source_loc.py                    # C++ performance validation
    ├── profile_source_loc.sh                    # shell script for performance validation
    └── compare_outputs.py                       # Comparing the C++ outputs and python outputs for validation
```

---

## **Usage**

### **1. Input Data**

The code expects **surface pressure fields** (complex or time-domain) from scale resolving simulation, stored in an `.h5` file extracted with FWH_extract from AVBP with attributes:

- `pressure` – time-resolved surface pressure (N:nodes × M:time array)  
- `x`, `y`, `z` – coordinates of surface points  
- `n` – local surface normal vector
- `dt` – time step (for FFT)  

---

### **2. Running the Localization**

**Command-line:**
```bash
python -m source_localization.py --mesh_dir <> --mesh_file <> --FWH_data_dir <> --freq_select [1000,2000,3000] --source_localization True --surface_patches ["Airfoil_Surface","Airfoil_Side"],
```

**run_source_loc.py:**
Execute the module from a wrapper function. Note that the C++ method can only be submitted via bash script to set the number of MPI and OMP processes.
```python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Source_Localization")

from source_localization_core import main

config = {
    "working_dir"       : "./",
    "mesh_dir"          : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/MESH_Medium_Aug25",
    "mesh_file"         : "Bombardier_10AOA_U50_Combine_Medium.mesh.h5",
    "FWH_data_dir"      : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/RUN_Medium/FWH_Airfoil/FWH_Data_TTG",
    "var"               : "pressure",
    "reload"            : False,
    "extract_FWH"       : False,
    "freq_select"       : [1000,2000,3000],
    "source_localization": True,
    "fft_method"        : "FFT",
    "surface_patches"   : ["Airfoil_Surface","Airfoil_Trailing_Edge","Airfoil_Side_LE","Airfoil_Side_Mid","Airfoil_Side_TE"],
    "compute_method"    : "python"
}

args = Namespace(**config)
main(args)
```

**run_source_loc_cpp.sh:**
Execute the module with C++ backend for acceleration.
```bash
#!/bin/bash
#SBATCH --job-name=B10_SourceLoc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8        # 8 MPI ranks
#SBATCH --cpus-per-task=24         # 24 OMP threads per rank
#SBATCH --time=12:00:00

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
```
**Output**
This computes:
- `p`: surface pressure flucuations 
- `p_f`: reduced acoustic surface pressure \( \hat{p}_f \)
- `p_s`: reduced acoustic surface pressure \( \hat{p}_s \)
---

### **3. Visualization**

Use paraview to view the results in .h5 format 

---

## **Dependencies**

- Python ≥ 3.10  
- NumPy ≥ 1.24  
- SciPy ≥ 1.11  
- h5py ≥ 3.15  
- Matplotlib ≥ 3.8  
- numba ≥ 0.58
- future ≥ 1.0.0

---

## **Example Output**

Typical outputs include:
- 2D surface contour plots of |p|, |p_f|, and |p_s|  
- Line profiles across suction/pressure sides  
- Far-field comparison using Kirchhoff/FW–H propagation  

| Quantity | Description |
|-----------|--------------|
| `p_hat` | Total surface pressure (LES) in the frequency domain |
| `p_f` | Freefield pressure in the frequency domain (reflection component)|
| `p_s` | Acoustic surface pressure (diffraction component) |

---

## **References**

1. **Delfs, J.W. & Ruck, R.** (2024).  
   *A quantity to identify turbulence related sound generation on surfaces.*  
   *Journal of Sound and Vibration*, 586, 118490.  
   DOI: [10.1016/j.jsv.2024.118490](https://doi.org/10.1016/j.jsv.2024.118490)

2. **Curle, N.** (1955). *The influence of solid boundaries upon aerodynamic sound.*  
   *Proceedings of the Royal Society A.*

3. **Ffowcs Williams, J.E. & Hawkings, D.L.** (1969). *Sound generation by turbulence and surfaces in arbitrary motion.*  
   *Philosophical Transactions of the Royal Society A.*

---

## **License**

This implementation is provided for **research and educational use** only.  
Please cite both the original **Delfs & Ruck (2024)** article and this repository if used in published work.
