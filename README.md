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

The method derives from the **Kirchhoff surface integral formulation** of the FW–H equation.
Given the total surface pressure ($$\hat{p}(\mathbf{x})$$), the acoustically relevant component ($\hat{p}_S$) is defined by:

$$
\hat{p}_S(\mathbf{x}) = \frac{1}{2\pi} \text{P.V.} \int_{\partial V_B} e^{-ikr} (ikr + 1) \frac{\mathbf{e}_r \cdot \mathbf{n}}{r^2} \hat{p}(\boldsymbol{\xi}) dS(\boldsymbol{\xi})
$$

where $\text{P.V.}$ denotes the **Cauchy principal value integral**,
$k = \omega / a_\infty$ is the **wavenumber**,
$\mathbf{e}_r$ is the **unit vector** between surface points, and
$\mathbf{n}$ is the **local surface normal**.
The **surface source term** ($q(\mathbf{x})$) is subsequently defined as:

$$
q(\mathbf{x}) = \frac{1}{2} \hat{p}_S - \frac{1}{4\pi} \text{P.V.} \int_{\partial V_B} e^{-ikr} (ikr + 1) \frac{\mathbf{e}_r \cdot \mathbf{n}}{r^2} \hat{p}_S(\boldsymbol{\xi}) dS(\boldsymbol{\xi})
$$

This quantity ($$q$$) provides a **scalar, observer-independent map** of local sound generation intensity.

---

## **Repository Structure**

```
source_localization/
│
├── source_localization.py           # Main driver for source localization
├── source_localization_func.py      # Core implementation of Eqs. (16)–(17)
├── source_localization_func_rev.py  # Revised version with optimizations & vectorization
├── fft_surface.py                   # Frequency-domain transformations (Welch FFT, filtering)
├── extract.py                       # Data I/O: reads .h5 surface pressure datasets
├── utils.py                         # Utilities for coordinate handling, geometry, normalization
├── __init__.py                      # Package initialization
└── JSV_586_2024...pdf               # Reference article (Delfs & Ruck, 2024)
```

---

## **Usage**

### **1. Input Data**

The code expects **surface pressure fields** (complex or time-domain) from LES or experimental measurements, typically stored in an `.h5` file with attributes:

- `Pressure` – time-resolved surface pressure (N × M array)  
- `x`, `y`, `z` – coordinates of surface points  
- `n` – local surface normals  
- `dt` – time step (for FFT)  
- Optional attributes: `rho`, `Uinf`, `a_inf`

---

### **2. Running the Localization**

**Command-line:**
```bash
python source_localization.py --input surface_data.h5 --freq 2000 --nchunk 8
```

**Python API:**
```python
from source_localization_func_rev import compute_source_quantity

p_s, q = compute_source_quantity(pressure, coords, normals, freq, a_inf=343.0)
```

This computes:
- `p_s`: reduced acoustic surface pressure \( \hat{p}_S \)
- `q`: localized source term proportional to diffraction intensity

---

### **3. Visualization**

Use the utilities in `fft_surface.py` to visualize spectral maps or compare results:

```python
from fft_surface import plot_surface_map

plot_surface_map(coords, abs(q), title="Source Strength |q|", cmap="plasma")
```

---

## **Validation and Comparison**

The implementation reproduces the key findings of Delfs & Ruck (2024):

- The **acoustic surface pressure** \( \hat{p}_S \) isolates leading and trailing edge regions as dominant sound sources.  
- The **source quantity** \( q \) provides sharp localization independent of wavelength, consistent with diffraction-based theory.  
- Far-field spectra computed with \( \hat{p}_S \) and with the original \( \hat{p} \) are nearly identical, confirming physical consistency.

---

## **Dependencies**

- Python ≥ 3.10  
- NumPy ≥ 1.24  
- SciPy ≥ 1.11  
- h5py ≥ 3.9  
- Matplotlib ≥ 3.8  
- tqdm (for progress bars)

**Optional:**  
- cupy (for GPU acceleration)  
- mpi4py (for distributed surface integration)

---

## **Example Output**

Typical outputs include:
- 2D surface contour plots of |p|, |p_S|, and |q|  
- Line profiles across suction/pressure sides  
- Far-field comparison using Kirchhoff/FW–H propagation  

| Quantity | Description |
|-----------|--------------|
| `p_hat` | Total surface pressure (LES) |
| `p_s` | Acoustic surface pressure (diffraction component) |
| `q` | Source localization quantity |
| `SPL(f)` | Sound pressure spectrum |

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
