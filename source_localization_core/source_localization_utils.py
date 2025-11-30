import numpy as np
import h5py
import os
import math
import logging
from .utils import print
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
from antares import Reader, Base, Zone, Instant, Writer

# function to setup logging for each worker process
def _setup_worker_logger(log_file):
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="a")],  # append, not overwrite
        force=True
    )
    for h in logging.getLogger().handlers:
        h.flush = h.stream.flush
        
def load_airfoil_mesh(working_dir):
    """
    Load airfoil mesh from HDF5 file and extract specified surface patches.

    Parameters
    ----------
    mesh_file : str
        Path to the HDF5 mesh file.
    surface_patches : list of str
        List of surface patch names to extract.
    working_dir : str
        Working directory for any temporary files.

    Returns
    -------
    airfoil_mesh : dict
        Dictionary containing node coordinates, normals, and areas.
    """
    airfoil_mesh = os.path.join(working_dir,'Airfoil_Surface_Mesh.h5')
    r = Reader('hdf_antares')
    r['filename'] = airfoil_mesh
    mesh = r.read()
    nodes = mesh[0][0]['x'].shape[0]
    return airfoil_mesh

def load_surface_pressure_fft_data(working_dir, var:str='pressure'):
    fft_data = os.path.join(working_dir,var+'_airfoil_fft.hdf5')
    # if not os.path.exists(fft_data):
    #     raise FileNotFoundError(f"FFT data file not found: {fft_data}")
    # with h5py.File(fft_data, 'r') as h5data:
    #     fft_data = h5data['pressure_fft'][:]
    return fft_data

def estimate_computation_time(nodes, n_target_freq, n_workers=None):
    """
    Rough estimate of computation time based on mesh size and number of target frequencies.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Rough estimate: ~1e-8 seconds per surface element pair per frequency
    total_ops = nodes * nodes * n_target_freq * 10
    ops_per_second = 2e8 * n_workers  # With parallelization and numba
    
    estimated_time = total_ops / ops_per_second
    
    print(f"Estimated computation time: {estimated_time/3600:.2f} hours")
    print(f"With {nodes:,} nodes, {n_target_freq} target frequencies, {n_workers} workers")
    
    return estimated_time

def print_cpp(message, rank=0):
    """
    Print message only from the root MPI rank.
    """
    if rank == 0:
        print(message)

def compute_acoustic_surface_pressure_reference(
    p_hat,
    zeta,
    freq_all,
    normal,
    area,
    target_frequencies,
    speed_of_sound=343.0,
):
    """
    Single-process reference implementation of the Delfs surface source
    localization, matching the original Python implementation's operations
    but without multiprocessing / MPI.

    Parameters
    ----------
    p_hat : (nodes, nfreq_all) complex128
        Surface pressure in frequency domain at each node.
    zeta : (nodes, 3) float64
        Node coordinates.
    freq_all : (nfreq_all,) float64
        Full frequency array corresponding to columns of p_hat.
    normal : (nodes, 3) float64
        Surface normals at each node.
    area : (nodes,) float64
        Surface area associated with each node.
    target_frequencies : list/array of float
        Frequencies at which p_S is to be evaluated.
    speed_of_sound : float
        Speed of sound c0.

    Returns
    -------
    p_hat_s : (n_target_freq, nodes) complex128
        Acoustic surface pressure at each observer node for each target frequency.
    target_freq_indices : (n_target_freq,) int64
        Indices of the chosen frequencies in freq_all.
    """

    # Convert inputs to arrays with expected dtypes
    p_hat = np.asarray(p_hat, dtype=np.complex128)    # (nodes, nfreq_all)
    zeta = np.asarray(zeta, dtype=np.float64)         # (nodes, 3)
    normal = np.asarray(normal, dtype=np.float64)     # (nodes, 3)
    area = np.asarray(area, dtype=np.float64)         # (nodes,)
    freq_all = np.asarray(freq_all, dtype=np.float64)
    target_frequencies = np.asarray(target_frequencies, dtype=np.float64)

    # Transpose to match the original parallel implementation:
    #   p_hat:  (nodes, nfreq) -> (nfreq, nodes)
    #   zeta:   (nodes, 3)     -> (3, nodes)
    #   normal: (nodes, 3)     -> (3, nodes)
    p_hat = p_hat.T      # (nfreq_all, nodes)
    zeta = zeta.T        # (3, nodes)
    normal = normal.T    # (3, nodes)

    nfreq_all, nodes = p_hat.shape

    # Map each target frequency to the closest index in freq_all
    target_freq_indices = np.array(
        [int(np.argmin(np.abs(freq_all - f))) for f in target_frequencies],
        dtype=int,
    )
    target_freq_values = freq_all[target_freq_indices]
    n_target_freq = len(target_freq_values)

    # Extract only the needed frequencies (as in the original code)
    p_hat_target = p_hat[target_freq_indices, :]  # (n_target_freq, nodes)

    # Allocate output: (n_target_freq, nodes)
    p_hat_s = np.zeros((n_target_freq, nodes), dtype=np.complex128)

    # Main loops: frequency -> observer -> sources
    for k_idx in range(n_target_freq):
        f = float(target_freq_values[k_idx])
        k = 2.0 * math.pi * f / speed_of_sound
        p_current = p_hat_target[k_idx, :]  # (nodes,)

        for obs_idx in range(nodes):
            # Observer position x_obs: (3,)
            x_obs = zeta[:, obs_idx].astype(np.float64)

            surface_integral = 0.0 + 0.0j

            # Integrate over all source nodes
            for src_idx in range(nodes):
                if src_idx == obs_idx:
                    # Skip singular point as in original implementation
                    continue

                # r_vec = observer - source
                rx = x_obs[0] - zeta[0, src_idx]
                ry = x_obs[1] - zeta[1, src_idx]
                rz = x_obs[2] - zeta[2, src_idx]

                r2 = rx*rx + ry*ry + rz*rz
                if r2 <= 1.0e-12:
                    # Skip near-singular configurations
                    continue

                r = math.sqrt(r2)
                ex = rx / r
                ey = ry / r
                ez = rz / r

                nx = normal[0, src_idx]
                ny = normal[1, src_idx]
                nz = normal[2, src_idx]

                # e_r · n_ξ
                er_dot_n = ex*nx + ey*ny + ez*nz

                kr = k * r

                # exp(-i k r)
                exp_neg_ikr = np.exp(-1j * kr)

                # Green's factor: exp(-ikr) * (i k r + 1) * (e_r·n_xi) / r^2
                green_factor = exp_neg_ikr * (1j * kr + 1.0) * er_dot_n / r2

                # Contribution from this surface element
                surface_integral += green_factor * p_current[src_idx] * area[src_idx]

            # Apply 1/(2π) factor
            p_hat_s[k_idx, obs_idx] = surface_integral / (2.0 * math.pi)

    return p_hat_s, target_freq_indices

def compute_acoustic_surface_pressure_memory_efficient(p_hat, zeta, freq_all, normal, area,
                                                      target_frequencies,
                                                      speed_of_sound=343.0,
                                                      max_memory_gb=8.0):
    """
    Memory-efficient version that processes target frequencies one at a time.
    Suitable for very large meshes that don't fit in memory.
    
    Parameters:
    -----------
    p_hat : np.ndarray, shape (nfreq_all, nodes)
        Complete surface pressure data
    freq_all : np.ndarray, shape (nfreq_all,)
        Complete frequency vector
    target_frequencies : array-like
        Target frequencies to compute
    max_memory_gb : float
        Maximum memory usage in GB for intermediate arrays
    
    Returns:
    --------
    p_hat_s : np.ndarray, shape (len(target_frequencies), nodes)
        Acoustically active surface pressure for target frequencies
    target_freq_indices : np.ndarray
        Indices of target frequencies in the original frequency array
    """
    
    nfreq_all, nodes = p_hat.shape
    target_frequencies = np.asarray(target_frequencies)
    n_target_freq = len(target_frequencies)
    
    # Find indices of target frequencies
    target_freq_indices = []
    target_freq_values = []
    
    for target_freq in target_frequencies:
        idx = np.argmin(np.abs(freq_all - target_freq))
        closest_freq = freq_all[idx]
        
        if np.abs(closest_freq - target_freq) / target_freq > 0.01:
            print(f"Warning: Target frequency {target_freq:.1f} Hz not found.")
            print(f"Using closest frequency {closest_freq:.1f} Hz instead.")
        
        target_freq_indices.append(idx)
        target_freq_values.append(closest_freq)
    
    target_freq_indices = np.array(target_freq_indices)
    target_freq_values = np.array(target_freq_values)
    
    # Estimate memory usage
    complex_size = 16  # bytes for complex128
    memory_per_freq = nodes * nodes * complex_size / (1024**3)  # GB
    
    if memory_per_freq > max_memory_gb:
        print(f"Warning: Each frequency requires {memory_per_freq:.2f} GB of memory")
        print("Consider using a smaller subset of surface points or increasing max_memory_gb")
    
    p_hat_s = np.zeros((n_target_freq, nodes), dtype=complex)
    
    print(f"Processing {n_target_freq} target frequencies sequentially...")
    print(f"Target frequencies: {target_freq_values}")
    
    for i, (freq_idx, f) in enumerate(zip(target_freq_indices, target_freq_values)):
        print(f"Processing frequency {i+1}/{n_target_freq} ({f:.1f} Hz)")
            
        if f == 0:
            p_hat_s[i, :] = 0.0
            continue
            
        k = 2 * np.pi * f / speed_of_sound
        p_current = p_hat[freq_idx, :]
        
        # Use numba for the inner computation
        for obs_idx in range(nodes):
            observer_pos = zeta[:, obs_idx].astype(np.float64)
            
            p_hat_s[i, obs_idx] = _compute_surface_integral_numba(
                observer_pos, zeta.astype(np.float64), normal.astype(np.float64),
                p_current.astype(np.complex128), area.astype(np.float64), k
            )
    
    return p_hat_s, target_freq_indices