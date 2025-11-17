import numpy as np
import h5py
import os
import logging
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