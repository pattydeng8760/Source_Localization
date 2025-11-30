import sys
import os
import math
import numpy as np
import time
import logging
from . import source_localization_core_cpp

def get_mpi_rank_size_from_env():
    """Get MPI rank/size from environment variables (OpenMPI style)."""
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size

def compute_acoustic_surface_pressure_parallel_cpp(p_hat, zeta, freq_all, normal, area, 
                                             target_frequencies, n_workers:int, n_threads:int,
                                             speed_of_sound:float=343.0):
    """
    Compute acoustically active surface pressure p̂_S with parallel processing for specific frequencies using C++ backend.
    C++ backend for the surface-pressure integral.
    Same logical role as compute_acoustic_surface_pressure_parallel in source_localization_func, but implemented in C++.
    
    Parameters:
    -----------
    p_hat : np.ndarray, shape (nodes, nfreq_all) - BEFORE transpose
        Complex surface pressure in frequency domain for all frequencies
    zeta : np.ndarray, shape (nodes, 3) - BEFORE transpose
        Position vectors to surface elements [x, y, z coordinates]
    freq_all : np.ndarray, shape (nfreq_all,)
        Complete frequency vector [Hz] corresponding to p_hat
    normal : np.ndarray, shape (nodes, 3) - BEFORE transpose
        Unit normal vectors for each surface element
    area : np.ndarray, shape (nodes,)
        Surface area for each surface element
    target_frequencies : array-like
        List/array of target frequencies [Hz] to compute p̂_S for
    n_workers : int
        Number of MPI workers
    n_threads : int
        Number of threads per worker (OpenMP)
    speed_of_sound : float, optional
        Speed of sound [m/s], default 343.0
    Returns:
    --------
    p_hat_s : np.ndarray, shape (len(target_frequencies), nodes)
        Acoustically active surface pressure for target frequencies only
    target_freq_indices : np.ndarray
        Indices of target frequencies in the original frequency array
    """
    rank, size = get_mpi_rank_size_from_env()
    
    # custom log file to track the compute status
    if rank == 0:
        import logging
        log_file = f"log_source_localization_{target_frequencies[0]}Hz_compute.txt"
        # --- Configure logging (overwrite file each run, flush every call) ---
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",   # <-- ONLY the logged message
            handlers=[
                logging.FileHandler(log_file, mode='w', delay=False),
                logging.StreamHandler()   #also print to console
            ],
            force=True
        )
        # Force flush after every log message
        for handler in logging.getLogger().handlers:
            handler.flush = handler.stream.flush
    
    # ensure proper dtypes and shapes
    
    p_hat = np.asarray(p_hat, dtype=np.complex128)   # (nodes, nfreq)
    zeta = np.asarray(zeta, dtype=np.float64)        # (nodes, 3)
    normal = np.asarray(normal, dtype=np.float64)    # (nodes, 3)
    area = np.asarray(area, dtype=np.float64)        # (nodes,)
    freq_all = np.asarray(freq_all, dtype=np.float64)
    target_frequencies = np.asarray(target_frequencies, dtype=np.float64)
    
    nfreq_all, nodes = p_hat.shape
    target_frequencies = np.asarray(target_frequencies)
    n_target_freq = len(target_frequencies)
    
    if rank == 0:
        logging.info(f"\n{'='*60}")
        logging.info(f"STARTING ACOUSTIC SURFACE PRESSURE COMPUTATION")
        logging.info(f"{'='*60}")
        logging.info(f"    Input data shape: {p_hat.shape}")
        logging.info(f"    Number of target frequencies: {n_target_freq}")
        logging.info(f"    The shape of the surface pressure data is: {p_hat.shape}")
        logging.info(f"    The number of surface elements (nodes) is: {nodes}")
        logging.info(f"    The shape of the position vector is: {zeta.shape}")
        logging.info(f"    The shape of the normal vector is: {normal.shape}")
        logging.info(f"    The number of MPI workers is: {n_workers}")
        logging.info(f"    The number of OMP threads per worker is: {n_threads}")
    # Find indices of target frequencies in the complete frequency array
    target_freq_indices = []
    target_freq_values = []
    
    for target_freq in target_frequencies:
        # Find closest frequency in freq_all
        idx = np.argmin(np.abs(freq_all - target_freq))
        closest_freq = freq_all[idx]
        
        # Check if frequency is close enough (within 1% tolerance)
        if np.abs(closest_freq - target_freq) / target_freq > 0.01:
            if rank == 0:
                logging.info(f"    The target frequency: {target_freq:.1f} Hz")
                logging.info(f"    The closest available frequency is: {closest_freq:.1f} Hz")
        
        target_freq_indices.append(idx)
        target_freq_values.append(closest_freq)
    
    target_freq_indices = np.array(target_freq_indices)
    target_freq_values = np.array(target_freq_values)
    
    if rank == 0:
        logging.info(f"{'='*60}\n")
        logging.info(f"----> Starting parallel computation with {n_workers} MPI processes and {n_threads} OMP threads per process...\n")
    t0 = time.time()
    # Compute the surface pressure in parallel using C++ backend
    p_hat_s, target_indices = source_localization_core_cpp.compute_acoustic_surface_pressure_mpi(
        p_hat,
        zeta,
        normal,
        area,
        freq_all,
        target_frequencies,
        speed_of_sound,
    )
    t1 = time.time()
    if rank == 0:
        logging.info(f"\n{'='*60}")
        logging.info(f"COMPUTATION COMPLETE!")
        logging.info(f"Output shape: {p_hat_s.shape}")
        logging.info(f"Time taken for computation: {t1 - t0:.2f} seconds")
        logging.info(f"{'='*60}\n")
    
    return p_hat_s, target_indices
