"""
Worker functions for source localization method based on surface pressure from Scale-Resolving Simulations as per AIAA-2024-3007 by Jan Delfs.
"""

import numpy as np
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import warnings
from antares import Reader, Base, Zone, Instant, Writer
import h5py
import os

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore')

@jit(nopython=True, parallel=True, fastmath=True)
def _compute_surface_integral_numba(observer_pos, zeta, source_normals, p_current, area, k):
    """
    Numba-compiled function to compute surface integral for one observer point.
    
    Corrected formulation based on user feedback:
    p̂_S(x) = (1/2π) ∫ [exp(-ikr)(ikr + 1) * (e_r · n_ξ) / r²] * p̂(ξ) dS(ξ)
    
    where:
    - x = observer position
    - ξ = source position  
    - n_ξ = source normal (normal at source location ξ)
    - e_r = (x - ξ)/|x - ξ| = unit vector from source to observer
    - exp(-ikr) with NEGATIVE sign
    """
    nodes = zeta.shape[1]
    surface_integral = 0.0 + 0.0j
    
    for src_idx in prange(nodes):
        # Distance vector: r_vec = observer - source
        r_vec_x = observer_pos[0] - zeta[0, src_idx]
        r_vec_y = observer_pos[1] - zeta[1, src_idx]
        r_vec_z = observer_pos[2] - zeta[2, src_idx]
        
        r = np.sqrt(r_vec_x**2 + r_vec_y**2 + r_vec_z**2)
        
        if r < 1e-12:  # Skip singular point
            continue
            
        # Unit vector from source to observer: e_r = (observer - source) / |observer - source|
        e_r_x = r_vec_x / r
        e_r_y = r_vec_y / r
        e_r_z = r_vec_z / r
        
        # Dot product of unit distance vector with SOURCE normal at location ξ
        er_dot_n = (e_r_x * source_normals[0, src_idx] + 
                   e_r_y * source_normals[1, src_idx] + 
                   e_r_z * source_normals[2, src_idx])
        
        # Green's function with NEGATIVE exp(-ikr) and corrected formula
        kr = k * r
        exp_neg_ikr = np.exp(-1j * kr)  # NEGATIVE sign as requested
        green_factor = exp_neg_ikr * (1j * kr + 1) * er_dot_n / (r * r)
        
        # Surface element contribution
        element_contrib = green_factor * p_current[src_idx] * area[src_idx]
        surface_integral += element_contrib
    
    return surface_integral / (2 * np.pi)


def _compute_frequency_chunk(args):
    """
    Worker function to compute p_hat_s for a chunk of frequencies.
    This enables multiprocessing parallelization.
    """
    freq_indices, freq_values, p_hat_chunk, zeta, normal, area, speed_of_sound = args
    
    nfreq_chunk = len(freq_indices)
    nodes = zeta.shape[1]
    p_S_chunk = np.zeros((nfreq_chunk, nodes), dtype=complex)
    
    # Get process ID for unique identification
    import os
    pid = os.getpid()
    
    print(f"[Worker {pid}] Starting frequency chunk: {len(freq_values)} frequencies")
    print(f"[Worker {pid}] Frequency range: {freq_values[0]:.1f} - {freq_values[-1]:.1f} Hz")
    
    for i, (f_idx, f) in enumerate(zip(freq_indices, freq_values)):
        print(f"[Worker {pid}] Processing frequency {i+1}/{nfreq_chunk}: {f:.1f} Hz")
        
        if f == 0:
            p_S_chunk[i, :] = 0.0
            continue
            
        k = 2 * np.pi * f / speed_of_sound
        p_current = p_hat_chunk[i, :]
        
        # Process each observer point with progress tracking
        progress_step = max(1, nodes // 10)  # Print every 10% of nodes
        for obs_idx in range(nodes):
            if obs_idx % progress_step == 0:
                progress = (obs_idx / nodes) * 100
                print(f"[Worker {pid}] Freq {f:.1f} Hz: {progress:.0f}% complete ({obs_idx}/{nodes} nodes)")
                
            observer_pos = zeta[:, obs_idx].astype(np.float64)
            
            # Updated function call with source normals (not observer normal)
            p_S_chunk[i, obs_idx] = _compute_surface_integral_numba(
                observer_pos, zeta.astype(np.float64), normal.astype(np.float64),
                p_current.astype(np.complex128), area.astype(np.float64), k
            )
        
        print(f"[Worker {pid}] Completed frequency {f:.1f} Hz")
    
    print(f"[Worker {pid}] Frequency chunk COMPLETED!")
    return freq_indices, p_S_chunk


def _compute_observer_chunk(args):
    """
    Alternative worker function to compute p_hat_s for a chunk of observer points.
    Can be more memory efficient for very large meshes.
    """
    obs_indices, zeta, normal, area, p_hat, freq, speed_of_sound = args
    
    nfreq = p_hat.shape[0]
    n_obs = len(obs_indices)
    p_S_chunk = np.zeros((nfreq, n_obs), dtype=complex)
    
    # Get process ID for unique identification
    import os
    pid = os.getpid()
    
    print(f"[Worker {pid}] Starting observer chunk: {n_obs} observer points")
    print(f"[Worker {pid}] Observer range: {obs_indices[0]} - {obs_indices[-1]}")
    
    for f_idx, f in enumerate(freq):
        print(f"[Worker {pid}] Processing frequency {f_idx+1}/{nfreq}: {f:.1f} Hz")
        
        if f == 0:
            p_S_chunk[f_idx, :] = 0.0
            continue
            
        k = 2 * np.pi * f / speed_of_sound
        p_current = p_hat[f_idx, :]
        
        # Process each observer point with progress tracking
        progress_step = max(1, n_obs // 10)  # Print every 10% of observers
        for i, obs_idx in enumerate(obs_indices):
            if i % progress_step == 0:
                progress = (i / n_obs) * 100
                print(f"[Worker {pid}] Freq {f:.1f} Hz: {progress:.0f}% complete ({i}/{n_obs} observers)")
                
            observer_pos = zeta[:, obs_idx].astype(np.float64)
            
            # Updated function call with source normals
            p_S_chunk[f_idx, i] = _compute_surface_integral_numba(
                observer_pos, zeta.astype(np.float64), normal.astype(np.float64),
                p_current.astype(np.complex128), area.astype(np.float64), k
            )
        
        print(f"[Worker {pid}] Completed frequency {f:.1f} Hz")
    
    print(f"[Worker {pid}] Observer chunk COMPLETED!")
    return obs_indices, p_S_chunk


def compute_acoustic_surface_pressure_parallel(p_hat, zeta, freq_all, normal, area, 
                                             target_frequencies,
                                             speed_of_sound=343.0, 
                                             method='frequency',
                                             n_workers=None,
                                             chunk_size=None):
    """
    Compute acoustically active surface pressure p̂_S with parallel processing for specific frequencies.
    
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
    speed_of_sound : float, optional
        Speed of sound [m/s], default 343.0
    method : str, optional
        Parallelization method: 'frequency' or 'observer'
        'frequency': parallelize over frequencies (better for many frequencies)
        'observer': parallelize over observer points (better for few frequencies)
    n_workers : int, optional
        Number of worker processes. Default: number of CPU cores
    chunk_size : int, optional
        Size of chunks for processing. Default: auto-calculated
        
    Returns:
    --------
    p_hat_s : np.ndarray, shape (len(target_frequencies), nodes)
        Acoustically active surface pressure for target frequencies only
    target_freq_indices : np.ndarray
        Indices of target frequencies in the original frequency array
    """
    
    # Transpose to get the expected shapes for the algorithm
    p_hat = p_hat.T      # (nodes, nfreq) -> (nfreq, nodes)
    zeta = zeta.T        # (nodes, 3) -> (3, nodes)
    normal = normal.T    # (nodes, 3) -> (3, nodes)
    
    print('The shape of the surface pressure data is:', p_hat.shape)
    print('The shape of the position vector is:', zeta.shape)
    print('The shape of the normal vector is:', normal.shape)
    
    nfreq_all, nodes = p_hat.shape
    target_frequencies = np.asarray(target_frequencies)
    n_target_freq = len(target_frequencies)
    
    print(f"\n{'='*60}")
    print(f"STARTING ACOUSTIC SURFACE PRESSURE COMPUTATION")
    print(f"{'='*60}")
    print(f"Input data shape: {p_hat.shape}")
    print(f"Number of target frequencies: {n_target_freq}")
    print(f"Target frequencies: {target_frequencies}")
    print(f"Method: {method}")
    print(f"{'='*60}\n")
    
    # Find indices of target frequencies in the complete frequency array
    target_freq_indices = []
    target_freq_values = []
    
    for target_freq in target_frequencies:
        # Find closest frequency in freq_all
        idx = np.argmin(np.abs(freq_all - target_freq))
        closest_freq = freq_all[idx]
        
        # Check if frequency is close enough (within 1% tolerance)
        if np.abs(closest_freq - target_freq) / target_freq > 0.01:
            print(f"Warning: Target frequency {target_freq:.1f} Hz not found in freq_all.")
            print(f"Using closest frequency {closest_freq:.1f} Hz instead.")
        
        target_freq_indices.append(idx)
        target_freq_values.append(closest_freq)
    
    target_freq_indices = np.array(target_freq_indices)
    target_freq_values = np.array(target_freq_values)
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"Computing with {nodes:,} surface elements and {n_target_freq} target frequencies")
    print(f"Target frequencies: {target_freq_values}")
    
    # SPECIAL CASE: Single frequency - use observer-based parallelization
    if n_target_freq == 1:
        print(f"\nSINGLE FREQUENCY DETECTED - Using observer-based parallelization")
        print(f"Parallelizing across {nodes:,} surface nodes instead of frequencies")
        
        freq_idx = target_freq_indices[0]
        f = target_freq_values[0]
        
        print(f"Processing single frequency: {f:.1f} Hz (index {freq_idx})")
        
        if f == 0:
            print("DC frequency - setting to zero")
            p_hat_s = np.zeros((1, nodes), dtype=complex)
            return p_hat_s, target_freq_indices
        
        # Set up for observer-based parallelization
        k = 2 * np.pi * f / speed_of_sound
        p_current = p_hat[freq_idx, :].reshape(1, -1)  # Shape: (1, nodes) for single frequency
        freq_values_single = np.array([f])  # Single frequency array
        
        # Calculate optimal chunk size for observer parallelization
        if chunk_size is None:
            # For single frequency, use larger chunks to reduce overhead
            # Aim for ~4x as many chunks as workers for good load balancing
            chunk_size = max(1000, nodes // (n_workers*3))
        
        print(f"Using observer-based parallelization with chunk_size = {chunk_size:,}")
        print(f"This will create ~{(nodes + chunk_size - 1) // chunk_size} chunks for {n_workers} workers")
        
        # Create chunks of observer indices
        obs_chunks = []
        for i in range(0, nodes, chunk_size):
            end_idx = min(i + chunk_size, nodes)
            obs_indices = list(range(i, end_idx))
            
            # Args: (obs_indices, zeta, normal, area, p_hat_single_freq, freq_values_single, speed_of_sound)
            args = (obs_indices, zeta, normal, area, p_current, freq_values_single, speed_of_sound)
            obs_chunks.append((i, args))
        
        print(f"Created {len(obs_chunks)} observer chunks")
        chunk_sizes = [len(chunk[1][0]) for chunk in obs_chunks]
        print(f"Chunk sizes: min={min(chunk_sizes):,}, max={max(chunk_sizes):,}, avg={sum(chunk_sizes)/len(chunk_sizes):.0f}")
        
        # Initialize output for single frequency
        p_hat_s = np.zeros((1, nodes), dtype=complex)
        
        # Process chunks in parallel
        print(f"Starting parallel computation with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = []
            futures = []
            
            # Submit all chunks
            for chunk_start, args in obs_chunks:
                future = executor.submit(_compute_observer_chunk, args)
                futures.append((chunk_start, future))
            
            print(f"Submitted {len(futures)} observer chunks to workers")
            
            # Collect results as they complete
            completed_chunks = 0
            for chunk_start, future in futures:
                obs_indices, p_S_chunk = future.result()
                results.append((chunk_start, (obs_indices, p_S_chunk)))
                
                completed_chunks += 1
                progress = (completed_chunks / len(futures)) * 100
                chunk_end = min(chunk_start + chunk_size, nodes)
                print(f"✅ Completed chunk {completed_chunks}/{len(futures)} ({progress:.1f}%) - nodes {chunk_start:,} to {chunk_end-1:,}")
        
        # Reassemble results
        print(f"Reassembling results from {len(results)} chunks...")
        for chunk_start, (obs_indices, p_S_chunk) in results:
            chunk_end = min(chunk_start + chunk_size, nodes)
            # p_S_chunk has shape (1, chunk_nodes) for single frequency
            p_hat_s[0, chunk_start:chunk_end] = p_S_chunk[0, :]
        
        print(f"✅ Single frequency observer-based computation COMPLETE!")
        print(f"Final output shape: {p_hat_s.shape}")
        
        return p_hat_s, target_freq_indices
    
    # MULTI-FREQUENCY CASE: Use parallel processing
    print(f"\n🔧 MULTI-FREQUENCY CASE - Using parallel processing")
    print(f"Using {n_workers} worker processes with '{method}' parallelization")
    
    # Initialize output for target frequencies only
    p_hat_s = np.zeros((n_target_freq, nodes), dtype=complex)
    
    if method == 'frequency':
        print(f"Using frequency-based parallelization")
        # Parallelize over frequencies
        if chunk_size is None:
            chunk_size = max(1, n_target_freq // (n_workers * 2))
        
        print(f"Creating frequency chunks with chunk_size = {chunk_size}")
        
        # Create chunks of frequency indices
        freq_chunks = []
        for i in range(0, n_target_freq, chunk_size):
            end_idx = min(i + chunk_size, n_target_freq)
            # Use target frequency indices and values
            freq_indices_chunk = target_freq_indices[i:end_idx]
            freq_values_chunk = target_freq_values[i:end_idx]
            p_hat_chunk = p_hat[freq_indices_chunk, :]
            
            args = (list(range(len(freq_indices_chunk))), freq_values_chunk, p_hat_chunk, zeta, normal, area, speed_of_sound)
            freq_chunks.append((i, args))
        
        print(f"Created {len(freq_chunks)} frequency chunks")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = []
            futures = []
            for chunk_start, args in freq_chunks:
                future = executor.submit(_compute_frequency_chunk, args)
                futures.append((chunk_start, future))
            
            for chunk_start, future in futures:
                freq_indices, p_S_chunk = future.result()
                results.append((chunk_start, (freq_indices, p_S_chunk)))
        
        # Reassemble results
        for chunk_start, (freq_indices, p_S_chunk) in results:
            chunk_end = min(chunk_start + chunk_size, n_target_freq)
            p_hat_s[chunk_start:chunk_end, :] = p_S_chunk
                
    elif method == 'observer':
        print(f"Using observer-based parallelization")
        # Parallelize over observer points
        if chunk_size is None:
            chunk_size = max(100, nodes // (n_workers * 2))  # Larger chunks for single frequency
        
        print(f"Creating observer chunks with chunk_size = {chunk_size}")
        
        # Extract p_hat data for target frequencies only
        p_hat_target = p_hat[target_freq_indices, :]
        
        # Create chunks of observer indices
        obs_chunks = []
        for i in range(0, nodes, chunk_size):
            end_idx = min(i + chunk_size, nodes)
            obs_indices = list(range(i, end_idx))
            
            args = (obs_indices, zeta, normal, area, p_hat_target, target_freq_values, speed_of_sound)
            obs_chunks.append((i, args))
        
        print(f"Created {len(obs_chunks)} observer chunks")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = []
            futures = []
            for chunk_start, args in obs_chunks:
                future = executor.submit(_compute_observer_chunk, args)
                futures.append((chunk_start, future))
            
            for chunk_start, future in futures:
                obs_indices, p_S_chunk = future.result()
                results.append((chunk_start, (obs_indices, p_S_chunk)))
        
        # Reassemble results
        for chunk_start, (obs_indices, p_S_chunk) in results:
            chunk_end = min(chunk_start + chunk_size, nodes)
            p_hat_s[:, chunk_start:chunk_end] = p_S_chunk
    
    else:
        raise ValueError("Method must be 'frequency' or 'observer'")
    
    print(f"\n{'='*60}")
    print(f"COMPUTATION COMPLETE!")
    print(f"Output shape: {p_hat_s.shape}")
    print(f"{'='*60}\n")
    
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


def compute_source_localization(whole_mesh, airfoil_mesh, surface_pressure_fft_data, freq_select):
    """
    Run the source localization method.
    This is a placeholder for any additional logic that might be needed.
    """
    text = 'Performing Surface Source Localization'
    print(f'\n{text:.^80}\n')
    
    # Loading the surface mesh
    print('----> Loading the Airfoil Surface Mesh')
    reader = Reader('hdf_antares')
    reader['filename'] = airfoil_mesh
    base = reader.read()  # b is the Base object of the Antares API
    base.show()
    
    # Create zeta with correct shape (nodes, 3) then transpose to (3, nodes)
    x_coords = base[0][0]['x']
    y_coords = base[0][0]['y'] 
    z_coords = base[0][0]['z']
    
    print(f"Coordinate array shapes: x={x_coords.shape}, y={y_coords.shape}, z={z_coords.shape}")
    
    # Create zeta with shape (nodes, 3)
    zeta = np.column_stack([x_coords, y_coords, z_coords])
    
    print(f"zeta shape after creation: {zeta.shape} (should be (nodes, 3))")
    
    print('\n----> Loading the Airfoil Normal Vector')
    # Extracting the normal vector from the base
    faces = [6, 8, 9, 10, 11]  # The group index for the faces of the mesh on the airfoil
    shape = []                # Initializing an intermediate variable for the size of each group
    normals = []              # Initializing the normal vector
    
    with h5py.File(whole_mesh, 'r') as h5_file:
        data = h5_file['Boundary/bnode->normal'][:]
    data = np.reshape(data, (-1, 3))
    
    for i in range(1, 12):
        group = 'Patch/' + str(i) + '/Coordinates/x'
        with h5py.File(whole_mesh, 'r') as h5_file:
            shape.append(np.shape(h5_file[group][:])[0])
        
        if i in faces:
            start_idx = np.cumsum(shape[:-1])[-1] if len(shape) > 1 else 0
            end_idx = np.cumsum(shape)[-1]
            normals.append(data[start_idx:end_idx])
    
    # Convert normals to a numpy array after the loop
    normals = np.concatenate(normals, axis=0) if normals else np.array([])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    print(f"normals shape after processing: {normals.shape} (should be (nodes, 3))")
    
    # Loading the Fourier Transform data
    print('\n----> Loading the Fourier Transform of the Pressure Data: {0:s}'.format(surface_pressure_fft_data))
    with h5py.File(surface_pressure_fft_data, 'r') as h5f:
        p_hat = h5f['pressure_fft'][:]
        freq = h5f['frequency'][:]
        k = np.pi * 2 * freq / 340     # The wavenumber
        assert (np.shape(p_hat)[0] == np.shape(normals)[0]), 'The number of nodes in the pressure data and the normal vector do not match'
    
    print('     The pressure data is %d (nodes) x %d (frequency bins)' % (np.shape(p_hat)[0], np.shape(p_hat)[1]))
    
    # The surface area from the surface mesh
    print('\n----> Computing the airfoil surface area')
    with h5py.File(airfoil_mesh, 'r') as h5_file:
        # 1) Define side-length in meters:
        #a = 4.2e-4  # 4.2 mm = 4.2e-3 m
        # 2) Compute area of one equilateral triangle:
        #area_per_element = (np.sqrt(3) / 4) * a**2   # [m^2]
        # 3) Build a 1D array so that each triangle i has the same area:
        #    We assume `p_hat` is already defined (length = number of elements).
        #num_elems = p_hat.shape[0]
        #surface_area = np.full(num_elems, area_per_element)
        surface_volume = h5_file['/0000/instants/0000/variables/VD_volume_node'][:]
        surface_area = surface_volume**(2/3)
        # 4) Check sizes match:
        assert surface_area.shape[0] == p_hat.shape[0], (
            "     The number of elements in p_hat must match the number of area entries"
        )
    
    print("     The surface area of the airfoil is %f m^2" % np.sum(surface_area))
    
    # Debug print to check shapes
    print(f"\nFINAL Debug - Array shapes:")
    print(f"  zeta shape: {zeta.shape} (should be (nodes, 3))")
    print(f"  normals shape: {normals.shape} (should be (nodes, 3))")
    print(f"  p_hat shape: {p_hat.shape} (should be (nodes, nfreq))")
    print(f"  surface_area shape: {surface_area.shape} (should be (nodes,))")
    
    # Verify all shapes are consistent
    nodes = zeta.shape[0]
    assert zeta.shape == (nodes, 3), f"zeta shape {zeta.shape} != ({nodes}, 3)"
    assert normals.shape == (nodes, 3), f"normals shape {normals.shape} != ({nodes}, 3)"
    assert p_hat.shape[0] == nodes, f"p_hat nodes {p_hat.shape[0]} != {nodes}"
    assert surface_area.shape[0] == nodes, f"surface_area nodes {surface_area.shape[0]} != {nodes}"
    
    print("✅ All array shapes are consistent!")
    print("Starting computation...")
    
    estimate_computation_time(np.shape(p_hat)[0], len(freq_select))
    
    # Choose method based on dataset characteristics
    if len(freq_select) > np.shape(p_hat)[0] // 1000:
        method = 'frequency'
    else:
        method = 'observer'
    
    # Compute with parallel processing for target frequencies only
    p_hat_s, target_indices = compute_acoustic_surface_pressure_parallel(
        p_hat, zeta, freq, normals, surface_area,
        target_frequencies=freq_select,
        method=method,
        n_workers=mp.cpu_count(),  # Adjust based on your system
        chunk_size=None
    )
    
    print("Computation complete!")
    print(f"Output shape: {p_hat_s.shape}")
    print(f"Target frequency indices in original array: {target_indices}")
    
    # Show results for each target frequency
    for i, (freq_val, idx) in enumerate(zip(freq_select, target_indices)):
        original_max = np.max(np.abs(p_hat[:, idx]))
        acoustic_max = np.max(np.abs(p_hat_s[i, :]))
        reduction = original_max / acoustic_max if acoustic_max > 0 else np.inf
        
        print(f"Frequency {freq_val} Hz: Max reduction = {reduction:.1f}")
    
    return p_hat_s, target_indices


def output_source_localization_corrected(airfoil_mesh, p_hat_s, surface_pressure_fft_data, 
                                        freq_select, target_freq_indices, output_path):
    """
    Corrected function to output source localization results.
    
    Parameters:
    -----------
    airfoil_mesh : str
        Path to airfoil mesh file
    p_hat_s : np.ndarray, shape (n_target_freq, nodes)
        Acoustically active surface pressure for target frequencies
    surface_pressure_fft_data : str
        Path to original pressure FFT data
    freq_select : array-like
        Target frequencies that were computed
    target_freq_indices : np.ndarray
        Indices of target frequencies in original frequency array
    output_path : str
        Output directory path
    """
    
    # Load original data
    with h5py.File(surface_pressure_fft_data, 'r') as h5f:
        p_hat_orig = h5f['pressure_fft'][:]  # Shape: (nodes, nfreq_all)
        freq_all = h5f['frequency'][:]       # Shape: (nfreq_all,)
    
    print('\n----> Saving the source localization data')
    print(f"Original pressure data shape: {p_hat_orig.shape}")
    print(f"Acoustic pressure data shape: {p_hat_s.shape}")
    print(f"Target frequencies: {freq_select}")
    print(f"Target frequency indices: {target_freq_indices}")
    
    # Load mesh geometry
    reader = Reader('hdf_antares')
    reader['filename'] = airfoil_mesh
    base = reader.read()
    
    # Create output base structure
    animated_base = Base()
    animated_base['0'] = Zone()
    animated_base[0].shared["x"] = base[0][0]["x"]
    animated_base[0].shared["y"] = base[0][0]["y"]
    animated_base[0].shared["z"] = base[0][0]["z"]
    animated_base[0].shared.connectivity = base[0][0].connectivity
    animated_base[0][str(0)] = Instant()
    
    # Process each target frequency
    for k in range(len(freq_select)):
        target_freq = freq_select[k]
        freq_idx_in_orig = target_freq_indices[k]
        
        print(f"Processing frequency {k+1}/{len(freq_select)}: {target_freq:.1f} Hz")
        
        # Extract data for current frequency
        # CORRECTED: p_hat_s has shape (n_target_freq, nodes), so use [k, :]
        p_s_current = p_hat_s[k, :]  # Shape: (nodes,)
        
        # CORRECTED: Extract original pressure for the corresponding frequency
        p_orig_current = p_hat_orig[:, freq_idx_in_orig]  # Shape: (nodes,)
        
        # Compute free field pressure: p_f = p - p_s (from Equation 4 in paper)
        # p̂_S := p̂(x) - 2p̂_f(x), so p̂_f = (p̂ - p̂_S) / 2
        p_f_current = (p_orig_current - p_s_current) / 2  # Shape: (nodes,)
        
        # Convert to decibels (SPL with reference 20 μPa)
        ref_pressure = 2e-5  # 20 μPa reference
        
        # Ensure no zero values for log calculation
        p_s_abs = np.abs(p_s_current)
        p_f_abs = np.abs(p_f_current)
        p_orig_abs = np.abs(p_orig_current)
        
        # Replace zeros with very small values to avoid log(0)
        p_s_abs = np.where(p_s_abs == 0, 1e-20, p_s_abs)
        p_f_abs = np.where(p_f_abs == 0, 1e-20, p_f_abs)
        p_orig_abs = np.where(p_orig_abs == 0, 1e-20, p_orig_abs)
        
        # Calculate SPL in dB
        data_s_dB = 10 * np.log10(p_s_abs**2 / ref_pressure**2)
        data_f_dB = 10 * np.log10(p_f_abs**2 / ref_pressure**2)
        data_orig_dB = 10 * np.log10(p_orig_abs**2 / ref_pressure**2)
        
        # Store results in animated_base
        freq_int = int(target_freq)
        
        # Acoustically active pressure (p̂_S)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_dB'] = data_s_dB
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_magnitude'] = p_s_abs
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_real'] = np.real(p_s_current)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_imag'] = np.imag(p_s_current)
        
        # Free field pressure (p̂_f) 
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Pf_dB'] = data_f_dB
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Pf_magnitude'] = p_f_abs
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Pf_real'] = np.real(p_f_current)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Pf_imag'] = np.imag(p_f_current)
        
        # Original pressure (p̂)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_P_orig_dB'] = data_orig_dB
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_P_orig_magnitude'] = p_orig_abs
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_P_orig_real'] = np.real(p_orig_current)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_P_orig_imag'] = np.imag(p_orig_current)
        
        # Additional useful quantities
        # Reduction ratio: how much the acoustic pressure is reduced compared to original
        reduction_ratio = p_orig_abs / (p_s_abs + 1e-20)  # Avoid division by zero
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_reduction_ratio'] = reduction_ratio
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_reduction_dB'] = 20 * np.log10(reduction_ratio)
        
        # Phase information
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_phase'] = np.angle(p_s_current)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_P_orig_phase'] = np.angle(p_orig_current)
    
    # Write output file
    output_freq = "_".join(map(str, freq_select))
    output_name = f"Surface_Localization_{output_freq}"
    output_filename = os.path.join(output_path, output_name)
    w = Writer('hdf_antares')
    w['filename'] = output_filename
    w['base'] = animated_base
    w['dtype'] = 'float64'
    w.dump()
    
    del animated_base
    
    print(f'\n----> Saved source localization data to: {output_filename}')
    
    # Print summary statistics
    print('\n----> Summary Statistics:')
    for k, freq in enumerate(freq_select):
        p_s_max = np.max(np.abs(p_hat_s[k, :]))
        p_orig_max = np.max(np.abs(p_hat_orig[:, target_freq_indices[k]]))
        reduction = p_orig_max / p_s_max if p_s_max > 0 else np.inf
        
        print(f"  Frequency {freq:.1f} Hz:")
        print(f"    Max original pressure: {p_orig_max:.2e}")
        print(f"    Max acoustic pressure: {p_s_max:.2e}") 
        print(f"    Max reduction ratio: {reduction:.1f} ({20*np.log10(reduction):.1f} dB)")
    
    text = 'Source Localization Complete!'
    print(f'\n{text:.^80}\n')

# def output_source_localization(airfoil_mesh, p_hat_s, surface_pressure_fft_data, freq_select, output_path):
#     with h5py.File(surface_pressure_fft_data,'r') as h5f:
#         p_hat = h5f['pressure_fft'][:]
#         freq = h5f['frequency'][:]
    
#     print('\n----> Saving the source localization data')
#     reader = Reader('hdf_antares')
#     reader['filename'] = airfoil_mesh
#     base  = reader.read() # b is the Base object of the Antares API
#     animated_base = Base()
#     animated_base['0'] = Zone()
#     animated_base[0].shared["x"] = base[0][0]["x"]
#     animated_base[0].shared["y"] = base[0][0]["y"]
#     animated_base[0].shared["z"] = base[0][0]["z"]
#     animated_base[0].shared.connectivity = base[0][0].connectivity
#     animated_base[0][str(0)] = Instant()
    
#     for k in range(0, len(freq_select)):
#         freq_indx = np.array([np.argmin(np.abs(freq - freq_select[k]))])
#         data_s = 10 * np.log10(np.abs(p_hat_s[:, k])**2 / (2e-5**2))
#         data = 10 * np.log10(np.abs(p_hat[:, freq_indx])**2 / (2e-5**2))
#         p_hat_f = 1/2*(p_hat[:, freq_indx] - p_hat_s[:, k])
#         data_f = 10 * np.log10(np.abs(p_hat_f)**2 / (2e-5**2))
        
#         animated_base[0][str(0)]['frequency_{0:2d}_Hz_Ps_dB'.format(int(freq_select[k]))] = data_s
#         animated_base[0][str(0)]['frequency_{0:2d}_Hz_Ps_Spp'.format(int(freq_select[k]))] = np.abs(p_hat_s[:, k])
#         animated_base[0][str(0)]['frequency_{0:2d}_Hz_Pf_dB'.format(int(freq_select[k]))] = data_f
#         animated_base[0][str(0)]['frequency_{0:2d}_Hz_Pf_Spp'.format(int(freq_select[k]))] = np.abs(p_hat_f[:, k])
#         animated_base[0][str(0)]['frequency_{0:2d}_Hz_P_orig_dB'.format(int(freq_select[k]))] = data
#         animated_base[0][str(0)]['frequency_{0:2d}_Hz_P_orig_Spp'.format(int(freq_select[k]))] = np.abs(p_hat[:, freq_indx])
#     w = Writer('hdf_antares')
#     w['filename'] = os.path.join(output_path,'Surface_Localization')
#     # w['coordinates'] = ['x','y','z']
#     w['base'] = animated_base
#     w['dtype'] = 'float64'
#     w.dump()
#     del animated_base
#     print('\n----> Saving the output source localization data as: {0:s}'.format(os.path.join(output_path, 'Surface_Localization.h5')))
    
#     text = 'Source Localization Complete!'
#     print('The source localization is complete')
#     print(f'\n{text:.^80}\n')



