"""
main function for source localization method based on surface pressure from Scale-Resolving Simulations by Jan Delfs.
"""

import numpy as np
import h5py
import logging
import os
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from antares import Reader, Base, Zone, Instant, Writer
from .source_localization_block import _compute_observer_chunk
from .source_localization_utils import estimate_computation_time

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore')

def compute_acoustic_surface_pressure_parallel(p_hat, zeta, freq_all, normal, area, 
                                             target_frequencies,
                                             speed_of_sound=343.0, 
                                             n_workers=None,
                                             chunk_size=None,
                                             verbose=False):
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
    verbose : bool, optional
        Enable verbose logging. Default: False
    Returns:
    --------
    p_hat_s : np.ndarray, shape (len(target_frequencies), nodes)
        Acoustically active surface pressure for target frequencies only
    target_freq_indices : np.ndarray
        Indices of target frequencies in the original frequency array
    """
    # custom log file to track the compute status
    log_file = f"log_source_localization_{target_frequencies[0]}Hz_compute.txt"

    # --- Configure logging (overwrite file each run, flush every call) ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",   # <-- ONLY the logged message
        handlers=[
            logging.FileHandler(log_file, mode='w', delay=False),
            logging.StreamHandler()   # optional: also print to console
        ],
        force=True
    )
    
    # Force flush after every log message
    for handler in logging.getLogger().handlers:
        handler.flush = handler.stream.flush
    
    # Transpose to get the expected shapes for the algorithm
    p_hat = p_hat.T      # (nodes, nfreq) -> (nfreq, nodes)
    zeta = zeta.T        # (nodes, 3) -> (3, nodes)
    normal = normal.T    # (nodes, 3) -> (3, nodes)
    
    nfreq_all, nodes = p_hat.shape
    target_frequencies = np.asarray(target_frequencies)
    n_target_freq = len(target_frequencies)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"STARTING ACOUSTIC SURFACE PRESSURE COMPUTATION")
    logging.info(f"{'='*60}")
    logging.info(f"    Input data shape: {p_hat.shape}")
    logging.info(f"    Number of target frequencies: {n_target_freq}")
    logging.info(f"    The shape of the surface pressure data is: {p_hat.shape}")
    logging.info(f"    The number of surface elements (nodes) is: {nodes}")
    logging.info(f"    The shape of the position vector is: {zeta.shape}")
    logging.info(f"    The shape of the normal vector is: {normal.shape}")
    
    # Find indices of target frequencies in the complete frequency array
    target_freq_indices = []
    target_freq_values = []
    
    for target_freq in target_frequencies:
        # Find closest frequency in freq_all
        idx = np.argmin(np.abs(freq_all - target_freq))
        closest_freq = freq_all[idx]
        
        # Check if frequency is close enough (within 1% tolerance)
        if np.abs(closest_freq - target_freq) / target_freq > 0.01:
            logging.info(f"    The target frequency: {target_freq:.1f} Hz")
            logging.info(f"    The closest available frequency is: {closest_freq:.1f} Hz")
        
        target_freq_indices.append(idx)
        target_freq_values.append(closest_freq)
    
    target_freq_indices = np.array(target_freq_indices)
    target_freq_values = np.array(target_freq_values)
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Parallelize over observer points
    if chunk_size is None:
        chunk_size = max(100, nodes // (n_workers))  # Larger chunks for single frequency
    
    logging.info(f"    Creating observer chunks with chunk_size = {chunk_size} ")
    
    # Extract p_hat data for target frequencies only
    p_hat_target = p_hat[target_freq_indices, :]
    # Initialize output for target frequencies only
    p_hat_s = np.zeros((n_target_freq, nodes), dtype=complex)
    
    # Create chunks of observer indices
    obs_chunks = []
    for chunk_idx, i in enumerate(range(0, nodes, chunk_size)):
        end_idx = min(i + chunk_size, nodes)
        obs_indices = list(range(i, end_idx))
        # Only the first chunk / worker logs
        enable_logging = (chunk_idx == 0) if not verbose else True
        args = (obs_indices, zeta, normal, area, p_hat_target, target_freq_values, speed_of_sound, log_file, enable_logging)
        obs_chunks.append((i, args))
    
    logging.info(f"    Created {len(obs_chunks)} observer chunks")
    
    logging.info(f"{'='*60}\n")
    logging.info(f"----> Starting parallel computation with {n_workers} workers...\n")
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Use both futures and resutls to keep track of chunk starts and order within resutls
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
    
    logging.info(f"\n{'='*60}")
    logging.info(f"COMPUTATION COMPLETE!")
    logging.info(f"Output shape: {p_hat_s.shape}")
    logging.info(f"{'='*60}\n")
    
    return p_hat_s, target_freq_indices


def compute_source_localization(whole_mesh:str,input_surface:list ,airfoil_mesh:str, surface_pressure_fft_data:str, freq_select:list, verbose:bool=False):
    """
    Main function to compute source localization based on surface pressure data.
    Parameters:
    -----------
    whole_mesh : str
        Path to the whole mesh file
    input_surface : list
        List of surface names for extraction of the normals
    airfoil_mesh : str
        Path to the airfoil surface mesh file
    surface_pressure_fft_data : str
        Path to the surface pressure FFT data file
    freq_select : list
        List of target frequencies to compute
    """
    text = 'Performing Surface Source Localization'
    print(f'\n{text:.^80}\n')
    print(' Performing Source Localization for frequency: {0:s} Hz'.format(str(freq_select[0])))
    
    # Loading the surface mesh
    print('----> Loading the Airfoil Surface Mesh')
    reader = Reader('hdf_antares')
    reader['filename'] = airfoil_mesh
    base = reader.read()  # b is the Base object of the Antares API
    
    # Create zeta with correct shape (nodes, 3) then transpose to (3, nodes)
    x_coords = base[0][0]['x']
    y_coords = base[0][0]['y'] 
    z_coords = base[0][0]['z']
    
    print(f"      Coordinate array shapes: x={x_coords.shape}, y={y_coords.shape}, z={z_coords.shape}")
    
    # Create zeta with shape (nodes, 3)
    zeta = np.column_stack([x_coords, y_coords, z_coords])
    
    print(f"      zeta shape after creation: {zeta.shape} (should be (nodes, 3))")
    
    print('\n----> Loading the Airfoil Normal Vector')
    # Extracting the normal vector from the base
    shape = []                # Initializing an intermediate variable for the size of each group
    normals = []              # Initializing the normal vector
    
    # Extract normal data from whole mesh and the faces corresponding to input surfaces
    with h5py.File(whole_mesh, 'r') as h5_file:
        data = h5_file['Boundary/bnode->normal'][:]
        # Load patch labels as a list of Python strings
        labels = [label.decode('utf-8').strip() 
        for label in h5_file['Boundary']['PatchLabels'][()]]
        # Get 1-based indices for each input surface
        faces = [labels.index(name) + 1 for name in input_surface]
        
    data = np.reshape(data, (-1, 3))
    
    with h5py.File(whole_mesh, 'r') as f:
        patch_group = f["Patch"]
        num_patches = len(patch_group.keys())
    
    for i in range(1, num_patches+1):
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
    
    print(f"      normals shape after processing: {normals.shape} (should be (nodes, 3))")
    
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
        if len(input_surface) == 1:
            surface_volume = h5_file[f'/{input_surface[0]}/instants/0000/variables/VD_volume_node'][:]
        else: 
            surface_volume = h5_file['/0000/instants/0000/variables/VD_volume_node'][:]
        surface_area = surface_volume**(2/3)
        # 4) Check sizes match:
        assert surface_area.shape[0] == p_hat.shape[0], ("     The number of elements in p_hat must match the number of area entries")
    
    print("     The surface area of the airfoil is %f m^2" % np.sum(surface_area))
    
    # Debug print to check shapes
    print(f"\n----> FINAL Check - Array shapes:")
    print(f"      zeta shape: {zeta.shape} (should be (nodes, 3))")
    print(f"      normals shape: {normals.shape} (should be (nodes, 3))")
    print(f"      p_hat shape: {p_hat.shape} (should be (nodes, nfreq))")
    print(f"      surface_area shape: {surface_area.shape} (should be (nodes,))")
    
    # Verify all shapes are consistent
    nodes = zeta.shape[0]
    assert zeta.shape == (nodes, 3), f"zeta shape {zeta.shape} != ({nodes}, 3)"
    assert normals.shape == (nodes, 3), f"normals shape {normals.shape} != ({nodes}, 3)"
    assert p_hat.shape[0] == nodes, f"p_hat nodes {p_hat.shape[0]} != {nodes}"
    assert surface_area.shape[0] == nodes, f"surface_area nodes {surface_area.shape[0]} != {nodes}"
    
    print("      All array shapes are consistent!")
    print("\n----> Starting computation...")
    
    #estimate_computation_time(np.shape(p_hat)[0], len(freq_select))
    
    # Compute with parallel processing for target frequencies
    p_hat_s, target_indices = compute_acoustic_surface_pressure_parallel(
        p_hat, zeta, freq, normals, surface_area,
        target_frequencies=freq_select,
        n_workers=mp.cpu_count(),
        chunk_size=None,
        verbose=verbose
    )
    
    print("\n----> Computation complete!")
    print(f"      Output shape: {p_hat_s.shape}")
    print(f"      Target frequency indices in original array: {target_indices}")
    
    # Show results for each target frequency
    for i, (freq_val, idx) in enumerate(zip(freq_select, target_indices)):
        original_max = np.max(np.abs(p_hat[:, idx]))
        acoustic_max = np.max(np.abs(p_hat_s[i, :]))
        reduction = original_max / acoustic_max if acoustic_max > 0 else np.inf
        
        print(f"      Frequency {freq_val} Hz: Max reduction = {reduction:.1f}")
    
    print('\n----> Compelted Source Localization for frequency: {0:s} Hz'.format(str(freq_select)))
    return p_hat_s, target_indices


def output_source_localization(airfoil_mesh, p_hat_s, surface_pressure_fft_data, 
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
    print(f"      Original pressure data shape: {p_hat_orig.shape}")
    print(f"      Acoustic pressure data shape: {p_hat_s.shape}")
    print(f"      Target frequencies: {freq_select}")
    print(f"      Target frequency indices: {target_freq_indices}")
    
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




