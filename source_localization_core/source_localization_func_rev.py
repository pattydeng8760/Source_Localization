"""
Memory-efficient airframe noise source localization using node blocking approach
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import h5py
import gc
from antares import Reader, Writer, Base, Zone, Instant
import os

def compute_block_source_localization(args):
    """
    Compute source localization for a block of observer nodes.
    This function processes one block to minimize memory usage.
    """
    block_data, block_indices, block_num, zeta_all, normals_all, area_all, p_current, k = args
    
    n_obs_block = len(block_indices)
    n_nodes_total = zeta_all.shape[1]
    
    print(f"[Block {block_num+1}] Processing {n_obs_block} observer nodes")
    
    # Initialize output for this block
    p_S_block = np.zeros(n_obs_block, dtype=complex)
    
    # Process each observer in this block
    for i, obs_global_idx in enumerate(block_indices):
        observer_pos = zeta_all[:, obs_global_idx].astype(np.float64)
        surface_integral = 0.0 + 0.0j
        
        # Integrate over ALL source nodes (not blocked)
        for src_idx in range(n_nodes_total):
            if src_idx == obs_global_idx:
                continue  # Skip singular point
                
            # Distance calculation
            r_vec_x = observer_pos[0] - zeta_all[0, src_idx]
            r_vec_y = observer_pos[1] - zeta_all[1, src_idx]
            r_vec_z = observer_pos[2] - zeta_all[2, src_idx]
            
            r = np.sqrt(r_vec_x**2 + r_vec_y**2 + r_vec_z**2)
            
            if r < 1e-12:
                continue
                
            # Unit vector from source to observer
            e_r_x = r_vec_x / r
            e_r_y = r_vec_y / r
            e_r_z = r_vec_z / r
            
            # Dot product with source normal
            er_dot_n = (e_r_x * normals_all[0, src_idx] + 
                       e_r_y * normals_all[1, src_idx] + 
                       e_r_z * normals_all[2, src_idx])
            
            # Green's function
            kr = k * r
            exp_neg_ikr = np.exp(-1j * kr)
            green_factor = exp_neg_ikr * (1j * kr + 1) * er_dot_n / (r * r)
            
            # Add contribution
            surface_integral += green_factor * p_current[src_idx] * area_all[src_idx]
        
        # Apply 1/(2π) factor
        p_S_block[i] = surface_integral / (2 * np.pi)
        
        # Progress within block
        if (i + 1) % max(1, n_obs_block // 10) == 0:
            progress = ((i + 1) / n_obs_block) * 100
            print(f"[Block {block_num+1}] {progress:.1f}% complete ({i+1}/{n_obs_block} observers)")
    
    print(f"[Block {block_num+1}] COMPLETED")
    
    # Return results with block info for reassembly
    return p_S_block, block_indices, block_num


def compute_source_localization_blocked(whole_mesh, airfoil_mesh, surface_pressure_fft_data, 
                                       freq_select, nblocks=10, nproc=None):
    """
    Memory-efficient source localization using node blocking approach.
    
    Parameters:
    -----------
    whole_mesh : str
        Path to whole mesh file
    airfoil_mesh : str  
        Path to airfoil surface mesh
    surface_pressure_fft_data : str
        Path to pressure FFT data
    freq_select : list
        Target frequencies to compute
    nblocks : int
        Number of blocks to split observer nodes into
    nproc : int
        Number of processes (default: available CPUs)
    """
    
    print(f"\n{'Performing Memory-Efficient Surface Source Localization':=^100}\n")
    
    # 1) LOAD DATA
    print('----> Loading the Airfoil Surface Mesh')
    reader = Reader('hdf_antares')
    reader['filename'] = airfoil_mesh
    base = reader.read()
    base.show()
    
    # Create position array
    x_coords = base[0][0]['x']
    y_coords = base[0][0]['y'] 
    z_coords = base[0][0]['z']
    zeta = np.array([x_coords, y_coords, z_coords])  # Shape: (3, nodes)
    
    print('\n----> Loading the Airfoil Normal Vector')
    faces = [6, 8, 9, 10, 11]
    shape = []
    normals = []
    
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
    
    normals = np.concatenate(normals, axis=0) if normals else np.array([])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals.T  # Shape: (3, nodes)
    
    print('\n----> Loading the Fourier Transform of the Pressure Data')
    with h5py.File(surface_pressure_fft_data, 'r') as h5f:
        p_hat = h5f['pressure_fft'][:]  # Shape: (nodes, nfreq)
        freq = h5f['frequency'][:]
    
    # Surface area calculation
    print('\n----> Computing the airfoil surface area')
    a = 4.2e-4  # Triangle side length
    area_per_element = (np.sqrt(3) / 4) * a**2
    surface_area = np.full(p_hat.shape[0], area_per_element)
    
    nodes = zeta.shape[1]
    print(f"\nDataset info:")
    print(f"  Nodes: {nodes:,}")
    print(f"  Target frequencies: {len(freq_select)}")
    print(f"  Blocks: {nblocks}")
    
    # 2) DETERMINE PROCESSES
    avail = cpu_count()
    if nproc is None:
        nproc = min(avail, nblocks)
    
    print(f"  Available CPUs: {avail}")
    print(f"  Using processes: {nproc}")
    
    # 3) SPLIT OBSERVER NODES INTO BLOCKS
    node_indices = np.array_split(np.arange(nodes), nblocks)
    print(f'\n----> Partitioning observer nodes into {len(node_indices)} blocks')
    
    block_sizes = [len(indices) for indices in node_indices]
    print(f"Block sizes: min={min(block_sizes):,}, max={max(block_sizes):,}, avg={sum(block_sizes)/len(block_sizes):.0f}")
    
    # Initialize output array for all frequencies
    p_hat_s_all = np.zeros((len(freq_select), nodes), dtype=complex)
    target_indices = []
    
    # 4) PROCESS EACH FREQUENCY SEPARATELY TO SAVE MEMORY
    for f_idx, target_freq in enumerate(freq_select):
        print(f"\n{'='*60}")
        print(f"PROCESSING FREQUENCY {f_idx+1}/{len(freq_select)}: {target_freq:.1f} Hz")
        print(f"{'='*60}")
        
        # Find frequency index
        freq_idx = np.argmin(np.abs(freq - target_freq))
        closest_freq = freq[freq_idx]
        target_indices.append(freq_idx)
        
        print(f"Using frequency: {closest_freq:.1f} Hz (index {freq_idx})")
        
        # Extract pressure for this frequency only
        k = 2 * np.pi * closest_freq / 343.0
        p_current = p_hat[:, freq_idx]  # Shape: (nodes,)
        
        # Create blocks for this frequency
        blocks = []
        for block_num, indices in enumerate(node_indices):
            # Only pass essential data to minimize memory transfer
            block_data = None  # Not needed for this implementation
            args = (block_data, indices, block_num, zeta, normals, surface_area, p_current, k)
            blocks.append(args)
        
        # Free pressure data for other frequencies temporarily
        del p_current
        gc.collect()
        
        print(f"\n----> Processing {len(blocks)} blocks in parallel...")
        
        # 5) COMPUTE BLOCKS IN PARALLEL
        with Pool(nproc) as pool:
            block_results = pool.map(compute_block_source_localization, blocks)
        
        # 6) REASSEMBLE RESULTS
        print(f"----> Reassembling results for frequency {target_freq:.1f} Hz...")
        
        for p_S_block, block_indices, block_num in block_results:
            p_hat_s_all[f_idx, block_indices] = p_S_block
        
        # Clean up
        del block_results
        gc.collect()
        
        print(f"✅ Frequency {target_freq:.1f} Hz completed")
    
    print(f"\n{'='*60}")
    print(f"ALL FREQUENCIES COMPLETED")
    print(f"Final output shape: {p_hat_s_all.shape}")
    print(f"{'='*60}")
    
    return p_hat_s_all, np.array(target_indices)


def save_results_incrementally(airfoil_mesh, p_hat_s, surface_pressure_fft_data, 
                              freq_select, target_indices, output_path):
    """
    Save results incrementally to avoid memory buildup.
    """
    print('\n----> Saving results incrementally...')
    
    # Load original data
    with h5py.File(surface_pressure_fft_data, 'r') as h5f:
        p_hat_orig = h5f['pressure_fft'][:]
        freq_all = h5f['frequency'][:]
    
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
    
    # Process each frequency incrementally
    for k in range(len(freq_select)):
        target_freq = freq_select[k]
        freq_idx_in_orig = target_indices[k]
        
        print(f"Saving frequency {k+1}/{len(freq_select)}: {target_freq:.1f} Hz")
        
        # Extract data for current frequency only
        p_s_current = p_hat_s[k, :]
        p_orig_current = p_hat_orig[:, freq_idx_in_orig]
        p_f_current = (p_orig_current - p_s_current) / 2
        
        # Convert to dB
        ref_pressure = 2e-5
        p_s_abs = np.where(np.abs(p_s_current) == 0, 1e-20, np.abs(p_s_current))
        p_f_abs = np.where(np.abs(p_f_current) == 0, 1e-20, np.abs(p_f_current))
        p_orig_abs = np.where(np.abs(p_orig_current) == 0, 1e-20, np.abs(p_orig_current))
        
        data_s_dB = 10 * np.log10(p_s_abs**2 / ref_pressure**2)
        data_f_dB = 10 * np.log10(p_f_abs**2 / ref_pressure**2)
        data_orig_dB = 10 * np.log10(p_orig_abs**2 / ref_pressure**2)
        
        # Store results
        freq_int = int(target_freq)
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_dB'] = data_s_dB
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Ps_magnitude'] = p_s_abs
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_Pf_dB'] = data_f_dB
        animated_base[0][str(0)][f'frequency_{freq_int:04d}_Hz_P_orig_dB'] = data_orig_dB
        
        # Clean up frequency-specific arrays
        del p_s_current, p_orig_current, p_f_current
        del p_s_abs, p_f_abs, p_orig_abs
        del data_s_dB, data_f_dB, data_orig_dB
        gc.collect()
    
    # Write output file
    output_filename = os.path.join(output_path, 'Surface_Localization_Blocked.h5')
    w = Writer('hdf_antares')
    w['filename'] = output_filename
    w['base'] = animated_base
    w['dtype'] = 'float64'
    w.dump()
    
    del animated_base
    gc.collect()
    
    print(f'✅ Results saved to: {output_filename}')
    return output_filename
