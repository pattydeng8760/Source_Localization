from antares import *
import os
import h5py
import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.signal import welch, coherence
from datetime import datetime
import multiprocessing
from multiprocessing import Pool, cpu_count, Value, Lock

def fft_surface_data_old(surface_pressure_data:str, var:str, dt:float, weight='default',nOvlp=128,nDFT=256,window='default',method='fast', reload=False):
    text = 'Performing FFT on surface pressure data'
    print(f'\n{text:.^80}\n')
    fft_file_path = surface_pressure_data.replace('.hdf5', '_fft.hdf5')
    if os.path.exists(fft_file_path) == True and reload == False:  
        pass
    else:
        # Load the pressure data
        with h5py.File(surface_pressure_data , 'r') as h5f:
        # Extracting the data from the file as a numpy array
            data = h5f[var][:]  # Read the flow field data
            # Check for the chape o
            if np.shape(data)[0] > np.shape(data)[1]:
                data = np.swapaxes(data, 0, 1) # Swap the axes to have the time as the first dimension
        nt = np.shape(data)[0]
        nx = np.shape(data)[1] 
        print('The surface data is %d (nodes) x %d (timestep)' %(nx,nt))
        # SPOD parser
        [weight, window, nOvlp, nDFT, nBlks] = spod_parser(nt, nx, window, weight, nOvlp, nDFT, method)
        print('Calculating temporal DFT'              )
        print('--------------------------------------')
        # calculate time-averaged result
        x_mean  = np.mean(data,axis=0)
        # obtain frequency axis
        f     = np.arange(0,int(np.ceil(nDFT/2)+1))
        f     = f/dt/nDFT
        nFreq = f.shape[0]   
        # initialize all DFT result in frequency domain
        if method == 'fast':
            Q_hat = np.zeros((nx,nFreq,nBlks),dtype = complex) # RAM demanding here       
            
        elif method == 'lowRAM':
            Q_hat = h5py.File(os.path.join(save_path,'Q_hat.h5'), 'w')
            Q_hat.create_dataset('Q_hat', shape=(nx,nFreq,nBlks), chunks=True, dtype = complex, compression="gzip")
        # initialize block data in time domain
        Q_blk = np.zeros((nDFT,nx))
        Q_blk_hat = np.zeros((nx,nFreq),dtype = complex)
        # loop over each block
        for iBlk in range(nBlks):
            # get time index for present block
            it_end   = min(iBlk*(nDFT-nOvlp)+nDFT, nt)
            it_start = it_end-nDFT
            print('block {0:d} / {1:d} ({2:d} : {3:d})'.format(iBlk+1, nBlks, it_start+1, it_end))
            # subtract time-averaged results from the block
            Q_blk = data[it_start:it_end,:] - x_mean # column-wise broadcasting
            # add window function to block
            Q_blk = Q_blk.T * window # row-wise broadcasting
            # Fourier transform on block
            Q_blk_hat = 1/np.mean(window)/nDFT*fft(Q_blk)       
            Q_blk_hat = Q_blk_hat[:,0:nFreq]           
            # correct Fourier coefficients for one-sided spectrum
            Q_blk_hat[:,1:(nFreq-1)] *= 2
            # save block result to the whole domain result 
            if method == 'fast':
                Q_hat[:,:,iBlk] = Q_blk_hat
                
            elif method == 'lowRAM':
                Q_hat['Q_hat'][:,:,iBlk] = Q_blk_hat
            # remove vars to release RAM
        del data, Q_blk, Q_blk_hat
        print('--------------------------------------')
        print('Calculating FFT'                       )
        print('--------------------------------------')
        # # initialize output vars
        # if method == 'fast':
        #     L = np.zeros((nFreq,nBlks))
        #     P = np.zeros((nFreq,nx,nBlks),dtype = complex) # RAM demanding here
            
        # elif method == 'lowRAM':
        #     h5f = h5py.File(sol_file, 'w')
        #     h5f.create_dataset('L', shape=(nFreq,nBlks), compression="gzip")
        #     h5f.create_dataset('P', shape=(nFreq,nx,nBlks), chunks=True, dtype = complex, compression="gzip")
        #     h5f.create_dataset('f', data=f, compression="gzip")     
        # Initialize data for the FFT of the pressure data
        data_fft = np.zeros((nx, nFreq), dtype=complex)
        # loop over each frequency
        for iFreq in range(nFreq):
            print('Frequency {0:d} / {1:d} (f = {2:3.3f})'.format(iFreq+1,nFreq,f[iFreq]))
            if method == 'fast':
                Q_hat_f = Q_hat[:,iFreq,:]
                data_fft[:,iFreq] = np.mean(np.abs(Q_hat_f),axis=1)
            elif method == 'lowRAM':
                Q_hat_f = Q_hat['Q_hat'][:,iFreq,:]
                data_fft[:,iFreq] = np.mean(np.abs(Q_hat_f),axis=1)
        print('After FFT, the data is %d (nodes) x %d (frequency bins)' %(nx,int(len(f))))        
        # Saving the Fourier Transform output and frequency information as an hdf5 file
        with h5py.File(fft_file_path, 'w') as hdf:
            hdf.create_dataset('pressure_fft', data=data_fft, dtype=complex)
            hdf.create_dataset('frequency', data=f, dtype=float)
    # Printing information regaring the post-fft data
    print('\nThe Fourier Transform of the pressure data is saved in {0:s}' .format(fft_file_path ))
    print('The Fourier Transform of the pressure data storage size is {0:2.4f} MB' .format(os.path.getsize(fft_file_path )/1e6))
    text = 'FFT Complete!'
    print(f'\n{text:.^80}\n')
    return fft_file_path


def compute_FFT_block(args):
    """
    Computes the FFT for a block of nodes using FFT (vectorized).
    Parameters:
    args: tuple containing (block_data, dt, nchunk, block_number)
    - block_data: 2D numpy array of shape (nt, n_nodes_in_block)
    - dt: time-step (float)
    - nchunk: number of spatial chunks (not used in FFT computation)
    - block_number: integer identifier for the current block
    Returns:
    A tuple (FFT_block, freq) where:
    - FFT_block: 2D numpy array of FFT values with shape (n_nodes_in_block, n_freq)
    - freq: 1D numpy array of frequency values (identical for all nodes in the block)
    """
    block_data, dt, nchunk, block_number = args
    if block_number % 100 == 0:
        print(f" Computing FFT for block {block_number}")
    
    n_nodes = block_data.shape[1]
    n_time = block_data.shape[0]
    
    # Apply window function to all nodes at once (optional but recommended)
    window = np.hanning(n_time)[:, np.newaxis]  # Shape: (n_time, 1)
    windowed_data = block_data * window
    
    # Compute FFT for all nodes at once
    fft_result = np.fft.fft(windowed_data, axis=0)          # FFT along time axis
    # Compute power spectral density for all nodes
    sampling_rate = 1.0 / dt

    # Take only positive frequencies (first half + DC component)
    n_freq = n_time // 2 + 1
    fft_result = 2.0 * fft_result / n_time  # Normalize and take magnitude
    fft_block = fft_result[:n_freq, :].T  # Transpose to get shape (n_nodes, n_freq)
    
    # Create frequency array
    freq = np.fft.fftfreq(n_time, dt)[:n_freq]
    
    return fft_block, freq

def fft_surface_data(surface_pressure_data: str, var: str, dt: float, reload: bool = True, block_size: int = 1000, nchunk: int = 3):
    """
    Calculates the FFT of surface pressure fluctuations for all nodes on an airfoil grid.
    The FFT is computed via Welch's method and parallelized over blocks of nodes.
    
    The output FFT array is of shape (n_nodes, n_freq_bins).
    
    Parameters:
      surface_pressure_data (str): Path to the HDF5 file containing surface pressure data.
      var (str): Variable name in the HDF5 file to extract.
      dt (float): Time-step between successive samples.
      reload (bool): If True, forces recalculation even if the output file exists.
      block_size (int): Number of nodes per block for multiprocessing.
      nchunk (int): Number of chunks for dividing the time-series data when computing FFT.
    """
    header = "Performing FFT on surface pressure data"
    print(f"\n{header:.^80}\n")
    
    fft_file_path = surface_pressure_data.replace('.hdf5', '_FFT.hdf5')
    if os.path.exists(fft_file_path) and not reload:
        print(f"{fft_file_path} already exists. Skipping FFT computation.")
    else:
        print("---->Computing FFT for surface pressure data.")
        # Load the pressure data from the HDF5 file.
        with h5py.File(surface_pressure_data, 'r') as h5f:
            data = h5f[var][:]
            # Ensure time is the first dimension.
            if np.shape(data)[0] > np.shape(data)[1]:
                data = np.swapaxes(data, 0, 1)
                
        nt, nx = data.shape
        print("The surface data is %d (nodes) x %d (timesteps)" % (nx, nt))
        
        # Determine number of blocks.
        nblocks = int(np.ceil(nx / block_size))
        num_processes = multiprocessing.cpu_count()
        print("Processing %d blocks of up to %d nodes each using %d cores." % (nblocks, block_size, num_processes))
        
        # Create list of arguments for each block.
        block_args = []
        for b in range(nblocks):
            start = b * block_size
            end = min((b + 1) * block_size, nx)
            block_data = data[:, start:end]
            block_args.append((block_data, dt, nchunk, b))
        
        # Process each block in parallel.
        with Pool(processes=num_processes) as pool:
            results = pool.map(compute_FFT_block, block_args)
        
        # Extract frequency (assumed identical across blocks) and combine FFT blocks.
        freq = results[0][1]
        # Concatenate along axis 0 so that the final shape is (n_nodes, n_freq).
        fft_all = np.concatenate([res[0] for res in results], axis=0)
        
        # Save the computed frequency and FFT data.
        print("Saving FFT data to file:", fft_file_path)
        with h5py.File(fft_file_path, 'w') as h5f_out:
            h5f_out.create_dataset('frequency', data=freq)
            h5f_out.create_dataset('pressure_fft', data=fft_all, dtype=complex)
            h5f_out.attrs['Date Computed'] = datetime.now().strftime("%Y-%m-%d")
            h5f_out.attrs['Reference Signal'] = var
            h5f_out.attrs['Reference Signal Path'] = surface_pressure_data
            h5f_out.attrs['Reference Signal Sampling Frequency'] = 1/dt
            h5f_out.attrs['Nodes'] = nx
            h5f_out.attrs['Time Steps'] = nt
    header = "FFT Computation Complete"
    print(f"\n{header:.^80}\n")
    return fft_file_path


def source_fft(output_path: str, surface_mesh: str,data:str, data_fft: str, freq_select: list = [500, 2000]):
    text = 'Performing Surface Source Localization'
    print(f'\n{text:.^80}\n')
    # Loading the surface mesh
    print('----> Loading the Airfoil Surface Mesh')
    reader = Reader('hdf_antares')
    reader['filename'] = surface_mesh
    base = reader.read()  # base is the Base object of the Antares API
    base.show()
    # Extract the coordinates of the mesh nodes
    x,y,z = base[0][0]["x"],  base[0][0]["y"],  base[0][0]["z"]
    num_nodes = len(x)
    # Loading the Surface pressure data
    print('\n----> Loading the Pressure Data: {0:s}'.format(data))
    with h5py.File(data, 'r') as h5f:
        p_mean = h5f['mean_pressure'][:]
        p_rms = h5f['rms_pressure'][:]
    # Loading the Fourier Transform data
    print('\n----> Loading the Fourier Transform of the Pressure Data: {0:s}'.format(data_fft))
    with h5py.File(data_fft, 'r') as h5f:
        p_hat = h5f['pressure_fft'][:]
        freq = h5f['frequency'][:]
        k = np.pi * 2 * freq / 340  # The wavenumber
    assert p_hat.shape[0] == num_nodes, 'The number of nodes in the pressure data and the mesh do not match'
    print('The pressure data is %d (nodes) x %d (frequency bins)' % (p_hat.shape[0], p_hat.shape[1]))
    # Create an animated base for the results
    print('\n----> Saving the surface fft data for visualization')
    animated_base = Base()
    animated_base['0'] = Zone()
    animated_base[0].shared["x"],animated_base[0].shared["y"],animated_base[0].shared["z"] = x, y, z
    animated_base[0].shared.connectivity = base[0][0].connectivity
    animated_base[0][str(0)] = Instant()
    
    # Save the filtered data for the selected frequencies
    for k in range(0, len(freq_select)):
        argi = np.argmin(np.abs(freq - freq_select[k]))
        data_orig = 10 * np.log10(np.abs(p_hat[:, argi])/ (2e-5 ** 2))
        animated_base[0][str(0)]['frequency_{0:2d}_Hz_P_orig_dB'.format(int(freq_select[k]))] = data_orig
        animated_base[0][str(0)]['frequency_{0:2d}_Hz_P_orig_Spp'.format(int(freq_select[k]))] = np.abs(p_hat[:, argi])
    # Write to output file
    animated_base[0][str(0)]['Pressure_Mean'] = p_mean
    animated_base[0][str(0)]['Pressure_RMS'] = p_rms
    w = Writer('hdf_antares')
    w['filename'] = os.path.join(output_path, 'Surface_fft')
    w['base'] = animated_base
    w['dtype'] = 'float64'
    w.dump()
    del animated_base
    print('\n----> Saving the output surface fft data as: {0:s}'.format(os.path.join(output_path, 'Surface_fft.h5')))
    text = 'Surface FFT Complete!'
    print(f'\n{text:.^80}\n')