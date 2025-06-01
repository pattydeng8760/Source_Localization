from scipy.fft import fft, ifft
import scipy.signal as signal
from scipy.signal import welch, coherence
from antares import *
import os
import h5py
import numpy as np

def fft_surface_data(surface_pressure_data:str, var:str, dt:float, weight='default',nOvlp=128,nDFT=256,window='default',method='fast', reload=False):
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
