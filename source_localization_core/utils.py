import numpy as np
import warnings
import os, builtins, time, sys
warnings.filterwarnings("ignore", category=RuntimeWarning)


def spod_parser(nt, nx, window, weight, nOvlp, nDFT, method):
    '''
    Purpose: determine data structure/shape for SPOD
        
    Parameters
    ----------
    nt     : int; number of time snapshots
    nx     : int; number of grid point * number of variable
    window : expect 1D numpy array, float; specified window function values
    weight : expect 1D numpy array; specified weight function
    nOvlp  : expect int; specified number of overlap
    nDFT   : expect int; specified number of DFT points (expect to be same as weight.shape[0])
    method : expect string; specified running mode of SPOD
    
    Returns
    -------
    weight : 1D numpy array; calculated/specified weight function
    window : 1D numpy array, float; window function values
    nOvlp  : int; calculated/specified number of overlap
    nDFT   : int; calculated/specified number of DFT points
    nBlks  : int; calculated/specified number of blocks
    '''
    # check SPOD running method
    try:
        # user specified method
        if method not in ['fast', 'lowRAM']:
            print('WARNING: user specified method not supported')
            raise ValueError
        else:
            print('Using user specified method...')
    except:        
        # default method
        method = 'lowRAM'
        print('Using default low RAM method...')
    # check specified weight function value
    try:
        # user specified weight
        nweight = weight.shape[0]
        if nweight != nx:
            print('WARNING: weight does not match with data')
            raise ValueError
        else:
            wgt_name = 'user specified'
            print('Using user specified weight...')
    except:        
        # default weight
        weight   = np.ones(nx)
        wgt_name = 'unity'
        print('Using default weight...')
    # calculate or specify window function value
    try:
        # user sepcified window
        nWinLen  = window.shape[0]
        win_name = 'user specified'
        nDFT     = nWinLen # use window shape to over-write nDFT (if specified)
        print('Using user specified nDFT from window length...')         
        print('Using user specified window function...')  
        
    except:
        # default window with specified/default nDFT
        try:
            # user specified nDFT
            nDFT  = int(nDFT)
            nDFT  = int(2**(np.floor(np.log2(nDFT)))) # round-up to 2**n type int
            print('Using user specified nDFT ...')             
                
        except:
            # default nDFT
            nDFT  = 2**(np.floor(np.log2(nt/10))) #!!! why /10 is recommended?
            nDFT  = int(nDFT)
            print('Using default nDFT...')
            
        window   = hammwin(nDFT)
        win_name = 'Hamming'
        print('Using default Hamming window...') 
    # calculate or specify nOvlp
    try:
        # user specified nOvlp
        nOvlp = int(nOvlp)
        
        # test feasibility
        if nOvlp > nDFT-1:
            print('WARNING: nOvlp too large')
            raise ValueError
        else:
            print('Using user specified nOvlp...')
    except:            
        # default nOvlp
        nOvlp = int(np.floor(nDFT/2))
        print('Using default nOvlp...')
    # calculate nBlks from nOvlp and nDFT    
    nBlks = int(np.floor((nt-nOvlp)/(nDFT-nOvlp)))
    # test feasibility
    if (nDFT < 4) or (nBlks < 2):
        raise ValueError('User sepcified window and nOvlp leads to wrong nDFT and nBlk.')
    print('--------------------------------------')
    print('SPOD parameters summary:'              )
    print('--------------------------------------')
    print('number of DFT points :{0:d}'.format(int(nDFT)))
    print('number of blocks is  :{0:d}'.format(int(nBlks)))
    print('number of overlap percent is :{0:d}'.format(nOvlp))
    print('Window function      :{0:s}'.format(win_name))
    print('Weight function      :{0:s}'.format(wgt_name))
    print('Running method       :{0:s}'.format(method))
    
    return weight, window, nOvlp, nDFT, nBlks

def hammwin(N):
    '''
    Purpose: standard Hamming window
    
    Parameters
    ----------
    N : int; window lengh

    Returns
    -------
    window : 1D numpy array; containing window function values
             n = nDFT
    '''
    
    window = np.arange(0, N)
    window = 0.54 - 0.46*np.cos(2*np.pi*window/(N-1))
    window = np.array(window)

    return window

def replace_zeros_vectorized(data):
    """ Efficiently replaces zeros in a 2D array with the nearest nonzero neighbor using forward & backward filling.
        Prints the node index and replaced value for each zero detected.
    """

    mask = data == 0  # Boolean mask of zero entries
    
    if not np.any(mask):  # If no zeros are found, return early
        print("      No zeros detected. Skipping replacement.")
        return data  

    print("      Replacing zeros with nearest neighbor values...")

    # Forward fill: Copy last nonzero value forward
    for i in range(1, data.shape[1]):
        update_mask = mask[:, i] & (data[:, i - 1] != 0)  # Only replace where needed
        if np.any(update_mask):  # Print details of replacement
            replaced_nodes = np.where(update_mask)[0]  # Indices of affected nodes
            replaced_values = data[replaced_nodes, i - 1]
            for node, value in zip(replaced_nodes, replaced_values):
                print(f"            Node {node}: Zero replaced with forward value {value}")
        data[:, i] = np.where(update_mask, data[:, i - 1], data[:, i])

    # Backward fill: Copy first nonzero value backward
    for i in range(data.shape[1] - 2, -1, -1):
        update_mask = mask[:, i] & (data[:, i + 1] != 0)
        if np.any(update_mask):  # Print details of replacement
            replaced_nodes = np.where(update_mask)[0]
            replaced_values = data[replaced_nodes, i + 1]
            for node, value in zip(replaced_nodes, replaced_values):
                print(f"Node {node}: Zero replaced with backward value {value}")
        data[:, i] = np.where(update_mask, data[:, i + 1], data[:, i])

    return data

def next_greater_power_of_2(x):
    return 2**(x-1).bit_length()

def print(text, **kwargs):
    builtins.print(text, **kwargs)
    os.fsync(sys.stdout)

def setup_logging(log_file):
    sys.stdout = open(log_file, "w", buffering=1)

def init_logging_from_cut(var, freq_select):
    freq_str = "_".join(map(str, freq_select))
    log_file = f"log_source_localization_{var}_{freq_str}.txt"
    setup_logging(log_file)

def timer(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"The total compute time is: {int(time.time() - start)} s")
        return result
    return inner
