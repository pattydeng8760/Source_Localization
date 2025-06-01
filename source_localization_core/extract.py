import os
import glob
import h5py
import shutil
import numpy as np
from antares import Reader
import datetime
from .utils import replace_zeros_vectorized
from antares import Reader, Writer, Family, Treatment

def extract_data(working_dir:str, data_dir:str, airfoil_mesh:str, dtype='float64', reload:bool=False):
    """
    This function copies the surface pressure data from the FWH files to the working directory to 
    avoid I/O issues and corruption of the data inside the source directory.
    
    Args:
        working_dir (str): Path to the working directory where the output will be saved.
        data_dir (str): Path to the directory containing the FWH data files.
        airfoil_mesh (str): Path to the airfoil mesh file.
        dtype (str, optional): Data type for the pressure data. Defaults to 'float64'.
        reload (bool, optional): Flag to reload the data from the working directory. Defaults to False.
    Returns:
        surface_pressure_data (str): Path to the extracted surface pressure data file.
        dt (float): Time step between the FWH data files.
    """
    
    text = 'Beginning Data Extraction'
    print(f'\n{text:.^80}\n') 
    if os.path.exists(os.path.join(working_dir,'pressure_airfoil.hdf5')) == True and reload == False:
        print('----> The pressure data is already extracted at: \n{0:s}'.format(working_dir,'pressure_airfoil.hdf5'))
        print('\n----> Loading the pressure data')
        surface_pressure_data = os.path.join(working_dir,'pressure_airfoil.hdf5')
        l=sorted(glob.glob(os.path.join(data_dir,'FWH_Airfoil_0000*.h5')))
        with h5py.File(l[-2], 'r') as f:
            t1 = f['frame_data/time'][()]
        with h5py.File(l[-1], 'r') as f:
            t2 = f['frame_data/time'][()]
        dt = t2 - t1
        dt = dt[0]
    else:  
        print('\n----> Extacting the pressure data from FWH files')
        # The directory information
        l=sorted(glob.glob(os.path.join(data_dir,'FWH_Airfoil_0000*.h5')))
        # The number of files (timesteps)
        nb_files=len(l)
        print('     The number of time steps is %d\n' %nb_files)
        # Extract the number of nodal points from the mesh
        file = airfoil_mesh
        r = Reader('hdf_antares')
        r['filename'] = os.path.join(working_dir,file)
        base  = r.read() # b is the Base object of the Antares API
        nb_points = int(base[0][0].shape[0])
        # Pre allocating space for the pressure data array of size nb_points (nodes) x nb_files (time steps)
        data = np.zeros((nb_points,nb_files), dtype=dtype)
        data_time = np.zeros((nb_files), dtype=dtype)
        print('The surface data will be extracted to a %d (nodes) x %d (timestep), array' %(nb_points,nb_files))
        # Running the loop to extract the pressure data and export as hdf5 file
        for it,filename in enumerate(l):
                print('Extracting file %s ...' %os.path.split(filename)[1]) if np.mod(it,100) == 0 else None
                with h5py.File(filename, 'r') as f:
                        # The pressure array from FWH data into 1D 
                        press = f['frame_data/pressure'][()]
                        if np.any(press == 0):
                            print("Warning: Zero values detected in pressure data.")
                        # Check if the number of pressure points match the number of nodal points
                        if len(press) != nb_points:
                                raise ValueError(f"Number of pressure points in {filename} ({len(press)}) does not match expected number of nodal points ({nb_points})")
                        # Saving the pressure data into the array
                        time = f['frame_data/time'][()]
                        data[:,it]=press.astype(dtype)
                        data_time[it] = time[0]
                print('The last file extracted is %s' %os.path.split(filename)[1]) if it == nb_files-1 else None
        # Saving the output as hdf5 file
        dt = np.mean(np.diff(data_time))
        data = replace_zeros_vectorized(data)
        mean_pressure = np.mean(data, axis=1)  # Mean across timesteps
        rms_pressure = np.sqrt(np.mean((data - mean_pressure[:, None]) ** 2, axis=1))  # RMS using mean-subtracted values
        with h5py.File('pressure_airfoil.hdf5', 'w') as f:
            f.create_dataset('pressure', data=data, dtype=dtype)
            f.create_dataset("mean_pressure", data=mean_pressure, dtype=dtype)
            f.create_dataset("rms_pressure", data=rms_pressure, dtype=dtype)
            f.attrs['dt'] = dt
            f.attrs['Extracted Date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            f.attrs['Source Path'] = data_dir
            f.attrs['Mesh Path'] = airfoil_mesh
        # Path to the surface pressure data
        surface_pressure_data = os.path.join(working_dir,'pressure_airfoil.hdf5')
    print('\n----> Statistics of the extracted pressure data')
    print('     The pressure data is saved in: \n%s' %surface_pressure_data)
    print('     The pressure data storage size is {0:2.4f} MB' .format(os.path.getsize(surface_pressure_data)/1e6))
    print('     The time step is dt = {0:5.6e}' .format(dt))
    text = 'Data Extraction Complete!'
    print(f'\n{text:.^80}\n')  
    return surface_pressure_data, dt



def extract_surface(mesh_file:str,working_dir:str,reload:bool=False):
    """ Extracting the airfoil surface mesh from the main LES mesh
    Args:
        mesh_file (str): path to the full mesh file
        working_dir (str): current working directory
    Returns:
        airfoil_mesh (str): path to the extracted airfoil surface mesh
        nodes (int): number of nodes on the airfoil surface mesh
    """
    text = 'Extracting Surface Mesh'
    print(f'\n{text:.^80}\n')
    airfoil_mesh = os.path.join(working_dir,'Airfoil_Surface_Mesh.h5')
    if os.path.exists(airfoil_mesh) == True and reload == False:
        print('----> LES Airfoil Surface Mesh already extracted at: \n{0:s}'.format(airfoil_mesh))
        # Loading the mesh
        r = Reader('hdf_antares')
        r['filename'] = airfoil_mesh
        mesh = r.read()
        mesh.show()
        nodes = mesh[0][0]['x'].shape[0]
    else:
        ## Surface Extraction of airfoil 
        print('----> Beginning Surface Extraction')
        ## Loading the mesh
        print('Loading the Main LES Mesh File')
        ## Loading the Main LES Mesh File
        r = Reader('hdf_avbp')
        r['filename'] = mesh_file
        base  = r.read() # b is the Base object of the Antares API
        airfoil_base = Family()
        airfoil_base['Airfoil_Surface'] = base.families['Patches']['Airfoil_Surface']
        airfoil_base['Airfoil_Trailing_Edge'] = base.families['Patches']['Airfoil_Trailing_Edge']
        airfoil_base['Airfoil_Side_LE'] = base.families['Patches']['Airfoil_Side_LE']
        airfoil_base['Airfoil_Side_Mid'] = base.families['Patches']['Airfoil_Side_Mid']
        airfoil_base['Airfoil_Side_TE']  = base.families['Patches']['Airfoil_Side_TE']
        base.families['SKIN'] = airfoil_base
        skin_base = base[base.families['SKIN']]
        # The data for the original mesh
        text = 'The Original Mesh Surface'
        print(f'\n{text:.^80}\n') 
        skin_base.show()
        ## Merging the extracted base objects to the same zone
        text = 'Merging the Base Objects'
        print(f'\n{text:.^80}\n')  
        myt = Treatment('merge')
        myt['base'] = skin_base
        myt['duplicates_detection'] = False
        myt['tolerance_decimals'] = 13
        # Writing the extraced mesh
        text = 'Writing the Mesh File'
        print(f'\n{text:.^80}\n') 
        merged = myt.execute()
        writer = Writer('hdf_antares')
        writer['base'] = merged
        writer['filename'] = airfoil_mesh.replace('.h5','')
        writer.dump()
        # The data for the extracted and merged mesh
        text = 'The Post Extraced Mesh Surface'
        print(f'\n{text:.^80}\n') 
        merged.show()
        nodes = merged['0000'][0].shape[0]
    print('\nThe Extracted surface mesh is saved in: \n%s' %airfoil_mesh)
    print('The number of nodes in on the airofoil surface is: %d nodes' %nodes)
    text = 'Surface Extraction Complete!'
    print(f'\n{text:.^80}\n')  
    return airfoil_mesh



def extract_files(sol_dir:str, working_dir:str,option:int=1,nskip:int=1,max_file:int=5000, reload:bool=False):
    """ Copying the surface pressure datafrom the AVBP surface FWH solution directory to the working directory, 
        Required to copy locally to avoid I/O issues and corruption of the data inside the source directory
    Args:
        sol_dir (str): path to the AVBP surface FWH solution directory
        working_dir (str): path to the working directory
        option (int, optional): extract options. 1 = sequential, 2 = skip every n files. Defaults to 1.
        nskip (int, optional): skip option. Defaults to 1.
        max_file (int, optional): maximum number of files to extract. Defaults to 5000.
    """
    text = 'Extracting the Transient Files'
    print(f'\n{text:.^80}\n')  
    output = os.path.join(working_dir,'FWH_Data')
    # An array with all the files in the source directory
    arr,arr_dest = [], []
    arr += sorted(glob.glob('{0:s}/{1:s}*.h5'.format(sol_dir,'FWH_Airfoil_')))
    if os.path.exists(output):
        print('     There are {0:d} files in the source directory'.format(len(arr)))
        arr_dest += sorted(glob.glob('{0:s}/*.h5'.format(output)))
        print('     There are {0:d} files in the destination directory'.format(len(arr_dest)))
        print('     After extraction there should be {0:d} files in the destination directory'.format(int(np.floor(len(arr))/nskip)))
    # Checking if all the files from the source directory are already extracted
    if int(len(arr_dest)) == int(np.floor(len(arr))/nskip) or (reload == False and os.path.exists(output)):
        text = 'All the files are already extracted to destination directory'
        print(f'{text}')
    else:
        # Removing the directory if exists
        if os.path.exists(output):
            shutil.rmtree(output, ignore_errors=True)
        os.makedirs(output, exist_ok = False)
        if Option == 1:
            # Only take n files
            arr = os.listdir(sol_dir)
            arr = list(arr)
            arr.sort()
            nameonly = np.array([])
            for i in range(0,np.min([int(max_file),int(len(arr))])):
                source = os.path.join(sol_dir,arr[i])
                destination = output
                shutil.copy(source,destination)
                print('Copying file %s' %os.path.split(source)[1]) if np.mod(i,100) == 0 else None
            arr2 = os.listdir(output)
            arr2 = list(arr2)
            arr2.sort()
        elif Option == 2:
            # Skip every n files
            arr = os.listdir(sol_dir)
            arr = list(arr)
            arr.sort()
            nameonly = np.array([]) 
            for i in range(0,int(len(arr)/nskip)):
                source = os.path.join(sol_dir,arr[int(i*nskip-1)])
                destination = output
                shutil.copy(source,destination)
                print('Copying file %s' %os.path.split(source)[1]) if np.mod(i,100) == 0 else None
    list_files=[]
    list_files+=sorted(glob.glob('{0:s}/*.h5'.format(output)))
    ntime = np.shape(list_files)[0]
    print('     The files are copied to: \n%s' %output)
    text = 'File Extraction Complete'
    print(f'\n{text:.^80}\n')  
    return ntime, output