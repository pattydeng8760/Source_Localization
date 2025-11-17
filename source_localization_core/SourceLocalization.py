import os, glob
import numpy as np
from multiprocessing import Pool, cpu_count
from .extract import extract_data, extract_files, extract_surface
from .fft_surface import dft_surface_data,fft_surface_data, source_fft
from .utils import *
from .source_localization_func import *
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute Source Localization method based on surface pressure from Scale-Resolving Simulations as per AIAA-2024-3007 by Jan Delfs."
    )
    
    parser.add_argument("--working-dir", "-w", type=str, default="./",help="Working directory to place the output.")
    parser.add_argument("--mesh-dir", "-md", type=str, default="./", help="Directory containing the mesh file.")
    parser.add_argument("--mesh-file", "-mf", type=str, default="mesh.h5", help="Name of the mesh file inside the mesh direcotry.")
    parser.add_argument("--FWH-data-dir", "-fwh", type=str, default="./FWH_data", help="Directory containing the FWH data files.")
    parser.add_argument("--extract-FWH", "-ef", type=bool, default=False, help="Flag to copy the FWH data files to the working directory.")
    parser.add_argument("--var", "-v", type=str, default="pressure", help="Variable to be used for source localization. Default is pressure.")
    parser.add_argument("--reload", "-r", type=bool, default=False, help="Flag to reload the data and mesh from the working directory.")
    parser.add_argument("--option", "-o", type=int, default=1, help="extract options. 1 = sequential, 2 = skip every n files. Defaults to 1.")
    parser.add_argument("--nskip", "-n", type=int, default=1, help="Skip option. Defaults to 1.")
    parser.add_argument("--max-file", "-m", type=int, default=5000, help="Maximum number of files to extract. Defaults to 5000.")
    parser.add_argument("--freq-select", "-fs", type=list, default=[1000,1500,2000,2500,3000], help="Frequency selection for source localization. Defaults to 1000 Hz.")
    parser.add_argument("--source-localization", "-sl", type=bool, default=False, help="Flag to perform source localization. Defaults to False.")
    parser.add_argument("--fft-method", "-fm", type=str, default="DFT", choices=['FFT', 'DFT'],help="Method for FFT computation. Defaults to 'DFT'.")
    parser.add_argument("--surface-patches", "-sp", type=list, default=['Airfoil_Surface'], help="List of surface patches to include in the analysis. Defaults to all patches.")

class SourceLocalization():
    def __init__(self, args):
        text = "Beginning Source Localization"
        print(f'\n{text:=^100}\n')
        self.args = args
        self.working_dir = args.working_dir
        self.mesh_file = os.path.join(args.mesh_dir, args.mesh_file)
        self.reload = args.reload
        self.fft_method = args.fft_method
        self.freq_select = args.freq_select
        if os.path.exists(self.mesh_file) == False:
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")
        self.FWH_data_dir = args.FWH_data_dir
        self.var = args.var
        self.surface_patches = args.surface_patches
        self._print_args()
    
    def _print_args(self):
        """Print all CLI / input arguments stored in self.args."""
        text = " Source Localization Input Arguments "
        print(f'\n{text:.^80}\n')
        for name, value in vars(self.args).items():
            print(f"{name:20s}: {value}")
        text = " End of Input Arguments "
        print(f'\n{text:.^80}\n')

    def run(self):
        # Extracting the surface mesh file
        self.airfoil_mesh =  extract_surface(self.mesh_file, self.surface_patches, self.working_dir, reload=self.reload)
        
        # extracting the FWH data files
        if self.args.extract_FWH:
            ntime, self.FWH_data_dir = extract_files(self.FWH_data_dir, self.working_dir, self.args.option, 
                                                self.args.nskip, self.args.max_file, reload=True)
        else:
            if os.path.exists(self.FWH_data_dir) == False:
                raise FileNotFoundError(f"FWH data directory not found: {self.FWH_data_dir}")
            ntime = 0
            for filename in os.listdir(self.FWH_data_dir):
                if filename.endswith('.h5'):
                    ntime += 1
        
        # Extracting the surface pressure data
        self.surface_pressure_data,self.dt = extract_data(self.working_dir, self.FWH_data_dir, self.airfoil_mesh, dtype='float64', reload=self.reload)
        
        # Performing FFT on the surface pressure data
        if self.fft_method == 'DFT':
            self.surface_pressure_fft_data = dft_surface_data(self.surface_pressure_data, self.var, self.dt, weight='default',nOvlp=256,nDFT=512,window='default',method='fast', reload=True)
        else:
            self.surface_pressure_fft_data = fft_surface_data(self.surface_pressure_data, self.var, self.dt, reload=True)
        source_fft(self.working_dir, self.airfoil_mesh, self.surface_pressure_data, self.surface_pressure_fft_data, self.freq_select)
    
    def run_source_localization(self):
        for idx, freq in enumerate(self.freq_select):
            # Performing Source Localization for selected frequencies with iterateive output
            p_hat_s, target_indices = compute_source_localization(self.mesh_file, self.surface_patches, 
                                        self.airfoil_mesh, self.surface_pressure_fft_data, [freq])
            # Outputting the Source Localization results
            output_source_localization(self.airfoil_mesh, p_hat_s, self.surface_pressure_fft_data, 
                                        [freq], target_indices, self.working_dir)
        text = "Source Localization Complete"
        print(f'\n{text:=^100}\n')