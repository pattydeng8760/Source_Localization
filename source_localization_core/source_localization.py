import os, glob
import numpy as np
from multiprocessing import Pool, cpu_count
from .extract import extract_data, extract_files, extract_surface
from .fft_surface import fft_surface_data
from .utils import *
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
    parser.add_argument("--reload", "-r", type=bool, default=False, help="Flag to reload the data from the working directory.")
    parser.add_argument("--option", "-o", type=int, default=1, help="extract options. 1 = sequential, 2 = skip every n files. Defaults to 1.")
    parser.add_argument("--nskip", "-n", type=int, default=1, help="Skip option. Defaults to 1.")
    parser.add_argument("--max-file", "-m", type=int, default=5000, help="Maximum number of files to extract. Defaults to 5000.")

class SourceLocalization():
    def __init__(self, args):
        self.args = args
        self.working_dir = args.working_dir
        self.mesh_file = os.path.join(args.mesh_dir, args.mesh_file)
        self.reload = args.reload
        if os.path.exists(self.mesh_file) == False:
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")
        self.FWH_data_dir = args.FWH_data_dir
        self.var = args.var

    def run(self):
        # Extracting the surface mesh file
        self.airfoil_mesh =  extract_surface(self.mesh_file, self.working_dir, reload=self.reload)
        
        # extracting the FWH data files
        if self.args.extract_FWH:
            ntime, self.FWH_data_dir = extract_files(self.FWH_data_dir, self.working_dir, self.args.option, 
                                                self.args.nskip, self.args.max_file, reload=self.reload)
        else:
            if os.path.exists(self.FWH_data_dir) == False:
                raise FileNotFoundError(f"FWH data directory not found: {self.FWH_data_dir}")
            ntime = 0
            for filename in os.listdir(self.FWH_data_dir):
                if filename.endswith('.h5'):
                    ntime += 1
        
        # Extracting the surface pressure data
        surface_pressure_data,dt = extract_data(self.working_dir, self.FWH_data_dir, self.airfoil_mesh, dtype='float64', reload=self.reload)
        
        # Performing FFT on the surface pressure data
        surface_pressure_fft_data = fft_surface_data(surface_pressure_data, self.var, dt, weight='default',nOvlp=128,nDFT=256,window='default',method='fast', reload=self.reload)

def main(args=None):
    # Parse CLI args
    if args is None:
        args = parse_arguments()
    runner = SourceLocalization(args)
    runner.run()

if __name__ == "__main__":
    main()