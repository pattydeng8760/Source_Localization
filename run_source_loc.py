#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Source_Localization")

from source_localization_core import main

config = {
    "working_dir"       : "./",
    "mesh_dir"          : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/MESH_Medium_Aug25",
    "mesh_file"         : "Bombardier_10AOA_U50_Combine_Medium.mesh.h5",
    "FWH_data_dir"      : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/RUN_Medium/FWH_Airfoil/FWH_Data_TTG",
    "var"               : "pressure",
    "reload"            : False,
    "extract_FWH"       : False,
    "freq_select"       : [1000,2000,3000],
    "source_localization": True,
    "fft_method"        : "FFT",
    "surface_patches"   : ["Airfoil_Surface", "Airfoil_Trailing_Edge", "Airfoil_Side_LE", "Airfoil_Side_Mid", "Airfoil_Side_TE"],
}

args = Namespace(**config)
main(args)