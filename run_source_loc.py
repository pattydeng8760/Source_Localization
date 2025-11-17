#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Source_Localization")

from source_localization_core import main

config = {
    "working_dir"       : "./",
    "mesh_dir"          : "/scratch/denggua1/CD_Ice_LES/MESH_Clean_Large",
    "mesh_file"         : "CD_Clean_Large_Combine.mesh.h5",
    "FWH_data_dir"      : "/scratch/denggua1/CD_Ice_LES/RUN_Clean_Large/FWH_Airfoil/FWH_Data_TTG",
    "var"               : "pressure",
    "reload"            : False,
    "extract_FWH"       : False,
    "freq_select"       : [1000,2000,3000],
    "source_localization": True,
    "fft_method"        : "FFT",
    "surface_patches"   : ["Airfoil"],
}

args = Namespace(**config)
main(args)