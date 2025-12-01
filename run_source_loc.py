#!/usr/bin/env python3
import sys
import os
from argparse import Namespace

# ---------------------------------------------------------------------
# Ensure package import works correctly
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the package CLI entrypoint
from source_localization_core.__main__ import main


# ---------------------------------------------------------------------
# Manual configuration to override CLI arguments
# ---------------------------------------------------------------------
config = {
    "working_dir"        : "./",
    "mesh_dir"           : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/MESH_Medium_Aug25",
    "mesh_file"          : "Bombardier_10AOA_U50_Combine_Medium.mesh.h5",
    "FWH_data_dir"       : "/scratch/denggua1/Bombardier_LES/B_10AOA_U50_LES/RUN_Medium/FWH_Airfoil/FWH_Data_TTG",
    "var"                : "pressure",
    "reload"             : False,
    "extract_FWH"        : False,
    "option"             : 2,
    "nskip"              : 1,
    "min_file"           : 5000,
    "max_file"           : 10000,
    "freq_select"        : [800],  # list required for argparse
    "source_localization": True,
    "fft_method"         : "FFT",
    "surface_patches"    : [
        "Airfoil_Surface",
        "Airfoil_Trailing_Edge",
        "Airfoil_Side_LE",
        "Airfoil_Side_Mid",
        "Airfoil_Side_TE",
    ],
    "compute_method"     : "python",   # or "C"
    "num_ranks"          : 1,
    "num_threads"        : 1,
    "verbose"            : True,
}

args = Namespace(**config)

# ---------------------------------------------------------------------
# Run using exact CLI entrypoint logic from `__main__.py`
# ---------------------------------------------------------------------
main(args)