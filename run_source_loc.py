#!/usr/bin/env python
import sys
from argparse import Namespace

# make sure this path points to where your module lives locally!
sys.path.insert(0, "./Source_Localization")

from source_localization_core.source_localization import main

config = {
    "working_dir"       : "./",
    "mesh_dir"          : "/project/p/plavoie/denggua1/BBDB_10AOA/MESH_ZONE_Apr24",
    "mesh_file"         : "Bombardier_10AOA_Combine_Apr24.mesh.h5",
    "FWH_data_dir"      : "/project/p/plavoie/denggua1/BBDB_10AOA/RUN_ZONE_Apr24/FWH_Airfoil/FWH_Data_TTG",
    "var"               : "pressure",
    "reload"            : False,
    "extract_FWH"       : False,
    "freq_select"       : [500, 1000, 2000, 3000],
}

args = Namespace(**config)
main(args)