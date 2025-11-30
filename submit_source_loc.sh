#!/bin/bash

# Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=0-13:59
#SBATCH --job-name=SLoc_A10U50_low
#SBATCH --mail-user=patrickgc.deng@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-plavoie

# =-=-=-=-=-=-=-=-=-=-=-=-=-
source /project/rrg-plavoie/denggua1/pd_env.s

python ./run_source_loc.py