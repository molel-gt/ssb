#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jpckd_sphrs
#SBATCH -N 1
#SBATCH --mem-per-cpu=128G
#SBATCH -t36:00:00
#SBATCH -qinferno
#SBATCH -opacked_spheres-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu
module load anaconda3
conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/

python3 packed_spheres.py --packing_fraction $1
#python3 conductivity.py --grid_extents 2-2-2 --root_folder mesh/packed_spheres/2-2-2/$1
