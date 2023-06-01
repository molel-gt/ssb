#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jgeometry
#SBATCH -N 1
#SBATCH --mem-per-cpu=384G
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -ogeometry.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu
module load anaconda3
conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/

mpirun python3 volume_reconstruction.py --phase 1 --grid_extents 220-220-200_000-000-000