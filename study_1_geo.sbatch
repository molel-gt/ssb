#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_1g
#SBATCH -N 1
#SBATCH --mem-per-cpu=384G
#SBATCH -t36:00:00
#SBATCH -qinferno
#SBATCH -ostudy-1-geo-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu
module load anaconda3
conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/

mpiexec python3 volume_reconstruction.py --grid_extents $1
