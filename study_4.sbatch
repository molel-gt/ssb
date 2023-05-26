#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_4
#SBATCH -N 1
#SBATCH --mem-per-cpu=64G
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -ostudy-4-geo.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate fenicsx-env
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/
cd $SLURM_SUBMIT_DIR

mpirun python3 study_4_geo.py --eps $1 --grid_extents $2
