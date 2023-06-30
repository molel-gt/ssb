#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_5m
#SBATCH -N 1
#SBATCH --mem-per-cpu=32G
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -ostudy-5-model.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu
module load anaconda3
conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/

srun python3 study_5.py --eps $1 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos $2 --n_pieces 2