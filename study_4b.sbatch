#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_4
#SBATCH -N 1
#SBATCH --mem-per-cpu=192G
#SBATCH --ntasks-per-node=1
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -ostudy-4-geo.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

module purge > /dev/null 2>&1
source $HOME/.bashrc

cd $SLURM_SUBMIT_DIR
mamba activate fea-env

python3 study_4_geo.py --eps $1 --grid_extents $2 --scaling CONTACT_LOSS_SCALING --study_name contact_loss_ref
