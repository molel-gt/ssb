#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_4
#SBATCH -N 1
#SBATCH --mem-per-cpu=192G
#SBATCH --ntasks-per-node=1
#SBATCH -t96:00:00
#SBATCH -qinferno
#SBATCH -ostudy-4b-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

module purge > /dev/null 2>&1
source $HOME/.bashrc

cd $SLURM_SUBMIT_DIR
mamba activate fea-env

python3 study_4_geo.py --eps $1 --dimensions $2 --max_resolution 0.75 --scaling CONTACT_LOSS_SCALING --name_of_study contact_loss_ref
