#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_2
#SBATCH -N 10
#SBATCH --mem-per-cpu=192G
#SBATCH --ntasks-per-node=1
#SBATCH -t36:00:00
#SBATCH -qinferno
#SBATCH -ostudy-2-model-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

module purge > /dev/null 2>&1

source $HOME/.bashrc

mamba activate fea-env

cd $SLURM_SUBMIT_DIR

python3 conductivity.py --name_of_study contact_loss_lma --dimensions 470-470-$2 --mesh_folder output/contact_loss_lma/470-470-$2/$1/0.5 --scaling CONTACT_LOSS_SCALING --compute_distribution True
