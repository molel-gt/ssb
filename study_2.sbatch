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

source :/storage/home/hcoda1/0/emolel3/.bashrc

mamba activate fea-env

cd $SLURM_SUBMIT_DIR

python3 conductivity.py --grid_extents 470-470-"$(($2 + 1))"_000-000-000 --root_folder mesh/study_2/test$1/470-470-$2_000-000-000 --scaling VOXEL_SCALING2 --compute_distribution 1
