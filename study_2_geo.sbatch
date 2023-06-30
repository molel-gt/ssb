#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_2g
#SBATCH -N 1
#SBATCH --mem-per-cpu=128G
#SBATCH -t36:00:00
#SBATCH -qinferno
#SBATCH -ostudy-2-geo-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

#cd /storage/coda1/p-tf74/0/shared/leshinka
#. ./spack/share/spack/setup-env.sh
#spack env activate fenicsx-env
#export PATH=$PATH:/storage/coda1/p-tf74/0/shared/emolel3

module load anaconda3 && conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR

srun python3 study_2_geo.py --img_id $1 --Lz $2 --resolution $3
