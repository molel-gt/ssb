#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_2
#SBATCH -N 1
#SBATCH --mem-per-cpu=384G
#SBATCH -t36:00:00
#SBATCH -qinferno
#SBATCH -ostudy-2-model-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

cd /storage/coda1/p-tf74/0/shared/leshinka
. ./spack/share/spack/setup-env.sh
spack env activate fenicsx-env
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/emolel3
#module load anaconda3 && conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR

mpiexec python3 transport.py --grid_extents 470-470-100_000-000-000 --root_folder mesh/study_2/test$1/470-470-100_000-000-000 --scaling VOXEL_SCALING2
