#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_4
#SBATCH -N 1
#SBATCH --mem-per-cpu=32G
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -ostudy-4.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

cd $SLURM_SUBMIT_DIR

cd /storage/coda1/p-tf74/0/shared/leshinka/ && . ./spack/share/spack/setup-env.sh && spack env activate fenicsx-env

export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/
cd $SLURM_SUBMIT_DIR

mpirun python3 study_4_geo.py --grid_info 26-26-$1 --contact_map data/current_constriction/test11.tif --eps $2
