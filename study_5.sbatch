#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_5
#SBATCH -N 1
#SBATCH --mem-per-cpu=32G
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -ostudy-5.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

cd $SLURM_SUBMIT_DIR

cd /storage/coda1/p-tf74/0/shared/leshinka/ && . ./spack/share/spack/setup-env.sh
spack env activate fenicsx-env
spack load gcc@10.1.0

export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/
cd $SLURM_SUBMIT_DIR

mpirun python3 study_5_geo.py --eps $1 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos $2 --n_pieces 2
#mpirun python3 study_5.py --eps $1 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos $2 --n_pieces 2