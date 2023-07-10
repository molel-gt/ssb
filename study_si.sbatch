#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_5g
#SBATCH -N 1
#SBATCH --mem-per-cpu=32G
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -osilicon-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu
module load anaconda3
conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/

python3 conductivity.py --grid_extents 1100-1100-11_400-1500-000 --root_folder mesh/study_si/$1/1101-1101-12_400-1500-000 --scaling VOXEL_SCALING3