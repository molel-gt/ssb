#!/bin/bash
#SBATCH -A gts-tf74
#SBATCH -Jstudy_si
#SBATCH -N 10
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks-per-node=1
#SBATCH -t6:00:00
#SBATCH -qinferno
#SBATCH -osilicon-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emolel3@gatech.edu

module purge > /dev/null 2>&1
module load anaconda3
conda activate fenicsx-env
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/
#python3 volume_reconstruction.py --grid_extents 1100-1100-11_400-1500-000 --segmentation_folder crack-volumes/$1 --phase 2 --scaling VOXEL_SCALING3
python3 conductivity.py --grid_extents 1061-1061-12_000-000-000 --root_folder mesh/study_si/$1/1061-1061-12_000-000-000/2 --scaling VOXEL_SCALING3
