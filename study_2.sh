#!/bin/bash
module load anaconda3
conda activate fenicsx-env
export PATH=/storage/coda1/p-tf74/0/shared/leshinka/opt/gmsh:$PATH

for img_id in 1 6 11 16 22;
do
  sbatch study_2.sbatch $img_id
done