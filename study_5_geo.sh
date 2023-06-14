#!/bin/bash
module load anaconda3
conda activate fenicsx-env
export PATH=/storage/coda1/p-tf74/0/shared/leshinka/opt/gmsh:$PATH

for eps in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
do
  for pos in 0.5 0.8;
  do
  sbatch study_5_geo.sbatch $eps $pos
  done
done