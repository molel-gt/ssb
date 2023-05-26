#!/bin/bash
module anaconda3
conda activate fenicsx-env

cd /storage/coda1/p-tf74/0/shared/leshinka/ssb
export PYTHONPATH=$PYTHONPATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/
export PATH=$PATH:/storage/coda1/p-tf74/0/shared/leshinka/opt/:/storage/home/hcoda1/0/emolel3/.local/

for Lz in 7.25 26;
  do
  for eps in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75;
    do
      sbatch study_4.sbatch $eps 26-26-$Lz
    done
 done
