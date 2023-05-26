#!/bin/bash

cd /storage/coda1/p-tf74/0/shared/leshinka/ && . ./spack/share/spack/setup-env.sh
spack env activate fenicsx-env
spack load gcc@10.1.0

for eps in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
#for eps in 0.01 0.05 0.1 0.2 0.3;
do
  for pos in 0.1 0.9;
  do
#    python3 study_5_geo.py --eps $eps --Lx 1 --Ly 1 --w 0.1 --h 0.2 --pos $pos --n_pieces 2 &
#    python3 study_5.py --eps $eps --Lx 1 --Ly 1 --w 0.1 --h 0.2 --pos $pos --n_pieces 2 &
  sbatch study_5.sbatch $eps $pos
  done
done