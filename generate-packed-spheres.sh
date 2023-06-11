#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "requires output directory as parameter"
    exit 1
fi


for packing_fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75;
do
  cd ~/spheres
  mkdir -p $1"/$packing_fraction"
  sed -i "/double maxpf\ = */c\double maxpf = ${packing_fraction};"
  ./spheres input
  cp centers.dat $1"/$packing_fraction"
  cd -
  sbatch packed_spheres.sbatch $packing_fraction
done