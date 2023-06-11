#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "requires output directory as parameter"
    exit 1
fi


for packing_fraction in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75;
do
  sbatch packed-spheres-model.sbatch $packing_fraction
done