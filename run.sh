#!/usr/bin/env bash

echo "Started on `/bin/hostname`"

for xpart in 0 50 100 150; do
    for ypart in 0 50 100 150; do
        for zpart in 0 50 100 150; do
            # nohup ./mesher.py --img_folder Archive/electrolyte --grid_info 51-51-51 --origin $xpart,$ypart,$zpart && mpirun -np 1 ./transport.py --grid_info 51-51-51 --origin $xpart,$ypart,$zpart &
            ./run.pbs $xpart $ypart $zpart
        done
    done
done
