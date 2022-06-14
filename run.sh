#!/usr/bin/env bash

for xpart in 0 50 100 150; do
    for ypart in 0 50 100 150; do
        for zpart in 0 50 100 150; do
            ./mesher.py --img_folder Archive/electrolyte --grid_info 51-51-51 --origin $xpart,$ypart,$zpart
            mpirun -np 2 ./transport.py --grid_info 101-101-101 --origin $xpart,$ypart,$zpart
        done
    done
done
