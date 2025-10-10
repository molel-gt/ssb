#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo 'Usage: ./hdg-tcd.sh procs mesh_folder'
    exit -1
fi

gamma=0.001
echo 'Running for stability parameter: '$gamma
for kr in 0.01 1 100; do
    for Wa_p in 1e-3 1 1e3; do
        echo 'Processing kr: '$kr ' and Wa_p: '$Wa_p
        rm -r ~/.cache/fenics
        mpiexec -n $1 python3 prog_copy.py -m $2 -kr $kr -gamma $gamma -L_C 80e-6 -Wa_p $Wa_p
    done
done
