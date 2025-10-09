#!/bin/bash

if [[ "$#" -ne 1 ]]; then
    echo 'Usage: ./hdg-tcd.sh procs'
    exit -1
fi

echo 'Running for stability parameter '$1
area_frac=6.30
gamma=0.01
for kr in 0.01 1 100; do
    for Wa_p in 1e-3 1.0 1e3; do
        echo 'Processing kr: '$kr ' and Wa_p: '$Wa_p
        rm -r ~/.cache/fenics
        mpiexec -n $1 python3 prog_copy.py -m ../output/cylinders/20-20-80/0.0125/$area_frac -kr $kr -gamma $gamma -L_C 80e-6 -Wa_p $Wa_p
    done
done
