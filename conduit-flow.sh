#!/bin/bash

resolution=0.01
Lc=10
for h_L in 0.1
do
    for w_L in 0.49 0.45 0.4 0.35 0.3 0.25 0.2 0.15 0.1 0.05 0.01
    do
        python3 conduit_flow_geo.py --mesh_folder output/conduit_flow --Lc $Lc --h_over_L $h_L --w_over_L $w_L --resolution_lc $resolution
        mpiexec -n 3 python3 conduit_flow.py --mesh_folder output/conduit_flow --Lc $Lc --h_over_L $h_L --w_over_L $w_L
    done
done
