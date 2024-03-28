#!/usr/bin/env bash

rm -r ~/.cache/fenics/

python3 lmb_3d_cc_2d_geo.py --name_of_study lithium_metal_3d_cc_2d  --resolution 0.2

mpiexec python3 lmb_3d_cc_new.py --dimensions 150-40-0 --mesh_folder output/subdomains_dg --Wa 1e3
mpiexec python3 lmb_3d_cc_new.py --dimensions 150-40-0 --mesh_folder output/lithium_metal_3d_cc_2d/150-40-0/20-55-20/2e-07/ --Wa 1e3