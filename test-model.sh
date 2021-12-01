#!/usr/bin/env bash

python3 sphere_in_box.py

./create_xdmf_meshfiles.py --input_meshfile mesh/100_0_100/sphere-in-box.msh

mpirun -n 2 python3 ion_transport.py --working_dir=/home/lesh/dev/ssb/ --grid_info=100_0_100 --file_shape=100_100