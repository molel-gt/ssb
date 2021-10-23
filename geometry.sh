#!/usr/bin/env bash

./create_node_files.py --files_dir=/home/lesh/dev/ssb/NavneetGeometry2/ --file_shape=90,90

tetgen porous-cathode.node -ak

gmsh -3 porous-cathode.geo -o porous-cathode.msh

./create_xdmf_meshfiles.py --input_meshfile=/home/lesh/dev/ssb/porous-cathode.gmsh