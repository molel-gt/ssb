#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates tetrahedral, triangle and line xdmf mesh files from images
        given the bmp_files parent directory and shape of image file data array]
        Example: ./geometry.sh /home/ubuntu/dev/ 5,5
        "
  exit 0
fi

if [ $# -ne 2 ]
  then
    echo Error: "No arguments supplied"
    exit 1
fi

./create_node_files.py --working_dir=$1 --file_shape=$2

tetgen $1porous-solid.node -ak

gmsh -3 $1porous-solid.geo -o $1porous-solid.msh

./create_xdmf_meshfiles.py --input_meshfile=$1porous-solid.msh