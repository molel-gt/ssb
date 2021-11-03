#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates tetrahedral, triangle and line xdmf mesh files from images
        given the bmp files parent directory, sub directory, and the shape of image file data array]
        Example: ./geometry.sh /home/ubuntu/dev/ bmp_files 5,5
        "
  exit 0
fi

if [ $# -ne 4 ]
  then
    echo Error: "requires three arguments"
    exit 1
fi

./create_node_files.py --working_dir=$1 --img_sub_dir=$2 --file_shape=$3 --grid_size=$4

tetgen $1porous-solid.node -ak

sed '1 i size = '$4';' $1porous-solid.geo >> $1porous-solid-$4.geo

gmsh -3 $1porous-solid-$4.geo -o $1porous-solid.msh

./create_xdmf_meshfiles.py --input_meshfile=$1porous-solid.msh