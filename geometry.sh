#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates tetrahedral, triangle and line xdmf mesh files from images
        given the bmp files parent directory, sub directory, shape of image file data array,
        and grid information (grid_size,)
        ]
        Example: ./geometry.sh /home/ubuntu/dev/ bmp_files 30.30.30
        "
  exit 0
fi

if [ $# -ne 3 ]
  then
    echo Error: "requires 3 arguments"
    exit 1
fi
echo "Making required sub directories: mesh/"

mkdir -p $1mesh/

./create_node_files.py --working_dir=$1 --img_sub_dir=$2 --grid_info=$3

tetgen $1mesh/$3.node -akEFNQIRB
grid_sizes=($(echo $3 | tr '.' "\n"))

sed '1 i size = '$grid_size';' $1porous-solid.geo | tee $1mesh/$3.geo

gmsh -3 $1mesh/$3.geo -o $1mesh/$3.msh

./create_xdmf_meshfiles.py --input_meshfile=$1mesh/$3.msh
