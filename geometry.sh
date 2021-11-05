#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates tetrahedral, triangle and line xdmf mesh files from images
        given the bmp files parent directory, sub directory, shape of image file data array,
        and grid information (grid_size,)
        ]
        Example: ./geometry.sh /home/ubuntu/dev/ bmp_files 5_5 30_0_30
        "
  exit 0
fi

if [ $# -ne 4 ]
  then
    echo Error: "requires 4 arguments"
    exit 1
fi
echo "Making required sub directories: mesh/$4/"

mkdir -p $1mesh/$4/

./create_node_files.py --working_dir=$1 --img_sub_dir=$2 --file_shape=$3 --grid_info=$4

tetgen $1mesh/$4/porous-solid.node -akEFNQI

grid_size=`echo $4 | cut -d \_ -f 1`

sed --quiet '1 i size = '$grid_size';' $1porous-solid.geo | tee $1mesh/$4/porous-solid.geo

gmsh -3 $1mesh/$4/porous-solid.geo -o $1mesh/$4/porous-solid.msh

./create_xdmf_meshfiles.py --input_meshfile=$1mesh/$4/porous-solid.msh
