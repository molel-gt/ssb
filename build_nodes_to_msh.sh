#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates tetrahedral, triangle and line xdmf mesh files from images
        given the bmp files parent directory, sub directory, shape of image file data array,
        and grid information (grid_size,)
        ]
        Example: ./geometry.sh /home/ubuntu/dev/ssb/mesh/101-101-101.node 101 30.30.30
        "
  exit 0
fi

if [ $# -ne 3 ]
  then
    echo Error: "requires 3 arguments"
    exit 1
fi
echo "Making required sub directories: mesh/"


tetgen $1 -akEFNQIRB

sed '1 i file_name = \"'$2'\";' porous-solid.geo | tee $3

gmsh -3 $3 -o $4