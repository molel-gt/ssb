#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates msh file from node file using tetgen and gmsh]
        Example: ./build_nodes_to_msh.sh 101-101-101.node 101-101-101.vtk 101-101-101.geo 101-101-101.msh
        "
  exit 0
fi

if [ $# -ne 4 ]
  then
    echo Error: "requires 4 arguments"
    exit 1
fi

cd "$(dirname "$0")"

tetgen $1 -akEFNQIRB

sed '1 i file_name = \"'$2'\";' porous-solid.geo | tee $3

gmsh -3 $3 -o $4