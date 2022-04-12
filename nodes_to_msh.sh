#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [generates msh file from node file using tetgen and gmsh]
        Example: ./build_nodes_to_msh.sh s101-101-101o0-0-0.node 101-101-101o0-0-0.vtk s101-101-101o0-0-0.geo s101-101-101o0-0-0.msh
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
vtk_file=$(echo "$3" | sed "s/.*\///")

sed '1 i file_name = \"'$vtk_file'\";' porous.geo | tee $2

gmsh -3 $2 -o $4 -optimize_netgen
