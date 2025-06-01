#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "Error: No arguments provided."
  echo "Usage: $0 Nx-Ny-Nz"
  exit 1
fi

python3 volsurf.py --size $1 --scale 0.08e-6,0.08e-6,0.2e-6 --L_sep 15e-6

meshio ascii output/segmentation/$1/0-0-0/voids.stl
tetgen -pkAY output/segmentation/$1/0-0-0/voids.stl
sed -i '/#/d' output/segmentation/$1/0-0-0/voids.1.node
sed -i '/#/d' output/segmentation/$1/0-0-0/voids.1.face
sed -i '/#/d' output/segmentation/$1/0-0-0/voids.1.ele

meshio ascii output/segmentation/$1/0-0-0/cam.stl
tetgen -pkAY output/segmentation/$1/0-0-0/cam.stl
sed -i '/#/d' output/segmentation/$1/0-0-0/cam.1.node
sed -i '/#/d' output/segmentation/$1/0-0-0/cam.1.face
sed -i '/#/d' output/segmentation/$1/0-0-0/cam.1.ele

meshio ascii output/segmentation/$1/0-0-0/sse.stl
tetgen -pkAY output/segmentation/$1/0-0-0/sse.stl
sed -i '/#/d' output/segmentation/$1/0-0-0/sse.1.node
sed -i '/#/d' output/segmentation/$1/0-0-0/sse.1.face
sed -i '/#/d' output/segmentation/$1/0-0-0/sse.1.ele

python3 mesher.py --size $1 --scale 0.08e-6,0.08e-6,0.2e-6 --L_sep 15e-6
