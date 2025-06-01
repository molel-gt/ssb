#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "Error: No arguments provided."
  echo "Usage: $0 Nx-Ny-Nz"
  exit 1
fi

python3 volsurf.py --size $1 --scale 0.0858e-6,0.0858e-6,0.2e-6 --L_sep 15e-6
tetgen -pkAY output/segmentation/$1/0-0-0/voids.stl
tetgen -pkAY output/segmentation/$1/0-0-0/cam.stl
tetgen -pkAY output/segmentation/$1/0-0-0/sse.stl
python3 mesher.py --size $1 --scale 0.0858e-6,0.0858e-6,0.2e-6 --L_sep 15e-6
