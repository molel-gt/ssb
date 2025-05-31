#!/bin/bash

python3 volsurf.py --size 201-201-201 --scale 0.0858e-6,0.0858e-6,0.2e-6 --L_sep 15e-6
tetgen -pk output/segmentation/201-201-201/0-0-0/voids.stl
tetgen -pk output/segmentation/201-201-201/0-0-0/sse.stl
tetgen -pk output/segmentation/201-201-201/0-0-0/cam.stl
python3 mesher.py --size 201-201-201 --scale 0.0858e-6,0.0858e-6,0.2e-6 --L_sep 15e-6
