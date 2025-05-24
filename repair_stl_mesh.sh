#!/bin/bash

for stl_file in ls $WORK_DIR/output/segmentation/cam/201-201-201/0-0-0/aggs/*.stl
do
    python3 repair_stl_mesh.py $stl_file
done
