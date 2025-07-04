#!/bin/bash

for stl_file in $WORK_DIR/output/segmentation/cam/201-201-201/0-0-0/aggs/*.stl
do
    echo 'Repairing stl file '$stl_file
    python3 repair_stl_mesh.py $stl_file
done
