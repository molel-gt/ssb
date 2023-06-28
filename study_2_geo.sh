#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "usage: sh study_2_geo.sh L_z resolution"
    exit 1
fi

for img_id in 1 6 11 16 22;
do
  sbatch study_2_geo.sbatch $img_id $1 $2
done