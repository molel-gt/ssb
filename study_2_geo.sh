#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "requires L_z"
    exit 1
fi

for img_id in 1 6 11 16 22;
do
  sbatch study_2_geo.sbatch $img_id $1
done