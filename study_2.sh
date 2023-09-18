#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "requires L_z"
    exit 1
fi
for img_id in 6 11 16 22;
do
  sbatch study_2.sbatch  $img_id $1
done