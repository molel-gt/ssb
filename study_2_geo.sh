#!/bin/bash

for img_id in 1 6 11 16 22;
do
  sbatch study_2_geo.sbatch $img_id
done